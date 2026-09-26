"""Omni prefix regression with synthetic images/videos and locally spoken audio.

Requires macOS `say` (Samantha) and ffmpeg. Run on an otherwise idle Apple GPU.
"""

import argparse
import gc
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from PIL import Image

from mlx_vlm import load, stream_generate
from mlx_vlm.apc import APCManager


def fixtures(root):
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        raise RuntimeError("This example requires macOS say and ffmpeg")
    for color in ("red", "green", "blue"):
        Image.new("RGB", (128, 128), color).save(root / f"{color}.png")
    for name, words in (
        ("one", "The first code word is apple."),
        ("two", "The second code word is banana. Please remember banana."),
        ("changed", "The first code word is orange."),
    ):
        subprocess.run(
            ["say", "-v", "Samantha", "-o", str(root / f"{name}.aiff"), words],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(root / f"{name}.aiff"),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(root / f"{name}.wav"),
            ],
            check=True,
        )
    for color, seconds in (("red", 2), ("blue", 3)):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=224x224:r=2:d={seconds}",
                "-pix_fmt",
                "yuv420p",
                str(root / f"{color}.mp4"),
            ],
            check=True,
        )
    for color, name in (("red", "one"), ("blue", "two")):
        av = root / f"{color}-av.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(root / f"{color}.mp4"),
                "-i",
                str(root / f"{name}.wav"),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-t",
                "2" if color == "red" else "3",
                str(av),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(av),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(root / f"{name}-av.wav"),
            ],
            check=True,
        )
    for name in ("one", "changed"):
        samples, _ = sf.read(root / f"{name}.wav", dtype="float32")
        samples = np.pad(samples[:32000], (0, max(0, 32000 - len(samples))))
        sf.write(root / f"{name}-fixed.wav", samples, 16000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="omni-prefix-") as directory:
        root = Path(directory)
        fixtures(root)
        model, processor = load("mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit")
        manager = APCManager(overrides={"memory_max_gb": 4})
        vision, audio = [], []
        vc, ac = type(model.thinker.vision_tower), type(model.thinker.audio_tower)
        original_v, original_a = vc.__call__, ac.__call__

        def vision_call(self, pixels, *a, **kw):
            vision.append(int(pixels.shape[0]))
            return original_v(self, pixels, *a, **kw)

        def audio_call(self, features, *a, **kw):
            audio.append(kw["feature_lens"].tolist())
            return original_a(self, features, *a, **kw)

        vc.__call__, ac.__call__ = vision_call, audio_call
        system = {
            "role": "system",
            "content": "Answer concisely. These reference records are unrelated background. "
            + " ".join(
                f"Record {i}: the inventory contains {i+1} ordinary items."
                for i in range(180)
            ),
        }

        def message(parts, text):
            return {
                "role": "user",
                "content": [
                    {"type": kind, kind: str(root / name)} for kind, name in parts
                ]
                + [{"type": "text", "text": text}],
            }

        def run(label, messages, parts, cache=True, fps=1.0):
            # Omni does not use Qwen3.5's empty thinking prefill.
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images = [str(root / n) for k, n in parts if k == "image"]
            videos = [str(root / n) for k, n in parts if k == "video"]
            audios = [
                sf.read(root / n, dtype="float32")[0] for k, n in parts if k == "audio"
            ]
            vision.clear()
            audio.clear()
            start = time.perf_counter()
            first = log = None
            texts = []
            for result in stream_generate(
                model,
                processor,
                prompt,
                image=images or None,
                video=videos or None,
                audio=audios or None,
                fps=fps,
                max_tokens=48,
                temperature=0,
                apc_manager=manager if cache else None,
                apc_media_prefix=True,
            ):
                if first is None:
                    first = time.perf_counter() - start
                    log = result.logprobs.astype(mx.float32)
                    log = log - mx.logsumexp(log)
                    mx.eval(log)
                texts.append(result.text)
            row = {
                "case": label,
                "ttft_s": first,
                "text": "".join(texts),
                "prompt_tokens": result.prompt_tokens,
                "cached_tokens": result.cached_tokens,
                "vision_rows": list(vision),
                "audio_frames": list(audio),
                "resident_bytes": manager.resident_bytes(),
                "peak_bytes": mx.get_peak_memory(),
            }
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            assert row["resident_bytes"] <= manager.memory_max_bytes
            return row, log

        question = "What are the code words spoken in the two audio clips, in order? Reply with the two words only."
        cases = [
            (
                "image",
                [("image", "red.png")],
                [("image", "green.png")],
                "List the colors of both images in order, using only color words.",
                ("red", "green"),
            ),
            (
                "audio",
                [("audio", "one.wav")],
                [("audio", "two.wav")],
                question,
                ("apple", "banana"),
            ),
            (
                "video",
                [("video", "red.mp4")],
                [("video", "blue.mp4")],
                "List the dominant colors of both videos in order, using only color words.",
                ("red", "blue"),
            ),
            (
                "mixed",
                [("image", "red.png"), ("video", "red.mp4"), ("audio", "one.wav")],
                [("video", "blue.mp4"), ("audio", "two.wav")],
                question,
                ("apple", "banana"),
            ),
            (
                "audiovisual",
                [("video", "red-av.mp4"), ("audio", "one-av.wav")],
                [("video", "blue-av.mp4"), ("audio", "two-av.wav")],
                question,
                ("apple", "banana"),
            ),
        ]
        try:
            run("warmup", [{"role": "user", "content": "Say hello."}], [])
            for repeat in range(args.repeats):
                for label, old, new, q, expected in cases:
                    manager.clear()
                    base = [system, message(old, "Please inspect these inputs.")]
                    run(f"{label}_{repeat}_past", base, old)
                    extended = [
                        *base,
                        {"role": "assistant", "content": "Understood."},
                        message(new, q),
                    ]
                    hit, h = run(f"{label}_{repeat}_hit", extended, old + new)
                    cold, c = run(f"{label}_{repeat}_cold", extended, old + new, False)
                    kl = mx.sum(mx.exp(c) * (c - h)).item()
                    print(
                        json.dumps({"case": label, "repeat": repeat, "KL": kl}),
                        flush=True,
                    )
                    assert hit["cached_tokens"] > 0
                    assert len(hit["audio_frames"]) == sum(k == "audio" for k, _ in new)
                    assert (
                        sum(hit["vision_rows"]) < sum(cold["vision_rows"])
                        if any(k != "audio" for k, _ in old)
                        else True
                    )
                    for row in (hit, cold):
                        text = row["text"].lower()
                        assert all(w in text for w in expected), row
                        assert text.index(expected[0]) < text.index(expected[1]), row

            # Change audio bytes at the same path and keep the exact sample count.
            shutil.copyfile(root / "one-fixed.wav", root / "mutable.wav")
            old = [("audio", "mutable.wav")]
            new = [("audio", "two.wav")]
            manager.clear()
            base = [system, message(old, "Please inspect this clip.")]
            prior, _ = run("audio_before_change", base, old)
            shutil.copyfile(root / "changed-fixed.wav", root / "mutable.wav")
            ext = [
                *base,
                {"role": "assistant", "content": "Understood."},
                message(new, question),
            ]
            changed, _ = run("audio_changed", ext, old + new)
            control, _ = run("audio_changed_cold", ext, old + new, False)
            assert changed["cached_tokens"] < prior["prompt_tokens"] - 26
            assert len(changed["audio_frames"]) == 2
            for row in (changed, control):
                assert (
                    "orange" in row["text"].lower() and "banana" in row["text"].lower()
                ), row

            # FPS changes invalidate old video semantics, even with static frames.
            manager.clear()
            old = [("video", "red.mp4")]
            base = [system, message(old, "What color is the video?")]
            prior, _ = run("video_fps_base", base, old, fps=1)
            changed, _ = run("video_fps_changed", base, old, fps=2)
            assert changed["cached_tokens"] < prior["prompt_tokens"] - 98
            assert changed["vision_rows"]
            manager.clear()
            assert manager.resident_bytes() == 0
        finally:
            vc.__call__, ac.__call__ = original_v, original_a
            manager.clear()
            manager.close()
        del model, manager, h, c, _
        gc.collect()
        mx.clear_cache()
        print(json.dumps({"released_active_bytes": mx.get_active_memory()}), flush=True)


if __name__ == "__main__":
    main()
