# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black; `Reader` calls mlx-vlm's processors and
# returns MLX inputs; video paths decode with mlx-vlm's loader; `resampled` in
# NumPy; the torch joins removed.

"""Media: pictures, audio clips and videos, each placed in a prompt as its checkpoint's processor expands it.

    media=[picture, {"type": "audio", "path": "call.wav"}, {"type": "video", "path": "clip.mp4"}]
    {"type": "noul", "instructions": {"type": "audio", "path": "question.wav"}}   # a spoken question

- An item is a PIL image (a picture), or `{"type": "image" | "audio" | "video", "path": ...}`, the decoded value under
  its type instead of a path (`{"type": "audio", "audio": waveform, "sampling_rate": 16000}`, a PIL image, a video's
  frames with their `fps`). A spoken state is `state=None` with its clip in `media`.
- A request's media head its state, in order, as the chat template writes them; a question's spoken instructions sit
  where its text would. Each item's markup is the processor's own expansion, so a prompt is plain text and is packed,
  read both ways and pointed into as a text prompt is (`Reader`).
- Pictures are capped at the checkpoint's pixel budget first (`cap_pixels`); audio is read as 16-bit PCM WAV or a
  NumPy file and resampled to the feature extractor's rate; a decoded stereo waveform may be (channels, samples) or
  (samples, channels), as soundfile returns it, and its channels are averaged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KINDS = ("image", "audio", "video")
PROCESSORS = {
    "image": "image_processor",
    "audio": "feature_extractor",
    "video": "video_processor",
}
INPUTS = {
    "image": "images",
    "audio": "audio",
    "video": "videos",
}  # the processor's argument for each kind
SPOKEN = ""  # a spoken question's place in its suffix until its markup replaces it (`Reader.place`)


@dataclass(frozen=True, eq=False)
class Media:
    """One picture, audio clip or video: a path, or its decoded value (`sampling_rate`: a waveform's, `fps`: a video's
    frames')."""

    type: str
    value: Any
    sampling_rate: int | None = None
    fps: float | None = None


def as_media(item: Any) -> Media:
    """A caller's item (a PIL image, or a dict with its `type`), as `Media`; `Media` passes through."""
    if isinstance(item, Media):
        return item
    if isinstance(item, Mapping):
        kind = item.get("type")
        if kind not in KINDS or kind not in item and "path" not in item:
            raise ValueError(
                f"a media item is a picture or {{'type': one of {KINDS}, 'path' or <type>: its value}}"
            )
        return Media(
            kind,
            item[kind] if kind in item else item["path"],
            item.get("sampling_rate"),
            item.get("fps"),
        )
    if hasattr(item, "convert") and hasattr(item, "size"):  # a PIL image
        return Media("image", item)
    raise TypeError(
        f"a media item is a picture or a dict with its type, not {type(item).__name__}"
    )


def media_list(items: Sequence | None) -> list[Media]:
    return [as_media(x) for x in items or ()]


def spoken(q) -> Media | None:
    """A question's spoken instructions, or None for written ones."""
    return q.instructions if isinstance(q.instructions, Media) else None


def cap_pixels(image, max_pixels: int):
    """A picture downscaled to at most `max_pixels`, bicubic."""
    image = image.convert("RGB") if hasattr(image, "convert") else image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    from PIL import Image

    scale = math.sqrt(max_pixels / (w * h))
    return image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC
    )


class Reader:
    """One checkpoint's media: each item's markup and processor outputs (`encode`)."""

    def __init__(self, processor, max_pixels: int):
        self.processor, self.max_pixels = processor, max_pixels
        self._texts: dict[tuple[int, ...], str] = (
            {}
        )  # an expansion's ids, checked to read back, as text

    def encode(self, items: Sequence[Media]) -> tuple[str, list[dict]]:
        """The items' markup back to back, each the template's with its placeholder expanded by the processor, and
        each item's processor outputs (with its `type`)."""
        encoded = [self._encode(m) for m in items]
        return "".join(text for text, _ in encoded), [out for _, out in encoded]

    def place(self, suffix: str, q) -> tuple[str, list[dict]]:
        """A question's suffix with its spoken instructions' markup in place of `SPOKEN`, and their outputs."""
        voice = spoken(q)
        if voice is None:
            return suffix, []
        markup, parts = self.encode([voice])
        return suffix.replace(SPOKEN, markup), parts

    def _encode(self, item: Media) -> tuple[str, dict]:
        """One item's expanded markup, as text that encodes back to the processor's ids, and its outputs."""
        if getattr(self.processor, PROCESSORS[item.type], None) is None:
            raise ValueError(f"this checkpoint takes no {item.type}")
        template = template_markup(self.processor, item.type)
        value, extra = self._decoded(item)
        out = self.processor(
            text=[template],
            **{INPUTS[item.type]: [value]},
            **extra,
            add_special_tokens=False,
        )
        ids = tuple(int(i) for i in out["input_ids"][0].tolist())
        if (
            ids not in self._texts
        ):  # one check an expansion (a picture size, a clip length), not one a call
            tokenizer = self.processor.tokenizer
            text = tokenizer.decode(ids)
            if tuple(tokenizer.encode(text, add_special_tokens=False)) != ids:
                raise ValueError(
                    f"{type(self.processor).__name__}'s {item.type} markup does not read back as text"
                )
            if len(self._texts) >= 4096:
                self._texts.clear()
            self._texts[ids] = text
        out = {k: v for k, v in out.items() if k not in ("input_ids", "attention_mask")}
        return self._texts[ids], {"type": item.type, **out}

    def _decoded(self, item: Media) -> tuple[Any, dict]:
        """What the processor takes, and its keyword arguments: a capped RGB picture, a waveform at the feature
        extractor's rate, a video's frames sampled as the checkpoint's video processor samples them.
        """
        if item.type == "image":
            from PIL import Image

            image = (
                Image.open(item.value) if isinstance(item.value, str) else item.value
            )
            return cap_pixels(image, self.max_pixels), {}
        if item.type == "audio":
            rate = self.processor.feature_extractor.sampling_rate
            wave, given = (
                read_audio(item.value, rate)
                if isinstance(item.value, str)
                else (item.value, item.sampling_rate)
            )
            if given is None:
                raise ValueError("a waveform needs its sampling_rate")
            return resampled(wave, given, rate), {}
        sampler = self.processor.video_processor
        if isinstance(item.value, str):  # a path is decoded with its own clock
            from ...utils import load_video

            frames, metadata = load_video(
                item.value, fps=sampler.fps, frame_sampler=sampler.sample_frames
            )
            return frames, {"video_metadata": [metadata]}
        if item.fps is None:
            raise ValueError("a video's frames need their fps")
        from types import SimpleNamespace

        import numpy as np

        frames = np.asarray(item.value)
        if (
            frames.ndim == 4
            and frames.shape[-1] in (1, 3)
            and frames.shape[1] not in (1, 3)
        ):
            frames = frames.transpose(0, 3, 1, 2)  # (T, H, W, C) as decoders give them
        indices = sampler.sample_frames(
            SimpleNamespace(total_num_frames=len(frames), fps=item.fps)
        ).tolist()
        return frames[indices], {
            "video_metadata": [{"fps": item.fps, "frames_indices": indices}]
        }


def read_audio(path: str, rate: int):
    """A clip's waveform and its rate: 16-bit PCM WAV, or a NumPy file (int16 or float, at `rate`), with the standard
    library and NumPy; any other format through transformers' `load_audio` (torchcodec or librosa), at `rate`.
    """
    import numpy as np

    if path.endswith(".wav"):
        import wave

        with wave.open(path, "rb") as f:
            if f.getsampwidth() != 2:
                raise ValueError(f"{path}: 16-bit PCM WAV only")
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").reshape(
                -1, f.getnchannels()
            )
            return pcm.T.astype(np.float32) / 32768, f.getframerate()
    if path.endswith(".npy"):
        data = np.load(path)
        return (
            data.astype(np.float32) / 32768
            if data.dtype == np.int16
            else data.astype(np.float32)
        ), rate
    from transformers.audio_utils import load_audio

    return load_audio(path, sampling_rate=rate), rate


def resampled(wave, rate: int, target: int, zeros: int = 6, rolloff: float = 0.99):
    """A mono float32 waveform at `target` (channels averaged), resampled as torchaudio's `functional.resample` does
    by default: a Hann-windowed sinc over `zeros` zero crossings, cut at `rolloff` of the lower Nyquist, one filter per
    output phase."""
    import numpy as np

    wave = np.asarray(wave, dtype=np.float32)
    if (
        wave.ndim == 2
    ):  # (channels, samples) or (samples, channels): a clip never has more channels than samples
        wave = (wave.T if wave.shape[0] > wave.shape[1] else wave).mean(0)
    if rate == target:
        return wave
    gcd = math.gcd(int(rate), int(target))
    old, new = int(rate) // gcd, int(target) // gcd
    cut = min(old, new) * rolloff
    width = math.ceil(zeros * old / cut)
    t = np.arange(0, -new, -1, dtype=np.float32)[:, None] / np.float32(new)
    t = (
        t + np.arange(-width, width + old, dtype=np.float32)[None] / np.float32(old)
    ) * np.float32(cut)
    t = t.clip(-zeros, zeros).astype(np.float32)
    window = np.cos(t * np.float32(math.pi) / np.float32(zeros) / np.float32(2)) ** 2
    t = t * np.float32(math.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(t == 0, np.float32(1.0), np.sin(t) / t) * (
            window * np.float32(cut / old)
        )
    x = np.pad(wave, (width, width + old))
    frames = np.lib.stride_tricks.sliding_window_view(x, kernel.shape[1])[::old]
    out = (frames @ kernel.T.astype(np.float32)).reshape(-1)
    return out[: math.ceil(new * len(wave) / old)].astype(np.float32)


def template_markup(processor, kind: str) -> str:
    """One item's markup as the processor's chat template writes it at the head of a user turn."""
    messages = [
        {"role": "user", "content": [{"type": kind}, {"type": "text", "text": "\x00"}]}
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    head = "<|im_start|>user\n"
    return text[text.index(head) + len(head) : text.index("\x00")]
