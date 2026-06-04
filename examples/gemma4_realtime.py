"""Realtime voice chat with Gemma 4 (local, MLX) — talk and it replies.

Audio is ingested into a PERSISTENT KV cache in 200 ms micro-turns *while you
speak*, so the reply starts the moment you stop — no end-of-utterance re-prefill.
Exploits Gemma 4's encoder-free audio (a 200 ms chunk = 5×640 raw samples → one
linear projection → 5 tokens appended).

    python examples/gemma4_realtime.py                 # live mic (default)
    python examples/gemma4_realtime.py --audio q.wav   # one-shot from a file
"""

import argparse
import time

import numpy as np
from mlx_vlm import load
from mlx_vlm.realtime import RealtimeSession


def _snap(cap):
    """Grab one RGB frame from an open cv2 capture, or None."""
    import cv2
    from PIL import Image

    ok, frame = cap.read()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None


def run_mic(
    sess, camera=False, sr=16000, frame=3200, vad_on=0.02, vad_off=0.012, silence=0.6
):
    """Listen on the mic; on each utterance (energy VAD), stream the reply.
    With camera=True, one frame is captured at speech onset and shown first."""
    import sounddevice as sd

    cap = None
    if camera:
        import cv2

        cap = cv2.VideoCapture(0)
    print(
        f"🎤 listening{' + 👁 camera' if camera else ''} — just talk (Ctrl-C to quit)\n"
    )
    speaking, last_voice = False, 0.0
    with sd.InputStream(
        samplerate=sr, channels=1, blocksize=frame, dtype="float32"
    ) as stream:
        while True:
            block, _ = stream.read(frame)
            pcm = block[:, 0]
            rms = float(np.sqrt(np.mean(pcm**2)))
            now = time.time()
            if rms > vad_on:
                if not speaking:
                    speaking = True
                    sess.start_turn(image=_snap(cap) if cap is not None else None)
                    print("…you: ", end="", flush=True)
                last_voice = now
                sess.append_audio(pcm)
            elif speaking:
                sess.append_audio(pcm)
                if now - last_voice > silence:  # end of utterance
                    speaking = False
                    t = time.time()
                    print("\rGemma: ", end="", flush=True)
                    first = None
                    for piece in sess.respond():
                        if first is None:
                            first = time.time() - t
                        print(piece, end="", flush=True)
                    print(f"   ({first*1000:.0f} ms to first token)\n")


def run_file(sess, path, image=None):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    sess.start_turn(image=image)
    frame = 3200
    for s in range(0, len(wav) - len(wav) % frame, frame):
        sess.append_audio(wav[s : s + frame])
    print("Gemma: ", end="", flush=True)
    for piece in sess.respond():
        print(piece, end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-12B-it-4bit")
    ap.add_argument(
        "--audio", default=None, help="one-shot from a file instead of the mic"
    )
    ap.add_argument(
        "--image", default=None, help="(file mode) image shown before the question"
    )
    ap.add_argument(
        "--camera", action="store_true", help="(mic mode) snap a webcam frame per turn"
    )
    args = ap.parse_args()

    model, processor = load(args.model)
    sess = RealtimeSession(model, processor)

    if args.audio:
        image = None
        if args.image:
            from PIL import Image

            image = Image.open(args.image).convert("RGB")
        run_file(sess, args.audio, image)
    else:
        try:
            run_mic(sess, camera=args.camera)
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
