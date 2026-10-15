"""Torch-free processor for Youtu-Parsing-Omni (``model_type == "youtu_vita"``), image path.

Re-implements ``YoutuVITAImageProcessor.process_native_v2`` /
``add_image_input_discrete_or_contiguous`` from the HF remote code with PIL + numpy:

* an image is only resized when it is outside ``[min_pixels, max_pixels]``
  (aspect ratio kept); otherwise its size is left untouched;
* it is then pasted at the top-left of a canvas whose sides are rounded up to a
  multiple of ``patch_size * spatial_merge_size`` (32), padded with
  ``int(mean * 255)`` = (127, 127, 127) -- right/bottom padding only;
* pixels are ``(x / 255 - mean) / std`` (mean = std = 0.5, "siglip"), float32;
* patches are emitted in 2x2-spatial-merge order as rows of ``3 * 16 * 16``;
* the ``<|image|>`` tag in the prompt becomes
  ``<|vision_start|> "H*W" \\n  (rows of <|image_pad|> * (gw/2) + \\n) * (gh/2)  <|vision_end|>``
  and ``image_indices`` ([2, N]: batch index, absolute sequence position) records
  where the vision features have to be scattered.

Audio (``<|audio|>`` tag) re-implements ``MelFilterBankTokenizer.encode`` +
``YoutuVITAFeatureExtractor.add_audio_input_discrete_or_contiguous``:
mono 16 kHz audio (torchaudio-style sinc resampling, numpy) -> Whisper log-mel
(128 bins, n_fft 400, hop 160) -> split into <= 30 s clips -> per clip
``[HH:MM:SS]`` timestamp text (only when there is more than one clip) +
``<|audio_start|> <|audio_pad|> * n <|audio_end|>`` and ``audio_indices``.

Video (``<|video|>`` tag) re-implements ``YoutuVITAVideoProcessor.add_video_input_discrete_or_contiguous``
(PyAV decode, torch-free): frames are sampled with step ``max(round(fps) / video_max_fps, n / max_frames)``,
each frame goes through the image path (resize into the per-frame pixel budget, grey pad to a multiple of 32);
the audio track is extracted, turned into one log-mel for the whole video and cut at the frame timestamps; frames
and cut audio are greedily grouped into chunks of >= ``video_audio_chunk_min_second`` of audio. Prompt:
``<|video_start|>`` then per chunk [``HH:MM:SS<|vision_start|>H*W`` + (rows of ``<|image_pad|>`` + ``\n``) +
``<|vision_end|>``] for each frame followed by ``HH:MM:SS<|audio_start|><|audio_pad|>*n<|audio_end|>`` for the chunk's
audio, then ``<|video_end|>``. Model inputs: ``video_images``, ``video_image_grid_thw``, ``video_image_indices``,
``video_audios``, ``video_audio_indices``, ``video_split``.

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch, load_chat_template

IMG_TAG_TOKEN = "<|image|>"
IMG_CONTEXT_TOKEN = "<|image_pad|>"
IMG_START_TOKEN = "<|vision_start|>"
IMG_END_TOKEN = "<|vision_end|>"

VID_TAG_TOKEN = "<|video|>"
VID_START_TOKEN = "<|video_start|>"
VID_END_TOKEN = "<|video_end|>"

AUD_TAG_TOKEN = "<|audio|>"
AUD_CONTEXT_TOKEN = "<|audio_pad|>"
AUD_START_TOKEN = "<|audio_start|>"
AUD_END_TOKEN = "<|audio_end|>"

_NORMALIZE = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


class YoutuVitaImageProcessor:
    """Native-resolution image preprocessor (numpy/PIL, no torch).

    Deliberately NOT a ``mlx_vlm.models.base.BaseImageProcessor`` so that
    ``prepare_inputs`` calls the full processor instead of the generic
    ``<image>``-split path.
    """

    model_input_names = ["images", "image_indices", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        image_min_num_tokens: int = 1,
        image_max_num_tokens: int = 32768,
        vision_normalize_type: str = "siglip",
        **kwargs,
    ):
        if vision_normalize_type not in _NORMALIZE:
            raise NotImplementedError(vision_normalize_type)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_min_num_tokens = image_min_num_tokens
        self.image_max_num_tokens = image_max_num_tokens
        self.vision_normalize_type = vision_normalize_type
        self.mean, self.std = _NORMALIZE[vision_normalize_type]
        unit = (patch_size * spatial_merge_size) ** 2
        self.min_pixels = unit * image_min_num_tokens
        self.max_pixels = unit * image_max_num_tokens

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]):
        cfg = dict(cfg or {})
        keys = (
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "image_min_num_tokens",
            "image_max_num_tokens",
            "vision_normalize_type",
        )
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    # ------------------------------------------------------------------ pixels
    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return image.convert("RGB")

    def resize_and_pad(self, image: Image.Image, min_pixels=None, max_pixels=None):
        """Returns (canvas, padded_h, padded_w, resized_h, resized_w)."""
        width, height = image.size
        min_pixels = self.min_pixels if min_pixels is None else min_pixels
        max_pixels = self.max_pixels if max_pixels is None else max_pixels
        factor = self.patch_size * self.spatial_merge_size

        resized_height, resized_width = height, width
        cur_pixels = resized_height * resized_width
        if cur_pixels > max_pixels:
            beta = math.sqrt(cur_pixels / max_pixels)
            resized_height = max(1, int(math.floor(height / beta)))
            resized_width = max(1, int(math.floor(width / beta)))
        elif cur_pixels < min_pixels:
            beta = math.sqrt(min_pixels / cur_pixels)
            resized_height = max(1, int(math.ceil(height * beta)))
            resized_width = max(1, int(math.ceil(width * beta)))

        padded_height = math.ceil(resized_height / factor) * factor
        padded_width = math.ceil(resized_width / factor) * factor
        if padded_height * padded_width > max_pixels:
            padded_height = max(factor, math.floor(resized_height / factor) * factor)
            padded_width = max(factor, math.floor(resized_width / factor) * factor)
            scale = min(padded_height / resized_height, padded_width / resized_width)
            resized_height = max(
                1, min(padded_height, int(round(resized_height * scale)))
            )
            resized_width = max(1, min(padded_width, int(round(resized_width * scale))))

        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BICUBIC
        )
        background = tuple(int(x * 255) for x in self.mean)
        canvas = Image.new("RGB", (padded_width, padded_height), background)
        canvas.paste(image, (0, 0))
        return canvas, padded_height, padded_width, resized_height, resized_width

    def normalize(self, canvas: Image.Image) -> np.ndarray:
        # Same float32 op sequence as the HF code: x * 1.0 / 255.0, (x - mean) / std
        x = np.array(canvas, dtype=np.float32)
        x = x * 1.0 / 255.0
        mean = np.array(self.mean, dtype=x.dtype)
        std = np.array(self.std, dtype=x.dtype)
        return (x - mean) / std  # [H, W, 3]

    def patchify(self, x: np.ndarray) -> np.ndarray:
        """[H, W, 3] -> [gh * gw, 3 * tps * p * p] in 2x2 spatial-merge order."""
        p, m, tps = self.patch_size, self.spatial_merge_size, self.temporal_patch_size
        if tps != 1:
            raise NotImplementedError("temporal_patch_size != 1 (single image only)")
        H, W, C = x.shape
        gh, gw = H // p, W // p
        x = x.reshape(gh // m, m, p, gw // m, m, p, C)
        # (gh/m, gw/m, m_h, m_w, C, p_h, p_w)  == HF permute (0,3,6,4,7,2,1,5,8) for tps=1
        x = x.transpose(0, 3, 1, 4, 6, 2, 5)
        return np.ascontiguousarray(x.reshape(gh * gw, C * p * p))

    def process_image(self, image, min_pixels=None, max_pixels=None):
        canvas, ph, pw, rh, rw = self.resize_and_pad(
            self._to_pil(image), min_pixels, max_pixels
        )
        patches = self.patchify(self.normalize(canvas))
        grid = [1, ph // self.patch_size, pw // self.patch_size]
        return patches, grid, (ph, pw, rh, rw)


# ====================================================================== audio
def audio_token_length(n_frames: int) -> int:
    """HF ``_get_feat_extract_output_lengths`` (mel frames -> audio tokens)."""
    leave = n_frames % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    logstep = 27.0 / math.log(6.4)
    return np.where(
        f >= 1000.0,
        15.0 + np.log(np.maximum(f, 1e-30) / 1000.0) * logstep,
        3.0 * f / 200.0,
    )


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    logstep = math.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), 200.0 * m / 3.0)


def slaney_mel_filters(
    n_freq=201, n_mels=128, sr=16000, fmin=0.0, fmax=8000.0
) -> np.ndarray:
    """[n_freq, n_mels]; identical to transformers.audio_utils.mel_filter_bank(norm='slaney', 'slaney')."""
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    filt = _mel_to_hz(mel_pts)
    fft_freqs = np.linspace(0, sr // 2, n_freq)
    diff = np.diff(filt)
    slopes = filt[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    return fb * (2.0 / (filt[2 : n_mels + 2] - filt[:n_mels]))[None, :]


class WhisperLogMel:
    """Whisper log-mel, numpy (float64 STFT), no padding to 30 s.

    Mirrors ``WhisperFeatureExtractor._np_extract_fbank_features`` with
    ``padding='do_not_pad'``: center reflect-padded STFT with a periodic Hann
    window, power spectrum, slaney mel, ``log10(max(x, 1e-10))``, drop the last
    frame, clamp to ``max - 8`` and ``(x + 4) / 4``.
    """

    def __init__(
        self, feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_filters = slaney_mel_filters(
            n_fft // 2 + 1, feature_size, sampling_rate
        )
        self.window = 0.5 - 0.5 * np.cos(
            2 * np.pi * np.arange(n_fft) / n_fft
        )  # periodic hann

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        """wave: float32 mono [N] -> log-mel [T, feature_size] float32 (T = N // hop)."""
        n_fft, hop = self.n_fft, self.hop_length
        x = np.pad(np.asarray(wave, dtype=np.float64), n_fft // 2, mode="reflect")
        n_frames = 1 + (len(x) - n_fft) // hop
        mel = np.empty((n_frames, self.feature_size), dtype=np.float64)
        step = 2048  # bound memory for long audio
        for s in range(0, n_frames, step):
            e = min(n_frames, s + step)
            idx = np.arange(s, e)[:, None] * hop + np.arange(n_fft)[None, :]
            spec = np.abs(np.fft.rfft(x[idx] * self.window, axis=-1)) ** 2
            mel[s:e] = spec @ self.mel_filters
        log_spec = np.log10(np.maximum(mel, 1e-10))[:-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return ((log_spec + 4.0) / 4.0).astype(np.float32)


def sinc_resample(wave: np.ndarray, orig_freq: int, new_freq: int) -> np.ndarray:
    """numpy port of ``torchaudio.transforms.Resample`` (sinc_interp_hann, width 6, rolloff 0.99)."""
    if orig_freq == new_freq:
        return wave
    g = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // g, int(new_freq) // g
    lpfw, rolloff = 6, 0.99
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lpfw * orig / base_freq)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx  # [new, K]
    t = np.clip(t * base_freq, -lpfw, lpfw)
    window = np.cos(t * math.pi / lpfw / 2) ** 2
    t = t * math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        kernels = np.where(t == 0, 1.0, np.sin(t) / t)
    kernels = (kernels * window * (base_freq / orig)).astype(np.float32)  # [new, K]
    length = len(wave)
    x = np.pad(np.asarray(wave, dtype=np.float32), (width, width + orig))
    K = kernels.shape[1]
    n = (len(x) - K) // orig + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, K), strides=(x.strides[0] * orig, x.strides[0])
    )
    out = (frames @ kernels.T).reshape(-1)  # [n * new]
    return np.ascontiguousarray(out[: math.ceil(new * length / orig)])


def _read_wav(path) -> tuple:
    """Exact RIFF/WAVE reader (PCM 8/16/24/32, float32/64) -> ([samples, channels] float32, rate).

    Needed because miniaudio's FLOAT32 path re-quantises float/24-bit WAVs to 16 bit.
    """
    import struct

    if isinstance(path, (bytes, bytearray)):
        data = bytes(path)
    else:
        with open(path, "rb") as f:
            data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if (
                tag == 0xFFFE
            ):  # extensible: real tag is the first 2 bytes of the sub-format GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            pcm = body if size <= len(data) - pos - 8 else data[pos + 8 :]
            break
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("bad WAVE file")
    tag, ch, rate, bits = fmt
    if tag == 1 and bits == 16:
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = (
            np.frombuffer(pcm[: len(pcm) // 4 * 4], "<i4").astype(np.float64)
            / 2147483648.0
        )
    elif tag == 1 and bits == 24:
        b = (
            np.frombuffer(pcm[: len(pcm) // 3 * 3], np.uint8)
            .reshape(-1, 3)
            .astype(np.int32)
        )
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float64) / 8388608.0
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(pcm, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 3 and bits == 32:
        x = np.frombuffer(pcm[: len(pcm) // 4 * 4], "<f4")
    elif tag == 3 and bits == 64:
        x = np.frombuffer(pcm[: len(pcm) // 8 * 8], "<f8")
    else:
        raise ValueError(f"unsupported WAVE format {fmt}")
    x = x.astype(np.float32)
    return x[: len(x) // ch * ch].reshape(-1, ch), rate


def load_audio(file, sr: int = 16000, timeout: int = 10) -> np.ndarray:
    """File path / URL / array / (array, rate) -> mono float32 at ``sr`` (sinc resampling like torchaudio)."""
    rate = None
    if isinstance(file, tuple) and len(file) == 2 and np.ndim(file[0]) >= 1:
        file, rate = file
    if isinstance(file, np.ndarray) or hasattr(file, "__array__"):
        audio = np.asarray(file, dtype=np.float32)
        rate = rate or sr
    else:
        from io import BytesIO

        from mlx_audio.audio_io import read as read_audio

        if isinstance(file, str) and file.startswith(("http://", "https://")):
            import requests

            resp = requests.get(file, stream=True, timeout=timeout)
            resp.raise_for_status()
            audio, rate = read_audio(BytesIO(resp.content), dtype="float32")
        elif hasattr(
            file, "read"
        ):  # file-like (the OpenAI server passes BytesIO for input_audio)
            data = file.read()
            if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
                audio, rate = _read_wav(data)
            else:
                audio, rate = read_audio(BytesIO(data), dtype="float32")
        else:
            try:
                if str(file).lower().endswith(".wav"):
                    audio, rate = _read_wav(file)
                else:
                    raise ValueError
            except Exception:
                audio, rate = read_audio(str(file), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)  # [samples, channels]
    if (
        audio.ndim == 2
    ):  # [samples, channels] -> mono (mean over channels, as HF audio.mean(0))
        audio = audio.mean(axis=1)
    if rate != sr:
        audio = sinc_resample(audio, int(rate), int(sr))
    return np.ascontiguousarray(audio, dtype=np.float32)


def split_audio_chunks(n_frames: int, duration: float, min_second=2.0, max_second=30.0):
    """The frame-by-frame greedy chunking of ``add_audio_input_discrete_or_contiguous``.

    Returns [(start_frame, end_frame, start_seconds)]. Float arithmetic is kept
    identical to the HF loop so boundaries match exactly.
    """
    spf = 1.0 * duration / n_frames
    chunks, starts = [], []  # chunk lengths, start seconds
    cur = 0
    for _ in range(n_frames):
        sec = 1 * spf
        chunk_sec = 1.0 * cur * spf
        if sec + chunk_sec < min_second:
            cur += 1
        elif sec + chunk_sec <= max_second:
            cur += 1
        else:
            starts.append(sum([n * spf for n in chunks]))
            chunks.append(cur)
            cur = 1
    if cur > 0:
        starts.append(sum([n * spf for n in chunks]))
        chunks.append(cur)
    out, a = [], 0
    for n, sec in zip(chunks, starts):
        out.append((a, a + n, sec))
        a += n
    return out


class YoutuVitaAudioProcessor:
    """Audio front-end config (``processor_config.json``/``feature_extractor``)."""

    model_input_names = ["audios", "audio_indices"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        n_fft=400,
        hop_length=160,
        audio_chunk_min_second=2,
        audio_chunk_max_second=30,
        temporal_merge_size=1,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.audio_chunk_min_second = audio_chunk_min_second
        self.audio_chunk_max_second = audio_chunk_max_second
        self.temporal_merge_size = temporal_merge_size
        self.log_mel = WhisperLogMel(feature_size, sampling_rate, n_fft, hop_length)

    @classmethod
    def from_config(cls, cfg):
        cfg = dict(cfg or {})
        keys = (
            "feature_size",
            "sampling_rate",
            "n_fft",
            "hop_length",
            "audio_chunk_min_second",
            "audio_chunk_max_second",
            "temporal_merge_size",
        )
        return cls(**{k: cfg[k] for k in keys if k in cfg})


# ====================================================================== video
def _timestamp(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(round(seconds)))


def sample_frame_indices(
    n_frames: int, avg_fps: float, max_fps: float, max_frames: int
):
    """HF ``get_video_frames`` sampling policy -> (indices, timestamps, rounded_fps)."""
    fps = max(1, round(avg_fps))  # HF: round(avg_fps) (0 would divide by zero)
    step = n_frames / max_frames
    step = max(fps / max_fps, step)
    indices = [int(i * step) for i in range(max_frames)]
    indices = [i for i in indices if i < n_frames]
    return indices, [1.0 / fps * i for i in indices], fps


def _decode_frames_av(path: str, max_fps: float, max_frames: int):
    import av

    def run(n_hint):
        with av.open(path) as c:
            st = c.streams.video[0]
            rate = st.average_rate or st.guessed_rate
            avg_fps = float(rate)
            n = n_hint if n_hint is not None else st.frames
            if not n:  # container without a frame count: count packets
                with av.open(path) as c2:
                    n = sum(1 for p in c2.demux(c2.streams.video[0]) if p.size)
            idx, ts, fps = sample_frame_indices(n, avg_fps, max_fps, max_frames)
            want = {}
            for k, i in enumerate(idx):
                want.setdefault(i, []).append(k)
            frames, seen = [None] * len(idx), 0
            for i, f in enumerate(c.decode(st)):
                seen = i + 1
                if i in want:
                    arr = f.to_ndarray(format="rgb24")
                    for k in want[i]:
                        frames[k] = arr
            return n, seen, avg_fps, idx, ts, fps, frames

    n, seen, avg_fps, idx, ts, fps, frames = run(None)
    if seen != n:  # header frame count was wrong: redo with the decoded count
        n, seen, avg_fps, idx, ts, fps, frames = run(seen)
    keep = [k for k, f in enumerate(frames) if f is not None]
    return [frames[k] for k in keep], [ts[k] for k in keep], fps, n / avg_fps


def _decode_frames_cv2(path: str, max_fps: float, max_frames: int):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    avg_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    allf = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        allf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    idx, ts, fps = sample_frame_indices(len(allf), avg_fps, max_fps, max_frames)
    return [allf[i] for i in idx], ts, fps, len(allf) / avg_fps


def decode_video_frames(path: str, max_fps: float = 1, max_frames: int = 128):
    """Sampled RGB frames of a video file -> (frames [HxWx3 uint8], timestamps, rounded_fps, duration_s)."""
    try:
        import av  # noqa: F401
    except ImportError:
        frames, ts, fps, dur = _decode_frames_cv2(path, max_fps, max_frames)
    else:
        frames, ts, fps, dur = _decode_frames_av(path, max_fps, max_frames)
    return frames, ts, fps, dur


def load_video_audio(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Mono ``sr`` waveform of a video's audio track, or None when it has none (HF: torchaudio.load fails)."""
    try:
        import av
    except ImportError:
        try:  # ffmpeg-CLI based loader (mlx_audio); also used for mp3/m4a prompts
            w = load_audio(path, sr)
            return w if len(w) else None
        except Exception:
            return None
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                return None
            st = c.streams.audio[0]
            rs = None
            chunks = []
            for f in c.decode(st):
                if f.format.name != "fltp":  # packed / integer formats -> planar float
                    if rs is None:
                        rs = av.AudioResampler(
                            format="fltp", layout=f.layout.name, rate=f.sample_rate
                        )
                    parts = rs.resample(f)
                else:
                    parts = [f]
                for p in parts:
                    chunks.append(
                        p.to_ndarray().astype(np.float32, copy=False)
                    )  # [C, n]
            rate = st.rate
    except (
        Exception
    ) as e:  # HF swallows every error here and treats the video as silent
        import warnings

        warnings.warn(
            f"could not read the audio track of {path}: {e!r}; treating the video as silent"
        )
        return None
    if not chunks:
        return None
    wave = np.concatenate(chunks, axis=1).mean(axis=0)  # HF: audio.mean(0)
    if rate != sr:
        wave = sinc_resample(wave, int(rate), int(sr))
    wave = np.ascontiguousarray(wave, dtype=np.float32)
    return wave if len(wave) else None


def chunk_audio_video_frames(
    image_frames, audio_frames, second_frames, duration_seconds, min_second, max_second
):
    """HF ``_chunk_audio_video_frames`` (frames / log-mel slices are numpy arrays)."""
    if audio_frames is None:
        return [image_frames], [second_frames], [[]], [[]]
    second_per_audio = 1.0 * duration_seconds / sum(len(x) for x in audio_frames)
    ic, isc, ac, asc = [], [], [], []
    image_chunk, image_second_chunk, audio_chunk, audio_second_chunk = [], [], [], []
    for image_frame, audio_frame, second_frame in zip(
        image_frames, audio_frames, second_frames
    ):
        audio_second = len(audio_frame) * second_per_audio
        audio_chunk_second = sum(len(x) * second_per_audio for x in audio_chunk)
        if audio_second + audio_chunk_second < min_second:
            image_chunk.append(image_frame)
            image_second_chunk.append(second_frame)
            audio_chunk.append(audio_frame)
            audio_second_chunk.append(second_frame)
        elif audio_second + audio_chunk_second <= max_second:
            image_chunk.append(image_frame)
            image_second_chunk.append(second_frame)
            audio_chunk.append(audio_frame)
            audio_second_chunk.append(second_frame)
            ic.append(image_chunk)
            isc.append(image_second_chunk)
            audio_chunk = [np.concatenate(audio_chunk, axis=0)]
            audio_second_chunk = [audio_second_chunk[0]]
            ac.append(audio_chunk)
            asc.append(audio_second_chunk)
            image_chunk, image_second_chunk, audio_chunk, audio_second_chunk = (
                [],
                [],
                [],
                [],
            )
        else:
            assert audio_chunk_second == 0
            image_chunk.append(image_frame)
            image_second_chunk.append(second_frame)
            size = int(max_second / second_per_audio)
            audio_chunk = [
                audio_frame[i : i + size] for i in range(0, len(audio_frame), size)
            ]
            cur = second_frame
            for x in audio_chunk:
                audio_second_chunk.append(cur)
                cur += len(x) * second_per_audio
            ic.append(image_chunk)
            isc.append(image_second_chunk)
            ac.append(audio_chunk)
            asc.append(audio_second_chunk)
            image_chunk, image_second_chunk, audio_chunk, audio_second_chunk = (
                [],
                [],
                [],
                [],
            )
    if len(image_chunk) > 0:
        ic.append(image_chunk)
        isc.append(image_second_chunk)
        audio_chunk = [np.concatenate(audio_chunk, axis=0)]
        audio_second_chunk = [audio_second_chunk[0]]
        ac.append(audio_chunk)
        asc.append(audio_second_chunk)
    return ic, isc, ac, asc


class YoutuVitaVideoProcessor:
    """Frame sampling + budget of the HF ``YoutuVITAVideoProcessor`` (``processor_config.json``/``video_processor``)."""

    def __init__(
        self,
        video_max_num_frames=64,
        video_max_fps=1,
        video_min_num_tokens=64,
        video_max_num_tokens=8192,
        video_image_min_num_tokens=4,
        video_image_max_num_tokens=256,
        video_audio_chunk_min_second=2,
        video_audio_chunk_max_second=30,
        use_audio_in_video=True,
        use_vision_in_video=True,
        temporal_patch_size=1,
        spatial_merge_size=2,
        temporal_merge_size=1,
        patch_size=14,
        video_key_frame=False,
        **kwargs,
    ):
        if video_key_frame:
            raise NotImplementedError("video_key_frame")
        self.video_max_num_frames = video_max_num_frames
        self.video_max_fps = video_max_fps
        self.video_min_num_tokens = video_min_num_tokens
        self.video_max_num_tokens = video_max_num_tokens
        self.video_image_min_num_tokens = video_image_min_num_tokens
        self.video_image_max_num_tokens = video_image_max_num_tokens
        self.video_audio_chunk_min_second = video_audio_chunk_min_second
        self.video_audio_chunk_max_second = video_audio_chunk_max_second
        self.use_audio_in_video = use_audio_in_video
        self.use_vision_in_video = use_vision_in_video
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_merge_size = temporal_merge_size
        self.patch_size = patch_size

    @classmethod
    def from_config(cls, cfg):
        return cls(**dict(cfg or {}))

    def pixel_budget(self, n_frames: int):
        """(min_pixels, max_pixels) of one frame."""
        unit = (self.patch_size * self.spatial_merge_size) ** 2
        lo = unit * self.video_min_num_tokens // n_frames
        hi = unit * self.video_max_num_tokens // n_frames
        lo = max(lo, unit * self.video_image_min_num_tokens)
        hi = min(hi, unit * self.video_image_max_num_tokens)
        return lo, hi


class YoutuVitaProcessor:
    """Tokenizer + image processor with the HF ``YoutuVITAProcessor`` prompt expansion."""

    # <|image_pad|>: generation stops when the model emits it (as in the HF reference run)
    additional_eos_token_ids = (128257,)
    image_token = IMG_TAG_TOKEN  # what the chat template expands
    supports_multiple_audio = True
    # mlx_vlm.utils.prepare_inputs hands video *paths* over: frames AND the audio track are decoded here
    decodes_video_files = True

    def __init__(
        self,
        image_processor: YoutuVitaImageProcessor = None,
        tokenizer=None,
        audio_processor: "YoutuVitaAudioProcessor" = None,
        chat_template: Optional[str] = None,
        video_processor: "YoutuVitaVideoProcessor" = None,
        **kwargs,
    ):
        self.image_processor = image_processor or YoutuVitaImageProcessor()
        self.video_processor = video_processor or YoutuVitaVideoProcessor(patch_size=16)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or YoutuVitaAudioProcessor()
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)
        if tokenizer is not None:
            ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            if isinstance(ctx_id, int) and ctx_id >= 0:
                self.additional_eos_token_ids = (ctx_id,)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # AutoTokenizer would resolve config.json's auto_map (remote code) first
        from transformers import PreTrainedTokenizerFast

        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(pretrained_model_name_or_path)))
        tok_kwargs = {
            k: v for k, v in kwargs.items() if k in ("revision", "local_files_only")
        }
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), **tok_kwargs)
        load_chat_template(tokenizer, path)
        cfg_file = path / "processor_config.json"
        img_cfg, aud_cfg, vid_cfg = {}, {}, {}
        if cfg_file.exists():
            full = json.loads(cfg_file.read_text())
            img_cfg = full.get("image_processor", {})
            aud_cfg = full.get("feature_extractor", {})
            vid_cfg = full.get("video_processor", {})
        return cls(
            image_processor=YoutuVitaImageProcessor.from_config(img_cfg),
            audio_processor=YoutuVitaAudioProcessor.from_config(aud_cfg),
            video_processor=YoutuVitaVideoProcessor.from_config(vid_cfg),
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    # ------------------------------------------------------- tokenizer proxy
    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, v):
        self.tokenizer.pad_token = v

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        # The "parsing" template branch (no role markers) is the one the model was
        # trained/evaluated with; mlx-vlm only injects enable_thinking.
        kwargs.setdefault("enable_parsing", True)
        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template or self.chat_template,
            **kwargs,
        )

    # ----------------------------------------------------------------- audio
    def load_audio(self, file, sr: int = 16000, **kwargs) -> np.ndarray:
        """Hook used by ``mlx_vlm.utils.prepare_inputs`` (torchaudio-compatible resampling)."""
        return load_audio(file, sr)

    def _expand_audio(self, input_ids: List[int], audios):
        tok = self.tokenizer
        tag_id = tok.convert_tokens_to_ids(AUD_TAG_TOKEN)
        ctx_id = tok.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        start_id = tok.convert_tokens_to_ids(AUD_START_TOKEN)
        end_id = tok.convert_tokens_to_ids(AUD_END_TOKEN)
        positions = [i for i, t in enumerate(input_ids) if t == tag_id]
        if len(positions) != len(audios):
            raise ValueError(
                f"Prompt has {len(positions)} {AUD_TAG_TOKEN} tag(s) but "
                f"{len(audios)} audio clip(s) were given."
            )
        fe = self.audio_processor
        new_ids: List[int] = []
        clips: List[np.ndarray] = []
        chunk_lens: List[int] = []
        st = 0
        for pos, audio in zip(positions, audios):
            new_ids += input_ids[st:pos]
            wave = load_audio(audio, fe.sampling_rate)
            feats = fe.log_mel(wave)  # [T, 128]
            if feats.shape[0] == 0:
                raise ValueError("Audio is too short (no mel frames).")
            duration = len(wave) / fe.sampling_rate
            spans = split_audio_chunks(
                feats.shape[0],
                duration,
                fe.audio_chunk_min_second,
                fe.audio_chunk_max_second,
            )
            for a, b, sec in spans:
                clip = np.ascontiguousarray(feats[a:b])
                clips.append(clip)
                if len(spans) > 1:
                    ts = time.strftime("%H:%M:%S", time.gmtime(round(sec)))
                    new_ids += tok(ts, add_special_tokens=False)["input_ids"]
                n = -(-audio_token_length(b - a) // fe.temporal_merge_size)
                assert n > 0
                new_ids.append(start_id)
                new_ids += [ctx_id] * n
                new_ids.append(end_id)
                chunk_lens.append(n)
            st = pos + 1
        new_ids += input_ids[st:]
        return new_ids, clips, chunk_lens

    def _video_frames(self, video, vp: "YoutuVitaVideoProcessor", max_fps, max_frames):
        """-> (frames, timestamps, fps, duration_s, audio wave or None)."""
        if isinstance(video, (str, Path)):
            path = str(video)
            if path.startswith("file://"):
                path = path[7:]
            frames, ts, fps, dur = decode_video_frames(path, max_fps, max_frames)
            wave = (
                load_video_audio(path, self.audio_processor.sampling_rate)
                if vp.use_audio_in_video
                else None
            )
            return frames, ts, fps, dur, wave
        # already decoded frames (list of PIL / HxWx3 arrays, or [T, C, H, W] / [T, H, W, C]): no audio, 1 fps
        arr = video
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim == 4
            and arr.shape[1] in (1, 3)
            and arr.shape[-1] not in (1, 3)
        ):
            arr = arr.transpose(0, 2, 3, 1)
        frames = [
            (
                np.asarray(f.convert("RGB"))
                if isinstance(f, Image.Image)
                else np.asarray(f)
            )
            for f in arr
        ]
        frames = frames[:max_frames]
        return (
            frames,
            [float(i) for i in range(len(frames))],
            1,
            float(len(frames)),
            None,
        )

    def _expand_video(
        self,
        input_ids: List[int],
        videos,
        max_fps=None,
        max_frames=None,
        use_audio=None,
    ):
        tok = self.tokenizer
        vp, ip, fe = self.video_processor, self.image_processor, self.audio_processor
        if not vp.use_vision_in_video:
            raise NotImplementedError("use_vision_in_video=False")
        max_fps = vp.video_max_fps if max_fps is None else max_fps
        max_frames = vp.video_max_num_frames if max_frames is None else max_frames
        use_audio = vp.use_audio_in_video if use_audio is None else use_audio
        ids = lambda t: tok.convert_tokens_to_ids(t)
        tag_id, vstart, vend = (
            ids(VID_TAG_TOKEN),
            ids(VID_START_TOKEN),
            ids(VID_END_TOKEN),
        )
        img_ctx, img_start, img_end = (
            ids(IMG_CONTEXT_TOKEN),
            ids(IMG_START_TOKEN),
            ids(IMG_END_TOKEN),
        )
        aud_ctx, aud_start, aud_end = (
            ids(AUD_CONTEXT_TOKEN),
            ids(AUD_START_TOKEN),
            ids(AUD_END_TOKEN),
        )
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        positions = [i for i, t in enumerate(input_ids) if t == tag_id]
        if len(positions) != len(videos):
            raise ValueError(
                f"Prompt has {len(positions)} {VID_TAG_TOKEN} tag(s) but {len(videos)} video(s) were given."
            )
        m, p = vp.spatial_merge_size, vp.patch_size
        pix, grids, img_cols, auds, aud_cols, split = [], [], [], [], [], []
        new_ids: List[int] = []
        st = 0
        for pos, video in zip(positions, videos):
            frames, ts, _fps, duration, wave = self._video_frames(
                video, vp, max_fps, max_frames
            )
            if len(frames) == 0:
                raise ValueError("No frames could be read from the video.")
            lo, hi = vp.pixel_budget(len(frames))
            patches, grid = [], None
            for f in frames:
                pt, grid, _ = ip.process_image(f, lo, hi)
                patches.append(pt)
            n_img = len(frames)

            audio_frames, audio_chunk_second_per = None, None
            if wave is not None and use_audio and len(wave) > 0:
                mel = fe.log_mel(wave)  # [T, 128], one log-mel for the whole track
                if mel.shape[0] > 0:
                    total_time = len(wave) / fe.sampling_rate
                    full_ts = list(ts) + [total_time]
                    audio_frames = []
                    for k in range(n_img):
                        a = int(full_ts[k] / total_time * len(mel))
                        b = int(full_ts[k + 1] / total_time * len(mel))
                        audio_frames.append(mel[a:b])
            if audio_frames is not None:
                n_keep = min(n_img, len(audio_frames))
                frames, patches, audio_frames, ts = (
                    frames[:n_keep],
                    patches[:n_keep],
                    audio_frames[:n_keep],
                    ts[:n_keep],
                )
                n_img = n_keep

            ic, isc, ac, asc = chunk_audio_video_frames(
                frames,
                audio_frames,
                ts,
                duration,
                vp.video_audio_chunk_min_second,
                vp.video_audio_chunk_max_second,
            )
            assert n_img == sum(len(x) for x in ic)
            pix.append(np.concatenate(patches, axis=0))
            if audio_frames is not None:
                new_auds = [
                    np.ascontiguousarray(xx) for x in ac for xx in x if len(xx) > 0
                ]
                auds += new_auds
            else:
                new_auds = []

            new_ids += input_ids[st:pos]
            new_ids.append(vstart)
            gh, gw = grid[1], grid[2]
            res_ids = tok(f"{gh * p}*{gw * p}", add_special_tokens=False)["input_ids"]
            for ichunk, isecs, achunk, asecs in zip(ic, isc, ac, asc):
                if audio_frames is None:
                    achunk, asecs = [], []
                for _frame, sec in zip(ichunk, isecs):
                    new_ids += tok(_timestamp(sec), add_special_tokens=False)[
                        "input_ids"
                    ]
                    new_ids.append(img_start)
                    new_ids += res_ids
                    row_len = gw // m
                    for _ in range(1 * gh // m):
                        img_cols.append(np.arange(len(new_ids), len(new_ids) + row_len))
                        new_ids += [img_ctx] * row_len
                        new_ids += nl
                    new_ids.append(img_end)
                    grids.append([1, gh, gw])
                for a_chunk, sec in zip(achunk, asecs):
                    if len(a_chunk) == 0:
                        continue
                    new_ids += tok(_timestamp(sec), add_special_tokens=False)[
                        "input_ids"
                    ]
                    new_ids.append(aud_start)
                    n = (
                        audio_token_length(len(a_chunk)) + vp.temporal_merge_size - 1
                    ) // vp.temporal_merge_size
                    assert n > 0
                    start = len(new_ids)
                    aud_cols.append(
                        np.stack(
                            [
                                np.zeros((1, n), np.int64),
                                np.arange(start, start + n, dtype=np.int64)[None],
                            ],
                            0,
                        )
                    )
                    new_ids += [aud_ctx] * n
                    new_ids.append(aud_end)
            new_ids.append(vend)
            split.append((n_img, len(new_auds)))
            st = pos + 1
        new_ids += input_ids[st:]
        idx = np.concatenate(img_cols).astype(np.int64)
        out = {
            "video_images": np.concatenate(pix, axis=0),
            "video_image_grid_thw": np.array(grids, dtype=np.int64),
            "video_image_indices": np.stack([np.zeros_like(idx), idx], axis=0),
            "video_split": np.array(split, dtype=np.int64),
        }
        if auds:
            out["video_audios"] = auds
            out["video_audio_indices"] = aud_cols
        return new_ids, out

    # --------------------------------------------------------------- __call__
    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images=None,
        audio=None,
        videos=None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        video_max_fps=None,
        video_max_num_frames=None,
        use_audio_in_video=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise NotImplementedError("Only a single prompt is supported.")
            text = text[0]
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        if images and all(isinstance(i, (list, tuple)) for i in images):
            images = [im for sub in images for im in sub]

        if audio is not None and not isinstance(audio, list):
            audio = [audio]

        tok = self.tokenizer
        input_ids = list(tok(text, add_special_tokens=add_special_tokens)["input_ids"])
        out: Dict[str, Any] = {}

        audio_chunk_lens: List[int] = []
        if audio:
            input_ids, clips, audio_chunk_lens = self._expand_audio(input_ids, audio)
            out["audios"] = clips

        if images:
            tag_id = tok.convert_tokens_to_ids(IMG_TAG_TOKEN)
            ctx_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            start_id = tok.convert_tokens_to_ids(IMG_START_TOKEN)
            end_id = tok.convert_tokens_to_ids(IMG_END_TOKEN)
            nl = tok("\n", add_special_tokens=False)["input_ids"]
            ip = self.image_processor
            m, p = ip.spatial_merge_size, ip.patch_size

            positions = [i for i, t in enumerate(input_ids) if t == tag_id]
            if len(positions) != len(images):
                raise ValueError(
                    f"Prompt has {len(positions)} {IMG_TAG_TOKEN} tag(s) but "
                    f"{len(images)} image(s) were given."
                )

            new_ids: List[int] = []
            pixel_rows, grids, index_cols = [], [], []
            st = 0
            for pos, image in zip(positions, images):
                new_ids += input_ids[st:pos]
                patches, grid, _ = ip.process_image(image)
                pixel_rows.append(patches)
                grids.append(grid)
                t, gh, gw = grid
                new_ids.append(start_id)
                new_ids += tok(f"{gh * p}*{gw * p}", add_special_tokens=False)[
                    "input_ids"
                ]
                new_ids += nl
                row_len = gw // m
                for _ in range(t * gh // m):
                    index_cols.append(np.arange(len(new_ids), len(new_ids) + row_len))
                    new_ids += [ctx_id] * row_len
                    new_ids += nl
                new_ids.append(end_id)
                st = pos + 1
            new_ids += input_ids[st:]
            input_ids = new_ids

            out["images"] = np.concatenate(pixel_rows, axis=0)
            idx = np.concatenate(index_cols).astype(np.int64)
            out["image_indices"] = np.stack([np.zeros_like(idx), idx], axis=0)
            out["image_grid_thw"] = np.array(grids, dtype=np.int64)

        if audio_chunk_lens:
            # positions are taken from the final ids so that image expansion
            # (done after the audio one, as in HF) cannot shift them.
            ctx = tok.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
            pos = np.nonzero(np.asarray(input_ids) == ctx)[0].astype(np.int64)
            assert len(pos) == sum(audio_chunk_lens)
            indices, a = [], 0
            for (
                n
            ) in audio_chunk_lens:  # HF layout: list of [2, 1, n] (batch idx, position)
                indices.append(
                    np.stack([np.zeros((1, n), np.int64), pos[a : a + n][None]], axis=0)
                )
                a += n
            out["audio_indices"] = indices

        if videos is not None and not (hasattr(videos, "__len__") and len(videos) == 0):
            if isinstance(videos, (str, Path)) or (
                isinstance(videos, np.ndarray) and videos.ndim == 4
            ):
                videos = [videos]
            input_ids, vout = self._expand_video(
                input_ids,
                list(videos),
                video_max_fps,
                video_max_num_frames,
                use_audio_in_video,
            )
            out.update(vout)

        ids = np.array([input_ids], dtype=np.int64)
        out["input_ids"] = ids
        out["attention_mask"] = np.ones_like(ids)
        return out


install_auto_processor_patch("youtu_vita", YoutuVitaProcessor)

__all__ = ["YoutuVitaProcessor", "YoutuVitaImageProcessor", "YoutuVitaAudioProcessor"]
