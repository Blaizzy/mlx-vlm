from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from .constants import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
)
from .packing import resolve_canvas_size
from .references import MINIMAX_H3_QWEN_TEMPORAL_PATCH, MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS


def decode_image(image: str | Path | Image.Image | np.ndarray | mx.array) -> mx.array:
    """Decode image I/O and return channels-last RGB pixels as an MLX array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as decoded:
            image = ImageOps.exif_transpose(decoded).convert("RGB")
            return mx.array(np.asarray(image).copy(), dtype=mx.uint8)
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        return mx.array(np.asarray(image).copy(), dtype=mx.uint8)
    array = image if isinstance(image, mx.array) else mx.array(image)
    if array.ndim != 3:
        raise ValueError(f"image must have three dimensions, got {array.shape}")
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.shape[-1] != 3:
        raise ValueError(f"image must have three RGB channels, got {array.shape}")
    return media_to_uint8(array)


def decode_video(
    video: str | Path | np.ndarray | mx.array,
) -> tuple[mx.array, float | None]:
    """Decode a video container at the I/O boundary without resizing frames."""
    if not isinstance(video, (str, Path)):
        frames = video if isinstance(video, mx.array) else mx.array(video)
        if frames.ndim != 4:
            raise ValueError(f"video must have four dimensions, got {frames.shape}")
        if frames.shape[-1] != 3 and frames.shape[1] == 3:
            frames = frames.transpose(0, 2, 3, 1)
        if frames.shape[-1] != 3:
            raise ValueError(
                f"video frames must have three RGB channels, got {frames.shape}"
            )
        return media_to_uint8(frames), None

    import cv2

    path = str(video)
    if path.startswith("file://"):
        path = path[7:]
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    try:
        while True:
            ok, decoded = capture.read()
            if not ok:
                break
            # OpenCV exposes decoded buffers as BGR. Move into MLX before the
            # channel permutation so all post-decode numeric work stays native.
            frame = mx.array(decoded, dtype=mx.uint8)
            frames.append(mx.take(frame, mx.array([2, 1, 0]), axis=-1))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {video}")
    return mx.stack(frames), fps or None


def decode_video_soundtrack(
    video: str | Path | np.ndarray | mx.array,
) -> tuple[mx.array, int] | None:
    """Decode an embedded video soundtrack at the container I/O boundary.

    In-memory frame arrays cannot carry a soundtrack. For file inputs, a missing
    audio stream returns ``None``; a present stream is decoded to channel-first
    float32 and moved into MLX immediately.
    """
    if not isinstance(video, (str, Path)):
        return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    path = str(video)
    if path.startswith("file://"):
        path = path[7:]
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    try:
        streams = json.loads(probe.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid metadata for {video}") from exc
    if not streams:
        return None
    sample_rate = int(streams[0].get("sample_rate", 0))
    channels = int(streams[0].get("channels", 0))
    if sample_rate <= 0 or channels not in (1, 2):
        raise ValueError(
            f"video soundtrack must be mono or stereo with a valid sample rate, "
            f"got {channels} channels at {sample_rate} Hz"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to decode the soundtrack discovered by ffprobe"
        )
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if decoded.returncode != 0:
        message = decoded.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot decode video soundtrack: {message or video}")
    samples = np.frombuffer(decoded.stdout, dtype="<f4")
    if not samples.size or samples.size % channels:
        raise ValueError(f"video soundtrack contains invalid PCM data: {video}")
    waveform = mx.array(samples.reshape(-1, channels).copy(), dtype=mx.float32)
    return waveform.transpose(1, 0), sample_rate


def decode_audio(
    audio: str | Path | np.ndarray | mx.array,
) -> tuple[mx.array, int | None]:
    """Decode waveform samples and return channel-first MLX data plus its rate."""
    sample_rate = None
    if isinstance(audio, (str, Path)):
        from mlx_audio.audio_io import read as read_audio

        decoded, sample_rate = read_audio(str(audio), dtype="float32")
        waveform = mx.array(decoded, dtype=mx.float32)
    else:
        waveform = audio if isinstance(audio, mx.array) else mx.array(audio)
    if waveform.ndim == 1:
        waveform = waveform[None]
    elif waveform.ndim == 2 and waveform.shape[0] not in (1, 2):
        if waveform.shape[-1] not in (1, 2):
            raise ValueError(
                f"audio must be mono or stereo, got shape {waveform.shape}"
            )
        waveform = waveform.transpose(1, 0)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            f"audio must have channel-first mono/stereo shape, got {waveform.shape}"
        )
    return waveform.astype(mx.float32), sample_rate


def media_to_uint8(media: mx.array | np.ndarray) -> mx.array:
    array = media if isinstance(media, mx.array) else mx.array(media)
    if array.dtype == mx.uint8:
        return array
    return mx.clip(mx.round(array.astype(mx.float32) * 255.0), 0, 255).astype(mx.uint8)


def _lanczos_kernel(distance: mx.array, support: float = 3.0) -> mx.array:
    absolute = mx.abs(distance)
    pi_distance = math.pi * distance
    sinc = mx.where(absolute < 1e-7, 1.0, mx.sin(pi_distance) / pi_distance)
    window_arg = pi_distance / support
    window = mx.where(
        absolute < 1e-7,
        1.0,
        mx.sin(window_arg) / window_arg,
    )
    return mx.where(absolute < support, sinc * window, 0.0)


def _bicubic_kernel(distance: mx.array, coefficient: float = -0.75) -> mx.array:
    absolute = mx.abs(distance)
    first = (coefficient + 2.0) * absolute**3 - (coefficient + 3.0) * absolute**2 + 1.0
    second = (
        coefficient * absolute**3
        - 5.0 * coefficient * absolute**2
        + 8.0 * coefficient * absolute
        - 4.0 * coefficient
    )
    return mx.where(absolute < 1.0, first, mx.where(absolute < 2.0, second, 0.0))


def _resample_axis_lanczos(
    images: mx.array,
    output_size: int,
    axis: int,
) -> mx.array:
    input_size = images.shape[axis]
    if input_size == output_size:
        return images
    scale = input_size / output_size
    filter_scale = max(1.0, scale)
    radius = math.ceil(3.0 * filter_scale)
    centers = (mx.arange(output_size, dtype=mx.float32) + 0.5) * scale - 0.5
    offsets = mx.arange(-radius + 1, radius + 1, dtype=mx.int32)
    source_indices = mx.floor(centers).astype(mx.int32)[:, None] + offsets[None]
    distances = (centers[:, None] - source_indices.astype(mx.float32)) / filter_scale
    weights = _lanczos_kernel(distances) / filter_scale
    valid = (source_indices >= 0) & (source_indices < input_size)
    weights = weights * valid
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    source_indices = mx.clip(source_indices, 0, input_size - 1)

    gathered = mx.take(images, source_indices, axis=axis)
    if axis == 1:
        # [N, out_h, taps, W, C]
        return mx.sum(gathered * weights[None, :, :, None, None], axis=2)
    if axis == 2:
        # [N, H, out_w, taps, C]
        return mx.sum(gathered * weights[None, None, :, :, None], axis=3)
    raise ValueError(f"unsupported resize axis {axis}")


def _resample_axis_bicubic(
    images: mx.array,
    output_size: int,
    axis: int,
) -> mx.array:
    input_size = images.shape[axis]
    if input_size == output_size:
        return images
    scale = input_size / output_size
    filter_scale = max(1.0, scale)
    radius = math.ceil(2.0 * filter_scale)
    centers = (mx.arange(output_size, dtype=mx.float32) + 0.5) * scale - 0.5
    offsets = mx.arange(-radius + 1, radius + 1, dtype=mx.int32)
    source_indices = mx.floor(centers).astype(mx.int32)[:, None] + offsets[None]
    distances = (centers[:, None] - source_indices.astype(mx.float32)) / filter_scale
    weights = _bicubic_kernel(distances) / filter_scale
    valid = (source_indices >= 0) & (source_indices < input_size)
    weights = weights * valid
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    source_indices = mx.clip(source_indices, 0, input_size - 1)
    gathered = mx.take(images, source_indices, axis=axis)
    if axis == 1:
        return mx.sum(gathered * weights[None, :, :, None, None], axis=2)
    if axis == 2:
        return mx.sum(gathered * weights[None, None, :, :, None], axis=3)
    raise ValueError(f"unsupported resize axis {axis}")


def resize_lanczos(
    pixels: mx.array,
    height: int,
    width: int,
) -> mx.array:
    """Resize one RGB image or a frame batch with an MLX Lanczos-3 filter."""
    if height <= 0 or width <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    squeeze = pixels.ndim == 3
    if squeeze:
        pixels = pixels[None]
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(
            "pixels must be (height, width, 3) or (frames, height, width, 3), "
            f"got {pixels.shape}"
        )
    if pixels.shape[1:3] == (height, width):
        return pixels[0] if squeeze else pixels

    source_dtype = pixels.dtype
    resized = pixels.astype(mx.float32)
    resized = _resample_axis_lanczos(resized, width, axis=2)
    # Pillow's uint8 path materializes and clips the horizontal pass before
    # running the vertical pass. This matters around Lanczos overshoot.
    if source_dtype == mx.uint8:
        resized = mx.clip(mx.round(resized), 0, 255)
    resized = _resample_axis_lanczos(resized, height, axis=1)
    if source_dtype == mx.uint8:
        resized = mx.clip(mx.round(resized), 0, 255).astype(mx.uint8)
    else:
        resized = resized.astype(source_dtype)
    return resized[0] if squeeze else resized


def resize_bicubic(
    pixels: mx.array,
    height: int,
    width: int,
) -> mx.array:
    """Antialiased MLX bicubic resize used by the Qwen3-VL conditioner."""
    squeeze = pixels.ndim == 3
    if squeeze:
        pixels = pixels[None]
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"pixels must be HWC or THWC RGB, got {pixels.shape}")
    if pixels.shape[1:3] == (height, width):
        return pixels[0] if squeeze else pixels
    source_dtype = pixels.dtype
    resized = _resample_axis_bicubic(pixels.astype(mx.float32), width, axis=2)
    resized = _resample_axis_bicubic(resized, height, axis=1)
    if source_dtype == mx.uint8:
        resized = mx.clip(mx.round(resized), 0, 255).astype(mx.uint8)
    else:
        resized = resized.astype(source_dtype)
    return resized[0] if squeeze else resized


def prepare_keyframe_image(
    image: mx.array,
    height: int,
    width: int,
    *,
    stretch: bool,
) -> mx.array:
    image = decode_image(image)
    if image.shape[:2] == (height, width):
        return image
    if stretch:
        return resize_lanczos(image, height, width)
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    resized = resize_lanczos(image, resized_height, resized_width)
    left = max(0, (resized_width - width) // 2)
    top = max(0, (resized_height - height) // 2)
    return resized[top : top + height, left : left + width]


def prepare_reference_image(
    image: mx.array,
    height: int,
    width: int,
) -> mx.array:
    image = decode_image(image)
    return resize_lanczos(image, height, width)


def resample_reference_frames(frames: mx.array, fps: float) -> mx.array:
    if fps <= 0:
        raise ValueError(f"reference video frame rate must be positive, got {fps}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {frames.shape}")
    if fps == MINIMAX_H3_FPS:
        return frames
    scale = MINIMAX_H3_FPS / fps
    slots = [math.floor(index * scale + 0.5) for index in range(frames.shape[0])]
    stream_end = math.floor(frames.shape[0] * scale + 0.5)
    repeats = [right - left for left, right in zip(slots, [*slots[1:], stream_end])]
    indices = [index for index, count in enumerate(repeats) for _ in range(count)]
    return mx.take(frames, mx.array(indices, dtype=mx.int32), axis=0)


def prepare_reference_frames(frames: mx.array, num_frames: int) -> mx.array:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {frames.shape}")
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    return resize_lanczos(frames, height, width)


def sample_reference_video_frames(
    frames: mx.array,
) -> tuple[mx.array, list[float]]:
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [
        index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))
    ]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return mx.take(frames, mx.array(indices, dtype=mx.int32), axis=0), block_timestamps


def normalize_visual_vae_pixels(pixels: mx.array) -> mx.array:
    """Convert uint8 or [0, 1] RGB THWC/HWC pixels to NCTHW VAE input."""
    if pixels.ndim == 3:
        pixels = pixels[None]
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"pixels must be HWC or THWC RGB, got {pixels.shape}")
    values = pixels.astype(mx.float32)
    if pixels.dtype == mx.uint8:
        values = values / 255.0
    mean = mx.array(MINIMAX_H3_PIXEL_MEAN, dtype=mx.float32)
    std = mx.array(MINIMAX_H3_PIXEL_STD, dtype=mx.float32)
    values = (values - mean) / std
    return values.transpose(3, 0, 1, 2)[None]


def normalize_qwen_pixels(pixels: mx.array) -> mx.array:
    values = pixels.astype(mx.float32)
    if pixels.dtype == mx.uint8:
        values = values / 255.0
    return (values - 0.5) / 0.5


def _smart_resize_image(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen image aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def _smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 4096,
    max_pixels: int = 25165824,
) -> tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"video dimensions must be at least {factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen video aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    padded_frames = math.ceil(num_frames / temporal_factor) * temporal_factor
    if padded_frames * resized_height * resized_width > max_pixels:
        scale = math.sqrt(num_frames * height * width / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif padded_frames * resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def _qwen_patchify(
    frames: mx.array,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[mx.array, tuple[int, int, int]]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Qwen frames must be THWC RGB, got {frames.shape}")
    padding = (-frames.shape[0]) % temporal_patch_size
    if padding:
        frames = mx.concatenate(
            [frames, mx.repeat(frames[-1:], padding, axis=0)],
            axis=0,
        )
    frames = normalize_qwen_pixels(frames).transpose(0, 3, 1, 2)
    grid_t = frames.shape[0] // temporal_patch_size
    grid_h = frames.shape[2] // patch_size
    grid_w = frames.shape[3] // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(
            f"Qwen grid {(grid_h, grid_w)} must be divisible by merge size {merge_size}"
        )
    patches = frames[None].reshape(
        1,
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = patches.reshape(
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )
    return patches, (grid_t, grid_h, grid_w)


def process_qwen_images(
    images: list[mx.array],
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[mx.array, mx.array]:
    all_patches = []
    all_grids = []
    factor = patch_size * merge_size
    for image in images:
        image = decode_image(image)
        height, width = _smart_resize_image(
            image.shape[0],
            image.shape[1],
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = resize_bicubic(image, height, width)
        frames = mx.repeat(image[None], temporal_patch_size, axis=0)
        # _qwen_patchify groups every temporal_patch_size input frames. The
        # explicit repeat gives images grid_t=1, as in Transformers.
        patches, grid = _qwen_patchify(
            frames,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )
        all_patches.append(patches)
        all_grids.append(grid)
    if not all_patches:
        width = 3 * temporal_patch_size * patch_size * patch_size
        return mx.zeros((0, width), mx.float32), mx.zeros((0, 3), mx.int32)
    return mx.concatenate(all_patches), mx.array(all_grids, dtype=mx.int32)


def process_qwen_videos(
    videos: list[mx.array],
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 4096,
    max_pixels: int = 25165824,
) -> tuple[mx.array, mx.array]:
    all_patches = []
    all_grids = []
    factor = patch_size * merge_size
    for frames in videos:
        if not isinstance(frames, mx.array):
            frames = mx.array(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Qwen video must be THWC RGB, got {frames.shape}")
        height, width = _smart_resize_video(
            frames.shape[0],
            frames.shape[1],
            frames.shape[2],
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        frames = resize_bicubic(frames, height, width)
        patches, grid = _qwen_patchify(
            frames,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )
        all_patches.append(patches)
        all_grids.append(grid)
    if not all_patches:
        width = 3 * temporal_patch_size * patch_size * patch_size
        return mx.zeros((0, width), mx.float32), mx.zeros((0, 3), mx.int32)
    return mx.concatenate(all_patches), mx.array(all_grids, dtype=mx.int32)


def _sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> tuple[mx.array, int, int, int]:
    gcd = math.gcd(orig_freq, new_freq)
    orig = orig_freq // gcd
    new = new_freq // gcd
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)

    with mx.stream(mx.cpu):
        orig_value = mx.array(orig, dtype=mx.float64)
        base_value = mx.array(base_freq, dtype=mx.float64)
        indices = mx.arange(-width, width + orig, dtype=mx.float64)[None, None]
        indices = indices / orig_value
        # TorchAudio constructs this arange at the default float32 dtype even
        # though `indices` is float64. Preserve that rounding before promotion.
        phases = mx.arange(0, -new, -1, dtype=mx.float32)[:, None, None]
        phases = (phases / new).astype(mx.float64)
        time = (phases + indices) * base_value
        time = mx.clip(time, -lowpass_filter_width, lowpass_filter_width)
        window_scale = mx.array(
            math.pi / lowpass_filter_width / 2,
            dtype=mx.float64,
        )
        window = mx.cos(time * window_scale) ** 2
        time = time * mx.array(math.pi, dtype=mx.float64)
        sinc = mx.where(mx.abs(time) < 1e-12, 1.0, mx.sin(time) / time)
        kernel = (sinc * window * (base_value / orig_value)).astype(mx.float32)
        mx.eval(kernel)
    # MLX Conv1d expects [out_channels, kernel_size, in_channels].
    return kernel[:, 0, :, None], width, orig, new


def resample_waveform(
    waveform: mx.array,
    orig_freq: int,
    new_freq: int,
) -> mx.array:
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(
            f"sample rates must be positive, got {orig_freq} and {new_freq}"
        )
    if waveform.ndim != 2:
        raise ValueError(
            f"waveform must have shape (channels, samples), got {waveform.shape}"
        )
    if orig_freq == new_freq:
        return waveform
    kernel, width, orig, new = _sinc_resample_kernel(orig_freq, new_freq)
    length = waveform.shape[-1]
    values = waveform.astype(mx.float32).transpose(1, 0)[None]
    # Treat channels as a batch so the phase kernels are shared.
    values = values.transpose(2, 1, 0)
    values = mx.pad(values, ((0, 0), (width, width + orig), (0, 0)))
    output = mx.conv1d(values, kernel, stride=orig)
    output = output.reshape(waveform.shape[0], -1)
    target_length = math.ceil(new * length / orig)
    return output[:, :target_length]


def prepare_reference_waveform(
    waveform: mx.array | np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
) -> mx.array:
    waveform = waveform if isinstance(waveform, mx.array) else mx.array(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            "reference waveform must be mono or stereo with shape "
            f"(channels, samples), got {waveform.shape}"
        )
    waveform = waveform.astype(mx.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] == 1:
        waveform = mx.repeat(waveform, MINIMAX_H3_AUDIO_CHANNELS, axis=0)
    return resample_waveform(waveform, sample_rate, target_sample_rate)


__all__ = [
    "decode_audio",
    "decode_image",
    "decode_video",
    "decode_video_soundtrack",
    "media_to_uint8",
    "normalize_qwen_pixels",
    "normalize_visual_vae_pixels",
    "prepare_keyframe_image",
    "prepare_reference_frames",
    "prepare_reference_image",
    "prepare_reference_waveform",
    "process_qwen_images",
    "process_qwen_videos",
    "resample_reference_frames",
    "resample_waveform",
    "resize_bicubic",
    "resize_lanczos",
    "sample_reference_video_frames",
]
