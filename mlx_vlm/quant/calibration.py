"""In-house activation calibration for weight quantization.

Captures, per ``nn.Linear``, the mean absolute input activation per input
channel and a small subsample of raw input rows. Pure mlx.core/mlx.nn.
"""

from typing import Callable, Dict, List

import mlx.core as mx
import mlx.nn as nn

DEFAULT_CALIBRATION_TEXT: List[str] = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In 1969, humans first walked on the surface of the Moon.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose.",
    "She sold seashells by the seashore on a bright summer morning.",
    "The stock market reacted sharply to the central bank's interest decision.",
    "A balanced diet includes proteins, carbohydrates, fats and vitamins.",
    "The ancient library held thousands of scrolls written in many languages.",
    "Machine learning models are trained on large datasets to find patterns.",
    "The recipe calls for two cups of flour, a pinch of salt and one egg.",
    "Mountains rose in the distance as the travelers crossed the wide valley.",
    "Electric vehicles are becoming more common as battery costs decline.",
    "He carefully repaired the old clock, replacing each worn gear by hand.",
    "The orchestra tuned their instruments before the evening performance began.",
    "Rainforests are home to more than half of the world's plant species.",
    "The programmer debugged the function and the tests finally passed.",
    "Under the calm sea, colorful fish darted among the coral reefs.",
]


def _named_linears(model: nn.Module) -> Dict[int, str]:
    return {
        id(module): path
        for path, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


def collect_activation_stats(
    model: nn.Module,
    run_forward: Callable[[], None],
    max_rows: int = 64,
) -> Dict[str, dict]:
    """Run ``run_forward`` while recording per-linear input statistics.

    Returns ``{path: {"scale": mean|x| [in], "inputs": X [rows, in]}}`` for
    every ``nn.Linear`` reached during the forward passes.
    """
    id_to_path = _named_linears(model)
    sums: Dict[int, mx.array] = {}
    counts: Dict[int, int] = {}
    rows: Dict[int, list] = {}
    original = nn.Linear.__call__

    def hooked(self, x, *args, **kwargs):
        i = id(self)
        if i in id_to_path:
            xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            s = mx.sum(mx.abs(xf), axis=0)
            n = int(xf.shape[0])
            mx.eval(s)
            if i in sums:
                sums[i] = sums[i] + s
                counts[i] += n
            else:
                sums[i] = s
                counts[i] = n
            kept = rows.setdefault(i, [])
            have = sum(int(r.shape[0]) for r in kept)
            if have < max_rows:
                take = min(max_rows - have, n)
                kept.append(xf[:take].astype(mx.float16))
        return original(self, x, *args, **kwargs)

    try:
        nn.Linear.__call__ = hooked
        run_forward()
    finally:
        nn.Linear.__call__ = original

    stats: Dict[str, dict] = {}
    for i, path in id_to_path.items():
        if i not in sums:
            continue
        scale = sums[i] / counts[i]
        inputs = mx.concatenate(rows[i], axis=0) if rows.get(i) else None
        mx.eval(scale)
        if inputs is not None:
            mx.eval(inputs)
        stats[path] = {"scale": scale, "inputs": inputs}
    return stats


def synthetic_calibration_images(n: int = 8, size: int = 256, seed: int = 0):
    """Varied synthetic RGB images (gradients, tiles, shapes, grids, noise)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = np.zeros((size, size, 3), np.uint8)
        k = i % 5
        if k == 0:
            a[:, :, 0] = np.linspace(0, 255, size)[None, :]
            a[:, :, 1] = np.linspace(0, 255, size)[:, None]
        elif k == 1:
            block = max(size // 8, 1)
            a = (
                rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                .repeat(block, 0)
                .repeat(block, 1)[:size, :size]
            )
        elif k == 2:
            yy, xx = np.mgrid[0:size, 0:size]
            m = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 < (40 + i * 5) ** 2
            a[m] = rng.integers(0, 255, 3)
            a[~m] = rng.integers(0, 255, 3)
        elif k == 3:
            a[:, ::3] = rng.integers(0, 255, 3)
            a[::3, :] = rng.integers(0, 255, 3)
        else:
            a = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        out.append(Image.fromarray(a))
    return out


def synthetic_calibration_audio(
    n: int = 8, seconds: float = 2.0, sample_rate: int = 16000, seed: int = 0
):
    """Varied synthetic mono waveforms (tones, square waves, noise, chirps)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            w = np.sin(2 * np.pi * (110 + 40 * i) * t)
        elif k == 1:
            w = np.sign(np.sin(2 * np.pi * (220 + 30 * i) * t))
        elif k == 2:
            w = rng.standard_normal(t.shape)
        else:
            w = np.sin(2 * np.pi * (300 + 50 * i) * t) * np.exp(-t)
        out.append((0.3 * w).astype(np.float32))
    return out


def load_calibration_media(path: str):
    """Load images and mono waveforms from a directory of calibration files."""
    import os

    from PIL import Image

    from ..utils import read_audio

    images, audios = [], []
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        low = name.lower()
        if low.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            images.append(Image.open(p).convert("RGB"))
        elif low.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")):
            audios.append(read_audio(p)[0])
    return images, audios
