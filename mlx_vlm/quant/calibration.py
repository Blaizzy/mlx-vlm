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
