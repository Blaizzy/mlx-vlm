"""Strict loading of the original Idol safetensors checkpoints."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...utils import get_model_path

WEIGHT_FILES = [
    "config.json",
    "denoiser.safetensors",
    "decoder.safetensors",
    "decoder_gs.safetensors",
    "birefnet.safetensors",
    "README.md",
]


def resolve_model(
    repo_or_dir="speridlabs/idol",
    *,
    local_files_only=False,
    revision=None,
    force_download=False,
):
    return get_model_path(
        str(Path(repo_or_dir).expanduser()),
        allow_patterns=WEIGHT_FILES,
        local_files_only=local_files_only,
        revision=revision,
        force_download=force_download,
    )


def load_weights(model, weights, *, dtype=None, mixed_precision=False):
    """Map Sequential indices and OIHW convolutions; reject missing/extra keys.

    Idol keeps embeddings, norms and learned RoPE in float32 and casts only
    transformer Linear layers to float16. This is significant for RePo angles.
    """
    if isinstance(weights, (str, Path)):
        weights = mx.load(str(weights))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    target = dict(tree_flatten(model.parameters()))
    # Strip only Sequential's extra component, not a backbone's layers list.
    names = {key: key for key in target}
    half_keys = set()
    for path, module in sorted(
        model.named_modules(), key=lambda item: len(item[0]), reverse=True
    ):
        if isinstance(module, nn.Sequential):
            prefix = f"{path}.layers." if path else "layers."
            for key in target:
                if key.startswith(prefix):
                    names[key] = names[key].replace(prefix, f"{path}." if path else "")
        if (
            mixed_precision
            and isinstance(module, nn.Linear)
            and path.startswith(("blocks.", "noise_refiner.", "context_refiner."))
        ):
            half_keys.update(f"{path}.{name}" for name in ("weight", "bias"))
    converted = []
    consumed = set()
    missing = []
    for key, expected in target.items():
        source = names[key]
        if source not in weights:
            missing.append(source)
            continue
        value = weights[source]
        consumed.add(source)
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        if value.shape != expected.shape:
            raise ValueError(
                f"Wrong shape for {source}: {value.shape}, expected {expected.shape}"
            )
        if mx.issubdtype(value.dtype, mx.floating):
            value = value.astype(
                dtype or (mx.float16 if key in half_keys else mx.float32)
            )
        converted.append((key, value))
    extra = set(weights) - consumed
    # PyTorch's batch counter is not used at inference.
    extra = {key for key in extra if not key.endswith(".num_batches_tracked")}
    if missing or extra:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={sorted(extra)}"
        )
    model.load_weights(converted, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
