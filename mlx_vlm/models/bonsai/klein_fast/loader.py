from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from mlx_vlm.models.bonsai.klein_fast.blocks import (
    DoubleBlockWeights,
    PackedWeight,
    SingleBlockWeights,
    WeightOrPacked,
    double_block_weight_keys,
    single_block_weight_keys,
)
from mlx_vlm.models.bonsai.klein_fast.megakernel import (
    Flux2KleinMegakernelSpec,
    MegakernelWeights,
)

_GLOBAL_KEYS: dict[str, str] = {
    "x_embedder": "x_embedder.weight",
    "context_embedder": "context_embedder.weight",
    "norm_out_linear": "norm_out.linear.weight",
    "proj_out": "proj_out.weight",
}

_SHARED_MODULATION_KEYS: dict[str, str] = {
    "double_mod_img": "double_stream_modulation_img.linear.weight",
    "double_mod_txt": "double_stream_modulation_txt.linear.weight",
    "single_mod": "single_stream_modulation.linear.weight",
}

_PACKED_DIR_NAME = "transformer-packed-mflux"
_QUANT_CONFIG_NAME = "quantization_config.json"


def _get_and_cast(raw: dict[str, mx.array], key: str, dtype: mx.Dtype) -> mx.array:
    if key not in raw:
        raise KeyError(f"Missing tensor '{key}' in transformer checkpoint")
    tensor = raw[key]
    return tensor if tensor.dtype == dtype else tensor.astype(dtype)


def _load_raw_safetensors(transformer_dir: Path) -> dict[str, mx.array]:
    shards = sorted(
        p for p in transformer_dir.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not shards:
        raise FileNotFoundError(f"No safetensors files found in {transformer_dir}")
    merged: dict[str, mx.array] = {}
    for shard in shards:
        merged.update(mx.load(str(shard)))
    return merged


def load_klein_fast_weights_from_hf(
    transformer_dir: Path,
    spec: Flux2KleinMegakernelSpec,
    dtype: mx.Dtype = mx.bfloat16,
) -> MegakernelWeights:
    """Map HF FLUX.2 Klein transformer weights into MegakernelWeights."""
    raw = _load_raw_safetensors(Path(transformer_dir))
    return _build_megakernel_weights_from_raw(raw, spec, dtype)


def _build_megakernel_weights_from_raw(
    raw: dict[str, mx.array],
    spec: Flux2KleinMegakernelSpec,
    dtype: mx.Dtype,
) -> MegakernelWeights:
    """Pure mapping logic; unit-testable without touching the filesystem."""
    shared_double_mod_img = _get_and_cast(
        raw, _SHARED_MODULATION_KEYS["double_mod_img"], dtype
    )
    shared_double_mod_txt = _get_and_cast(
        raw, _SHARED_MODULATION_KEYS["double_mod_txt"], dtype
    )
    shared_single_mod = _get_and_cast(raw, _SHARED_MODULATION_KEYS["single_mod"], dtype)

    double_block_weights: list[DoubleBlockWeights] = []
    for i in range(spec.num_double_blocks):
        keys = double_block_weight_keys(i)
        fields: dict[str, mx.array] = {}
        for field_name, hf_key in keys.items():
            if field_name == "modulation_img":
                fields[field_name] = shared_double_mod_img
            elif field_name == "modulation_txt":
                fields[field_name] = shared_double_mod_txt
            else:
                fields[field_name] = _get_and_cast(raw, hf_key, dtype)
        double_block_weights.append(DoubleBlockWeights(**fields))

    single_block_weights: list[SingleBlockWeights] = []
    for i in range(spec.num_single_blocks):
        keys = single_block_weight_keys(i)
        fields = {}
        for field_name, hf_key in keys.items():
            if field_name == "modulation":
                fields[field_name] = shared_single_mod
            else:
                fields[field_name] = _get_and_cast(raw, hf_key, dtype)
        single_block_weights.append(SingleBlockWeights(**fields))

    return MegakernelWeights(
        x_embedder=_get_and_cast(raw, _GLOBAL_KEYS["x_embedder"], dtype),
        context_embedder=_get_and_cast(raw, _GLOBAL_KEYS["context_embedder"], dtype),
        norm_out_linear=_get_and_cast(raw, _GLOBAL_KEYS["norm_out_linear"], dtype),
        proj_out=_get_and_cast(raw, _GLOBAL_KEYS["proj_out"], dtype),
        double_block_weights=double_block_weights,
        single_block_weights=single_block_weights,
    )


def _load_packed_or_dense_linear(
    raw: dict[str, mx.array],
    hf_weight_key: str,
    *,
    bits: int,
    group_size: int,
    dtype: mx.Dtype,
) -> WeightOrPacked:
    """Return PackedWeight if the layer has sibling `.scales`/`.biases`; else bf16."""
    if not hf_weight_key.endswith(".weight"):
        raise ValueError(f"Expected key to end in '.weight': {hf_weight_key}")
    layer_prefix = hf_weight_key[: -len(".weight")]
    scales_key = f"{layer_prefix}.scales"
    if scales_key not in raw:
        return _get_and_cast(raw, hf_weight_key, dtype)
    biases_key = f"{layer_prefix}.biases"
    if biases_key not in raw:
        raise KeyError(
            f"Found '{scales_key}' but missing '{biases_key}' in packed artifact"
        )
    if hf_weight_key not in raw:
        raise KeyError(f"Missing packed weight '{hf_weight_key}' in packed artifact")
    packed = raw[hf_weight_key]
    if packed.dtype != mx.uint32:
        raise ValueError(
            f"Packed weight '{hf_weight_key}' must be uint32, got {packed.dtype}"
        )
    return PackedWeight(
        packed=packed,
        scales=raw[scales_key].astype(dtype),
        biases=raw[biases_key].astype(dtype),
        bits=bits,
        group_size=group_size,
    )


def _read_packed_quant_config(packed_dir: Path) -> tuple[int, int]:
    config_path = packed_dir / _QUANT_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Packed artifact missing {_QUANT_CONFIG_NAME}: {config_path}"
        )
    cfg = json.loads(config_path.read_text())
    bits = int(cfg["bits"])
    group_size = int(cfg["group_size"])
    return bits, group_size


def load_klein_fast_packed_weights_from_disk(
    packed_dir: Path,
    spec: Flux2KleinMegakernelSpec,
    dtype: mx.Dtype = mx.bfloat16,
) -> MegakernelWeights:
    """Load a pre-packed quantized artifact directly into MegakernelWeights.

    Expected layout:

        packed_dir/
          diffusion_pytorch_model.safetensors   # uint32 packed + bf16 scales/biases/skips
          quantization_config.json              # bits, group_size, skip_patterns
          config.json, checkpoint_manifest.json # (passthrough; not consumed here)

    Block linears outside the skip list are returned as PackedWeight triples.
    Skip-pattern layers and the 4 outer linears remain plain bf16 mx.array.
    """
    packed_dir = Path(packed_dir)
    bits, group_size = _read_packed_quant_config(packed_dir)
    raw = _load_raw_safetensors(packed_dir)
    return _build_megakernel_weights_from_packed(
        raw, spec, dtype=dtype, bits=bits, group_size=group_size
    )


def _build_megakernel_weights_from_packed(
    raw: dict[str, mx.array],
    spec: Flux2KleinMegakernelSpec,
    *,
    dtype: mx.Dtype,
    bits: int,
    group_size: int,
) -> MegakernelWeights:
    shared_double_mod_img = _get_and_cast(
        raw, _SHARED_MODULATION_KEYS["double_mod_img"], dtype
    )
    shared_double_mod_txt = _get_and_cast(
        raw, _SHARED_MODULATION_KEYS["double_mod_txt"], dtype
    )
    shared_single_mod = _get_and_cast(raw, _SHARED_MODULATION_KEYS["single_mod"], dtype)

    def load_block_linear(hf_key: str) -> WeightOrPacked:
        return _load_packed_or_dense_linear(
            raw, hf_key, bits=bits, group_size=group_size, dtype=dtype
        )

    double_block_weights: list[DoubleBlockWeights] = []
    for i in range(spec.num_double_blocks):
        keys = double_block_weight_keys(i)
        fields: dict = {}
        for field_name, hf_key in keys.items():
            if field_name == "modulation_img":
                fields[field_name] = shared_double_mod_img
            elif field_name == "modulation_txt":
                fields[field_name] = shared_double_mod_txt
            elif field_name.startswith("norm_"):
                fields[field_name] = _get_and_cast(raw, hf_key, dtype)
            else:
                fields[field_name] = load_block_linear(hf_key)
        double_block_weights.append(DoubleBlockWeights(**fields))

    single_block_weights: list[SingleBlockWeights] = []
    for i in range(spec.num_single_blocks):
        keys = single_block_weight_keys(i)
        fields = {}
        for field_name, hf_key in keys.items():
            if field_name == "modulation":
                fields[field_name] = shared_single_mod
            elif field_name.startswith("norm_"):
                fields[field_name] = _get_and_cast(raw, hf_key, dtype)
            else:
                fields[field_name] = load_block_linear(hf_key)
        single_block_weights.append(SingleBlockWeights(**fields))

    return MegakernelWeights(
        x_embedder=_get_and_cast(raw, _GLOBAL_KEYS["x_embedder"], dtype),
        context_embedder=_get_and_cast(raw, _GLOBAL_KEYS["context_embedder"], dtype),
        norm_out_linear=_get_and_cast(raw, _GLOBAL_KEYS["norm_out_linear"], dtype),
        proj_out=_get_and_cast(raw, _GLOBAL_KEYS["proj_out"], dtype),
        double_block_weights=double_block_weights,
        single_block_weights=single_block_weights,
    )


def find_packed_artifact_dir(root_path: Path) -> Path | None:
    """Return the `transformer-packed-mflux/` directory if it exists, else None."""
    candidate = Path(root_path) / _PACKED_DIR_NAME
    if candidate.is_dir() and (candidate / _QUANT_CONFIG_NAME).exists():
        return candidate
    return None


__all__ = [
    "find_packed_artifact_dir",
    "load_klein_fast_packed_weights_from_disk",
    "load_klein_fast_weights_from_hf",
]
