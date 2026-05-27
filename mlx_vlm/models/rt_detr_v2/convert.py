"""Convert HuggingFace RT-DETRv2 checkpoints to MLX safetensors format.

The HF `config.json` is consumed as-is; the only transformation is on
the weights themselves (key renames + Conv2d NCHW->NHWC transpose), then
optional quantization.

Usage:
    python -m mlx_vlm.models.rt_detr_v2.convert \\
        --hf-path docling-project/docling-layout-heron \\
        --output rt-detr-v2-heron-mlx \\
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# `_HF_PREFIX` is stripped from every key before the rename pipeline runs.
_HF_PREFIX = "model."

# Phase-specific renames applied in order, AFTER `_HF_PREFIX` is stripped.
# Each entry is (regex pattern, replacement). Order matters: rules can
# rely on earlier renames having fired.
RENAMES: List[Tuple[str, str]] = [
    # Backbone: HF wraps the ResNet body in `backbone.model.X`; we surface
    # it as `vision.backbone.X`.
    (r"^backbone\.model\.", "vision.backbone."),
    # vd downsampling shortcut: HF Sequential[AvgPool, ShortCut] indexes
    # the inner ShortCut at `.1.` (AvgPool sits at `.0.` with no params).
    (r"\.shortcut\.1\.", ".shortcut.proj."),
    # RTDetrResNetConvLayer field names: convolution -> conv, normalization -> bn.
    (r"\.convolution\.", ".conv."),
    (r"\.normalization\.", ".bn."),
    # AIFI keys on disk are prefixed `encoder.encoder.` (older HF naming);
    # HF runtime auto-remaps to `encoder.aifi.`, we do the same explicitly.
    (r"^encoder\.encoder\.", "vision.hybrid_encoder.aifi."),
    # encoder_input_proj is Sequential[Conv, BN]: `.{N}.0.X` -> `.{N}.conv.X`
    # and `.{N}.1.X` -> `.{N}.bn.X`.
    (r"^encoder_input_proj\.(\d+)\.0\.", r"vision.encoder_input_proj.\1.conv."),
    (r"^encoder_input_proj\.(\d+)\.1\.", r"vision.encoder_input_proj.\1.bn."),
    # Hybrid encoder body (FPN/PAN/laterals/downsamples): all live under `encoder.*`.
    (r"^encoder\.", "vision.hybrid_encoder."),
    # RTDetrV2ConvNormLayer field rename (encoder body uses `.norm.` rather
    # than the backbone's `.normalization.`).
    (r"\.norm\.", ".bn."),
    # decoder_input_proj is also Sequential[Conv, BN].
    (r"^decoder_input_proj\.(\d+)\.0\.", r"decoder_input_proj.\1.conv."),
    (r"^decoder_input_proj\.(\d+)\.1\.", r"decoder_input_proj.\1.bn."),
    # enc_output is Sequential[Linear, LayerNorm].
    (r"^enc_output\.0\.", "enc_output.fc."),
    (r"^enc_output\.1\.", "enc_output.ln."),
]

# Keys excluded from the MLX checkpoint. MLX `nn.BatchNorm` has no slot
# for `num_batches_tracked`; trainers re-initialise this counter on resume.
DROP_PATTERNS: List[str] = [
    r"\.num_batches_tracked$",
]

# Keys kept in the MLX checkpoint but flagged as training-only — the
# inference forward path doesn't read them, but trainers (LoRA, full
# fine-tune) need them.
TRAINING_ONLY_PATTERNS: List[str] = [
    r"^denoising_class_embed",
]


def _strip_prefix(key: str) -> str:
    return key[len(_HF_PREFIX) :] if key.startswith(_HF_PREFIX) else key


def rename(key: str) -> str:
    out = _strip_prefix(key)
    for pat, repl in RENAMES:
        out = re.sub(pat, repl, out)
    return out


def should_drop(key: str) -> bool:
    stripped = _strip_prefix(key)
    return any(re.search(p, stripped) for p in DROP_PATTERNS)


def is_training_only(key: str) -> bool:
    stripped = _strip_prefix(key)
    return any(re.search(p, stripped) for p in TRAINING_ONLY_PATTERNS)


def convert(hf_path: str, output: str, dtype: str = "bfloat16") -> Path:
    """Convert a HuggingFace RT-DETRv2 checkpoint to MLX format.

    Args:
        hf_path: HF repo id (e.g. "docling-project/docling-layout-heron")
            or a local directory containing model.safetensors + config.json.
        output: destination directory for the MLX checkpoint.
        dtype: target dtype for the saved weights. One of the names in
            `mlx_vlm.utils.MODEL_CONVERSION_DTYPES`. Default bfloat16.
    Returns:
        Path to the written output directory.
    """
    try:
        import mlx.core as mx
        import safetensors.torch
        from huggingface_hub import snapshot_download

        from ...utils import MODEL_CONVERSION_DTYPES
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(2)

    if dtype not in MODEL_CONVERSION_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; expected one of {MODEL_CONVERSION_DTYPES}"
        )
    mlx_dtype = getattr(mx, dtype)

    src = Path(hf_path)
    if not src.exists():
        src = Path(snapshot_download(hf_path))
    ckpt_file = src / "model.safetensors"
    cfg_file = src / "config.json"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"No model.safetensors in {src}")
    if not cfg_file.is_file():
        raise FileNotFoundError(f"No config.json in {src}")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    raw = safetensors.torch.load_file(str(ckpt_file))
    print(f"Loaded {len(raw)} tensors from {ckpt_file}")
    sanitized: Dict[str, mx.array] = {}
    n_dropped = 0
    for k, v in raw.items():
        if should_drop(k):
            n_dropped += 1
            continue
        new_k = rename(k)
        arr = mx.array(v.detach().cpu().numpy())
        if new_k.endswith(".conv.weight") and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        sanitized[new_k] = arr.astype(mlx_dtype)
    print(f"  renamed {len(sanitized)} tensors, dropped {n_dropped}, cast to {dtype}")

    weights_path = out / "model.safetensors"
    # `metadata={"format": "mlx"}` is what `mlx_vlm.utils.load_model` keys on
    # to skip `Model.sanitize` — without it the loader would re-run the
    # rename + NCHW->NHWC transpose pipeline against already-MLX weights.
    mx.save_safetensors(str(weights_path), sanitized, metadata={"format": "mlx"})
    print(f"  wrote {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    # Copy config.json + preprocessor_config.json + any tokenizer files.
    for name in ("config.json", "preprocessor_config.json"):
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, out / name)
            print(f"  copied {name}")

    # Verification: load it back and run a forward pass.
    _verify(out)
    return out


def _verify(out: Path) -> None:
    """Smoke-test the converted checkpoint via the framework loader.

    Goes through `mlx_vlm.utils.load_model` (not raw `mx.load`) so the
    sanitize-metadata path is exercised — a saved file without
    `metadata['format'] = 'mlx'` would re-trigger `Model.sanitize`
    and double-transpose conv weights, and this verify catches it.
    """
    import mlx.core as mx

    from ...utils import load_model

    config = json.loads((out / "config.json").read_text())
    model = load_model(out)
    model.eval()

    size = config.get("image_size") or 640
    # Random input rather than zeros: zeros let weight-transpose bugs hide
    # because biases dominate.
    pixel = mx.random.normal((1, size, size, 3), dtype=mx.float32)
    out_dict = model(pixel)
    mx.eval(out_dict["pred_logits"], out_dict["pred_boxes"])
    print(
        f"  verified forward: pred_logits {tuple(out_dict['pred_logits'].shape)}, "
        f"pred_boxes {tuple(out_dict['pred_boxes'].shape)}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    from ...utils import MODEL_CONVERSION_DTYPES

    parser = argparse.ArgumentParser(description="Convert HF RT-DETRv2 -> MLX.")
    parser.add_argument(
        "--hf-path",
        required=True,
        help="HF repo id (e.g. docling-project/docling-layout-heron) or local dir",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the MLX checkpoint"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=MODEL_CONVERSION_DTYPES,
        help="Target dtype for saved weights (default bfloat16)",
    )
    args = parser.parse_args(argv)
    convert(args.hf_path, args.output, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
