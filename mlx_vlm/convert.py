import argparse
import glob
import shutil
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .quant_utils import get_quantization_params
from .utils import (
    MODEL_CONVERSION_DTYPES,
    create_model_card,
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
    upload_to_hub,
)

QUANT_RECIPES = [
    "mixed_2_6",
    "mixed_3_4",
    "mixed_3_5",
    "mixed_3_6",
    "mixed_3_8",
    "mixed_4_6",
    "mixed_4_8",
]


def _preserve_existing_deepseek_v4_quantization(
    config: dict,
    model: nn.Module,
    q_group_size: Optional[int],
    q_bits: Optional[int],
    q_mode: str,
):
    quantization_config = config.get("quantization_config", {})
    if (
        config.get("model_type") != "deepseek_v4"
        or "quantization" in config
        or not isinstance(quantization_config, dict)
        or quantization_config.get("quant_method") != "fp8"
    ):
        return

    from .models.deepseek_v4.language import make_quantization_config

    quantization = make_quantization_config(model)
    quantization.update(get_quantization_params(q_group_size, q_bits, q_mode))
    config["quantization"] = quantization
    config["quantization_config"] = quantization


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    group_size = 64

    recipe_config = {
        "mixed_2_6": (2, 6),
        "mixed_3_4": (3, 4),
        "mixed_3_5": (3, 5),
        "mixed_3_6": (3, 6),
        "mixed_3_8": (3, 8),
        "mixed_4_6": (4, 6),
        "mixed_4_8": (4, 8),
    }

    if recipe not in recipe_config:
        raise ValueError(f"Invalid quant recipe {recipe}")

    low_bits, high_bits = recipe_config[recipe]

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if skip_multimodal_module(path):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[1] % group_size != 0:
            return False

        path_parts = path.split(".")
        index = 0

        if len(path_parts) > layer_location:
            element = path_parts[layer_location]
            if element.isdigit():
                index = int(element)

        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )

        if use_more_bits and ("v_proj" in path or "down_proj" in path):
            return {"group_size": group_size, "bits": high_bits}

        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


def _has_decoder(module):
    if module is None:
        return False
    for _, sub in module.named_modules():
        if getattr(sub, "self_attn", None) is not None and getattr(sub, "mlp", None):
            return True
    return False


def _run_model_inputs(model, inputs):
    """Run ``model`` on a prepared ``prepare_inputs`` dict, returning its output."""
    extra = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }
    return model(
        inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        mask=inputs.get("attention_mask"),
        **extra,
    )


def _cap_image_pixels(image, max_side=1024):
    """Downscale a calibration image so its longest side is at most ``max_side``.

    Bounds the per-image vision-token count (a full-resolution photo can expand
    to tens of thousands of tokens, which overflows the distillation backward).
    """
    from PIL import Image

    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.BICUBIC,
    )


def _build_calibration_samples(model, processor, config, calibration_data):
    """Prepared multimodal (image/audio + text) calibration inputs, or ``[]``.

    Returns an empty list when the model has no vision/audio modality so callers
    can fall back to text calibration.
    """
    from .prompt_utils import apply_chat_template
    from .quant import (
        load_calibration_media,
        synthetic_calibration_audio,
        synthetic_calibration_images,
    )
    from .utils import prepare_inputs

    has_vision = bool(config.get("vision_config")) or (
        getattr(model, "vision_tower", None) is not None
    )
    has_audio = bool(config.get("audio_config")) or (
        getattr(model, "audio_tower", None) is not None
    )
    if not has_vision and not has_audio:
        return []

    if calibration_data:
        images, audios = load_calibration_media(calibration_data)
    else:
        images = synthetic_calibration_images(8) if has_vision else []
        audios = synthetic_calibration_audio(8) if has_audio else []
        print(
            "[INFO] Using synthetic calibration media; pass "
            "--calibration-data for real image/audio samples."
        )

    images = [_cap_image_pixels(im) for im in images]

    media = [(im, None, 1, 0) for im in images]
    media += [(None, au, 0, 1) for au in audios]

    cfg = model.config
    samples = []
    for image, audio, n_img, n_aud in media:
        prompt = (
            "Describe this image in detail."
            if n_img
            else "Describe what you hear in this audio."
        )
        formatted = apply_chat_template(
            processor, cfg, prompt, num_images=n_img, num_audios=n_aud
        )
        samples.append(
            prepare_inputs(
                processor,
                images=[image] if image is not None else None,
                audio=[audio] if audio is not None else None,
                prompts=formatted,
                image_token_index=getattr(cfg, "image_token_index", None),
                add_special_tokens=False,
                pad_to_uniform_size=False,
            )
        )
    return samples


def _build_multimodal_awq_run(model, processor, config, calibration_data):
    """Build a calibration forward that routes media+text through the full model.

    Returns ``(run, n_samples)``, or ``(None, 0)`` when the model has no
    vision/audio modality so the caller falls back to text calibration.
    """
    samples = _build_calibration_samples(model, processor, config, calibration_data)
    if not samples:
        return None, 0

    def run():
        for inputs in samples:
            mx.eval(_run_model_inputs(model, inputs))

    return run, len(samples)


def _build_dwq_calibration(
    model, processor, config, calibration, calibration_data, freeze_prefix=False
):
    """Build ``(inputs, forward)`` for DWQ: calibration samples + a logits forward.

    Uses multimodal samples when requested and available, otherwise falls back
    to the default text prompts. ``forward`` maps a sample to language-model
    logits through the model's own path.

    With ``freeze_prefix`` and a multimodal model, the frozen vision/embedding
    prefix is run once per sample and the fused embeddings are cached (detached),
    so distillation re-runs only the trainable decoder -- a large speedup when
    the vision tower is kept full precision, at the cost of not tuning the
    embedding/vision quantization.
    """
    inputs = []
    if calibration == "multimodal":
        inputs = _build_calibration_samples(model, processor, config, calibration_data)

    if not inputs:
        from .quant import DEFAULT_CALIBRATION_TEXT

        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = [
            {"input_ids": mx.array([tokenizer.encode(text)])}
            for text in DEFAULT_CALIBRATION_TEXT
        ]

    can_cache = (
        freeze_prefix
        and inputs
        and inputs[0].get("pixel_values") is not None
        and hasattr(model, "get_input_embeddings")
    )
    if can_cache:
        cached = []
        for mi in inputs:
            rest = {
                k: v
                for k, v in mi.items()
                if k not in ("input_ids", "pixel_values", "attention_mask")
                and v is not None
            }
            features = model.get_input_embeddings(
                mi["input_ids"],
                mi.get("pixel_values"),
                mask=mi.get("attention_mask"),
                **rest,
            )
            extra = {
                k: v
                for k, v in features.to_dict().items()
                if v is not None and k != "inputs_embeds"
            }
            entry = {
                "input_ids": mi["input_ids"],
                "attention_mask": mi.get("attention_mask"),
                "inputs_embeds": mx.stop_gradient(features.inputs_embeds),
                "_extra": extra,
            }
            mx.eval(entry["inputs_embeds"])
            cached.append(entry)

        def forward(sample):
            out = model.language_model(
                sample["input_ids"],
                mask=sample.get("attention_mask"),
                inputs_embeds=sample["inputs_embeds"],
                **sample["_extra"],
            )
            return out.logits if hasattr(out, "logits") else out

        return cached, forward

    def forward(sample):
        out = _run_model_inputs(model, sample)
        return out.logits if hasattr(out, "logits") else out

    return inputs, forward


def _apply_awq_calibration(
    model,
    processor,
    config,
    target,
    q_bits,
    q_group_size,
    calibration="text",
    calibration_data=None,
):
    """Calibrate (text or multimodal) and apply AWQ scaling to the decoder."""
    from .quant import DEFAULT_CALIBRATION_TEXT, apply_awq, collect_activation_stats

    tokenizer = getattr(processor, "tokenizer", processor)
    language_model = getattr(model, "language_model", None)
    root = language_model if _has_decoder(language_model) else target

    if calibration == "multimodal":
        run, n = _build_multimodal_awq_run(model, processor, config, calibration_data)
        if run is not None:
            print(f"[INFO] AWQ: multimodal calibration on {n} media samples.")
            stats = collect_activation_stats(root, run)
            summary = apply_awq(
                root, stats, bits=q_bits or 4, group_size=q_group_size or 64
            )
            print(f"[INFO] AWQ scaling applied: {summary}")
            return
        print("[INFO] AWQ: no vision/audio modality found; using text calibration.")

    probe = mx.array([tokenizer.encode(DEFAULT_CALIBRATION_TEXT[0])])
    forward = None
    for candidate in (getattr(root, "model", None), root, language_model):
        if candidate is None:
            continue
        try:
            mx.eval(candidate(probe))
            forward = candidate
            break
        except Exception:
            continue
    if forward is None:
        raise RuntimeError("Could not run a calibration forward pass for AWQ.")

    def run():
        for text in DEFAULT_CALIBRATION_TEXT:
            mx.eval(forward(mx.array([tokenizer.encode(text)])))

    stats = collect_activation_stats(root, run)
    summary = apply_awq(root, stats, bits=q_bits or 4, group_size=q_group_size or 64)
    print(f"[INFO] AWQ scaling applied: {summary}")


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    q_mode: str = "affine",
    quant_method: str = "rtn",
    calibration: str = "text",
    calibration_data: Optional[str] = None,
    dwq_steps: int = 200,
    dwq_lr: float = 1e-6,
    dwq_val_size: int = 4,
    dwq_top_k: int = 0,
    dwq_patience: int = 0,
    dwq_checkpoint: bool = False,
    dwq_freeze_prefix: bool = False,
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    trust_remote_code: bool = True,
    quant_predicate: Optional[str] = None,
    mtp: bool = False,
    mtp_output: Optional[str] = None,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    model_quant_predicate = getattr(model, "quant_predicate", None)

    def base_quant_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        if model_quant_predicate is not None:
            return model_quant_predicate(path, module)
        return True

    target = model

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, target)

    quant_predicate = quant_predicate or base_quant_predicate

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        target.update(tree_map_with_path(set_dtype, target.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        from .quant_utils import quantize_model

        _preserve_existing_deepseek_v4_quantization(
            config, target, q_group_size, q_bits, q_mode
        )

        do_awq = "awq" in quant_method
        do_dwq = "dwq" in quant_method

        dwq_forward = dwq_train = dwq_val = teacher_train = teacher_val = None
        if do_dwq:
            from .quant import capture_teacher

            dwq_inputs, dwq_forward = _build_dwq_calibration(
                model,
                processor,
                config,
                calibration,
                calibration_data,
                freeze_prefix=dwq_freeze_prefix,
            )
            if dwq_val_size > 0 and len(dwq_inputs) > dwq_val_size:
                dwq_train = dwq_inputs[:-dwq_val_size]
                dwq_val = dwq_inputs[-dwq_val_size:]
            else:
                dwq_train, dwq_val = dwq_inputs, []
            print(
                f"[INFO] Capturing DWQ teacher ({len(dwq_train)} train, "
                f"{len(dwq_val)} val samples)"
            )
            # Capture the teacher through the same (differentiable) path the
            # student uses during distillation, so the only difference measured
            # is quantization error.
            model.train()
            try:
                teacher_train = capture_teacher(dwq_forward, dwq_train, top_k=dwq_top_k)
                teacher_val = capture_teacher(dwq_forward, dwq_val, top_k=dwq_top_k)
            finally:
                model.eval()

        if do_awq:
            print("[INFO] Calibrating (AWQ)")
            _apply_awq_calibration(
                model,
                processor,
                config,
                target,
                q_bits,
                q_group_size,
                calibration=calibration,
                calibration_data=calibration_data,
            )

        print("[INFO] Quantizing")
        config.setdefault("vision_config", {})
        target, config = quantize_model(
            target,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

        if do_dwq:
            from .quant import apply_dwq

            print("[INFO] Distilling (DWQ)")
            summary = apply_dwq(
                target,
                dwq_forward,
                dwq_train,
                teacher_train,
                steps=dwq_steps,
                lr=dwq_lr,
                val_inputs=dwq_val,
                val_teacher=teacher_val,
                patience=dwq_patience,
                checkpoint=dwq_checkpoint,
            )
            if summary.get("initial_loss") is not None:
                print(
                    f"[INFO] DWQ validation CE to teacher: "
                    f"RTN baseline {summary['initial_loss']:.4f} -> "
                    f"DWQ {summary['final_loss']:.4f}"
                )
            print(f"[INFO] DWQ applied: {summary}")

    if dequantize:
        from .quant_utils import dequantize_model

        print("[INFO] Dequantizing")
        target = dequantize_model(target)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    save_weights(mlx_path, target, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            # Skip the index file - save_weights() already generated the correct one
            if Path(file).name == "model.safetensors.index.json":
                continue
            shutil.copy(file, mlx_path)

    # Copy folders from the model path to the MLX path
    for item in model_path.iterdir():
        if item.is_dir():
            dest = mlx_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # Not every remote-code processor inherits ProcessorMixin — Mage-VL's `MageVLProcessor`
    # deliberately does not ("We deliberately do NOT inherit transformers.ProcessorMixin"), so it
    # has no save_pretrained. The weights are already written by this point; losing the whole
    # conversion to the sidecar-copy step would be absurd. Fall back to copying the processor
    # files verbatim, which is what save_pretrained would have produced anyway.
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(mlx_path)
    else:
        # NOTE: no local `import shutil` here — convert.py already imports it at module scope,
        # and a function-local import would make the name local for the WHOLE function, unbinding
        # the earlier uses. That failure reads as "cannot access local variable 'shutil'".
        src = Path(hf_path)
        if src.is_dir():
            for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
                for f in src.glob(pattern):
                    if f.name not in ("config.json", "model.safetensors.index.json"):
                        shutil.copy2(f, Path(mlx_path) / f.name)
        print("[INFO] processor lacks save_pretrained; copied processor files verbatim")

    save_config(config, config_path=mlx_path / "config.json")

    if mtp:
        try:
            from .speculative.drafters.mtp_split import detect_mtp_splitter

            splitter = detect_mtp_splitter(model_path)
            if splitter is None:
                print(
                    "[INFO] --mtp: no native MTP tensors / registered splitter for "
                    "this model; skipping drafter"
                )
            else:
                drafter_path = mtp_output or f"{mlx_path}-mtp"
                print(f"[INFO] Extracting MTP drafter -> {drafter_path}")
                splitter.split(
                    str(model_path),
                    str(drafter_path),
                    q_bits=q_bits if quantize else None,
                    q_group_size=q_group_size,
                )
        except Exception as exc:
            # the base conversion already succeeded; a drafter failure must not
            # take the whole convert down with it
            print(
                f"[WARNING] --mtp: failed to extract MTP drafter "
                f"({type(exc).__name__}: {exc}); base conversion is unaffected"
            )

    hf_repo = None if Path(hf_path).exists() else hf_path
    create_model_card(mlx_path, hf_repo)

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Hugging Face revision (branch), when converting a model from the Hub.",
        default=None,
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
    )
    parser.add_argument(
        "--quant-method",
        help="Weight quantization method.",
        type=str,
        choices=["rtn", "awq", "dwq", "awq+dwq"],
        default="rtn",
    )
    parser.add_argument(
        "--calibration",
        help="AWQ/DWQ calibration inputs: text (default) or multimodal (image/audio+text).",
        type=str,
        choices=["text", "multimodal"],
        default="text",
    )
    parser.add_argument(
        "--calibration-data",
        help="Optional directory of real images/audio for --calibration multimodal.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dwq-steps",
        help="Distillation steps for DWQ quantization methods.",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--dwq-lr",
        help="Distillation learning rate for DWQ quantization methods.",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--dwq-val-size",
        help="Held-out calibration samples used to measure DWQ validation loss.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dwq-top-k",
        help="Distill against the teacher's top-k logits (0 = full vocabulary).",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dwq-patience",
        help="Stop DWQ early after this many reports with no val improvement (0 = off).",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dwq-checkpoint",
        help="Gradient-checkpoint decoder layers during DWQ (less memory, slower).",
        action="store_true",
    )
    parser.add_argument(
        "--dwq-freeze-prefix",
        help="Cache the frozen vision/embedding prefix and distill only the decoder.",
        action="store_true",
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameter. Defaults to config.json's `torch_dtype` or the current model weights dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mtp",
        help="Also extract the model's native MTP tensors into a standalone drafter.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mtp-output",
        help="Output path for the MTP drafter (default: <mlx-path>-mtp).",
        type=str,
        default=None,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.convert ...` directly is deprecated."
        " Use `mlx_vlm.convert ...` or `python -m mlx_vlm convert ...` instead."
    )
    main()
