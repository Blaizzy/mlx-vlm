"""Export a DeepSeek-V4 Flash Vision fixture from the official runtime.

Run this script in an environment containing the official inference
dependencies. The official repository must be checked out locally and its
checkpoint converted to a single tensor-parallel rank first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _official_modules(inference_dir: Path):
    encoding_dir = inference_dir.parent / "encoding"
    for path in (encoding_dir, inference_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing official reference directory: {path}")
        sys.path.insert(0, str(path))

    from encoding_dsv4 import IMAGE_PLACEHOLDER
    from image_processor import prepare_vl_inputs
    from model import ModelArgs, Transformer

    return IMAGE_PLACEHOLDER, prepare_vl_inputs, ModelArgs, Transformer


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def export_fixture(args: argparse.Namespace) -> Path:
    try:
        import torch
        from safetensors.torch import load_model
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the official inference/requirements.txt before exporting "
            "a DeepSeek-V4 reference fixture"
        ) from exc

    inference_dir = args.official_inference_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else inference_dir / "config.json"
    )
    image_path = args.image.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Fixture already exists: {output}")

    image_placeholder, prepare_vl_inputs, ModelArgs, Transformer = _official_modules(
        inference_dir
    )
    if args.prompt.count("{image}") != 1:
        raise ValueError("--prompt must contain exactly one {image} placeholder")
    prompt = args.prompt.replace("{image}", image_placeholder)

    with config_path.open() as stream:
        model_args = ModelArgs(**json.load(stream))
    model_args.max_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    input_ids, image_inputs = prepare_vl_inputs(
        prompt,
        [{"url": str(image_path)}],
        tokenizer,
        model_args,
    )
    if image_inputs is None or len(image_inputs) != 1:
        raise ValueError("The reference exporter requires exactly one image")
    model_args.max_seq_len = max(model_args.max_seq_len, len(input_ids) + 1)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    with torch.device(device):
        model = Transformer(model_args)
    checkpoint_file = checkpoint / "model0-mp1.safetensors"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            "Expected a TP1 official checkpoint at " f"{checkpoint_file}"
        )
    load_model(model, str(checkpoint_file))
    model.eval()
    torch.set_default_device(device)

    image_input = image_inputs[0]
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        aligned = model.encode_image(
            image_input.patches.to(device),
            image_input.n_vit_h,
            image_input.n_vit_w,
        )[image_input.perm.to(device)]
        _, logits, _ = model(tokens, start_pos=0, images=[image_inputs])
    if logits.ndim == 3:
        logits = logits[0, -1]
    elif logits.ndim == 2:
        logits = logits[0]
    else:
        raise ValueError(f"Unexpected official logits shape: {tuple(logits.shape)}")

    with Image.open(image_path) as source:
        image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    types = image_input.types.cpu().numpy()
    permutation = image_input.perm.cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fixture_format_version=np.asarray(1, dtype=np.int32),
        reference_revision=np.asarray(args.reference_revision),
        image_rgb=image_rgb,
        prompt=np.asarray(prompt),
        input_ids=np.asarray([input_ids], dtype=np.int64),
        pixel_values=_to_numpy(image_input.patches),
        image_grid_hw=np.asarray(
            [[image_input.n_vit_h, image_input.n_vit_w]], dtype=np.int32
        ),
        image_sample_indices=np.asarray([0], dtype=np.int32),
        image_offsets=np.asarray([image_input.start], dtype=np.int32),
        image_types=types,
        image_type_offsets=np.asarray([0, len(types)], dtype=np.int32),
        image_permutations=permutation,
        aligned_vision_features=_to_numpy(aligned),
        first_token_logits=_to_numpy(logits),
        vision_atol=np.asarray(args.vision_atol, dtype=np.float32),
        vision_rtol=np.asarray(args.vision_rtol, dtype=np.float32),
        logits_atol=np.asarray(args.logits_atol, dtype=np.float32),
        logits_rtol=np.asarray(args.logits_rtol, dtype=np.float32),
    )
    return output


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export official DeepSeek-V4 Flash Vision parity arrays."
    )
    parser.add_argument(
        "--official-inference-dir",
        type=Path,
        required=True,
        help="Path to the checkpoint repository's inference directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the official TP1 converted runtime checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Official runtime config (default: <inference-dir>/config.json).",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="{image}\nDescribe this image.",
        help="Prompt containing exactly one literal {image} placeholder.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=33377335)
    parser.add_argument(
        "--reference-revision",
        default="6821d6ad3681a4b137b066b76094fa82ebd0a380",
    )
    parser.add_argument("--vision-atol", type=float, default=2e-2)
    parser.add_argument("--vision-rtol", type=float, default=2e-2)
    parser.add_argument("--logits-atol", type=float, default=1e-1)
    parser.add_argument("--logits-rtol", type=float, default=2e-2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    output = export_fixture(configure_parser().parse_args())
    print(f"Wrote DeepSeek-V4 Flash Vision reference fixture to {output}")


if __name__ == "__main__":
    main()
