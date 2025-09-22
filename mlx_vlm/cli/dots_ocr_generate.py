import argparse
import json
import os
import sys

import numpy as np
import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX
from mlx_vlm.text.mlx_qwen_loader import MLXQwen, QwenLoadOpts, pick_vision_token_id


def mx_to_numpy(arr):
    return np.array(mx.array(arr, dtype=mx.float32))

try:  # Prefer shared helper when available.
    from mlx_vlm.utils.pdf_io import pdf_to_images
except ModuleNotFoundError:  # pragma: no cover - fallback for single-file utils module.
    def pdf_to_images(path: str, dpi: int = 200):
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise RuntimeError(
                "pdf2image is required to convert PDFs; install via `pip install pdf2image`"
            ) from exc

        pages = convert_from_path(path, dpi=dpi)
        return [page.convert("RGB") for page in pages]


def _resolve_text_model_dir(model_dir: str) -> str:
    """Locate the underlying Qwen text checkpoint for dots.ocr configs."""

    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return model_dir

    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return model_dir

    if config.get("model_type") != "dots_ocr":
        return model_dir

    text_ref = config.get("text_config_ref")
    if not text_ref:
        env_ref = os.environ.get("DOTS_QWEN_DIR")
        if env_ref:
            return env_ref
        raise ValueError(
            "dots.ocr config.json is missing `text_config_ref`; set it or export DOTS_QWEN_DIR"
        )

    if not os.path.isabs(text_ref):
        candidate = os.path.join(model_dir, text_ref)
        if os.path.exists(candidate):
            return candidate

    return text_ref


def main():
    ap = argparse.ArgumentParser("mlx-vlm.dots-ocr.generate (MLX-only)")
    ap.add_argument("--model-dir", required=True, help="Local dots.ocr HF folder (Qwen-compatible)")
    ap.add_argument("--vision-npz", default="weights/dots_ocr_vision.npz")
    ap.add_argument("--proj-npz", default="weights/dots_ocr_projector.npz")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--prompt", default="Extract the text in this document: <image>")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reduce", default="mean", choices=["mean"])
    args = ap.parse_args()

    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)
    adapter.load_vision_npz(args.vision_npz)
    projector_loaded = False
    try:
        adapter.load_projector_npz(args.proj_npz)
        projector_loaded = True
    except FileNotFoundError:
        print(
            f"[projector] file not found: {args.proj_npz}; continuing without projector",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[projector] failed to load {args.proj_npz}: {exc}; continuing without projector",
            file=sys.stderr,
        )

    pages = pdf_to_images(args.pdf, dpi=args.dpi)
    if not pages:
        raise ValueError("no pages extracted from PDF")
    page_index = min(max(args.page, 0), len(pages) - 1)
    page = pages[page_index]
    text_model_dir = _resolve_text_model_dir(args.model_dir)

    qwen = MLXQwen(
        QwenLoadOpts(
            model_dir=text_model_dir,
            max_new_tokens=args.max_new,
            temperature=0.0,
        )
    )
    tok_str, tok_id = pick_vision_token_id(qwen.tokenizer)
    if tok_id is None:
        tok_str, tok_id = "<image>", 151652

    base_prompt = (
        args.prompt.replace(tok_str, " ").replace("<image>", " ").strip()
    )
    prompt = (base_prompt + f" {tok_str}").strip()
    guard = " Do not include the literal token <image> in your answer. Output only the extracted text."
    prompt = prompt + guard

    vt, _ = adapter.encode_images([page])
    vproj = adapter.projector(vt) if projector_loaded and hasattr(adapter, "projector") else vt
    vproj_np = mx_to_numpy(vproj)

    embed_dim = int(qwen.embed_weight().shape[1])
    if vproj_np.shape[1] != embed_dim:
        original_shape = vproj_np.shape
        if vproj_np.shape[1] < embed_dim:
            pad_width = embed_dim - vproj_np.shape[1]
            vproj_np = np.pad(vproj_np, ((0, 0), (0, pad_width)), mode="constant")
        else:
            vproj_np = vproj_np[:, :embed_dim]

        reason = "missing projector" if not projector_loaded else "projector mismatch"
        print(
            f"[projector] ({reason}) vision features reshaped from {original_shape} -> {vproj_np.shape};"
            " generation quality may degrade. Consider supplying a projector npz that matches the text model.",
            file=sys.stderr,
        )

    out = qwen.generate_with_image_embedding(
        prompt,
        image_id=tok_id,
        vision_projected_seq=vproj_np,
        reduce=args.reduce,
        max_new_tokens=args.max_new,
        temperature=0.0,
        blend=0.65,
    )
    for t in [tok_str, "<image>", "<|vision_start|>", "<|vision_end|>"]:
        out = out.replace(t, "")
    print(out.strip())


if __name__ == "__main__":
    main()
