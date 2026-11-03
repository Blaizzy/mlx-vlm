"""Local full-checkpoint fidelity, outside CI. Requires torch, timm and Apple Silicon.

First publish the converted models, then pass their immutable revisions:
  python examples/nemotron_parse_v3_5_fidelity.py --source SOURCE_SNAPSHOT \
    --output evidence --images page1.png page2.png \
    --model user/model-bf16@COMMIT --model user/model-8bit@COMMIT

The reference dtype/device are explicit; the source file stores float32 weights.
Converted models use their published dtype. The report distinguishes numerical drift, geometry, and text.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
SOURCE_REVISION = "e7ea79242a87ea0ad3919b672bfedb429a683206"


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def structure(text):
    return {
        "classes": re.findall(r"<class_([^>]+)>", text),
        "boxes": re.findall(r"<([xy]_[^>]+)>", text),
        "text": " ".join(re.sub(r"<(?:x_|y_|class_)[^>]+>", "", text).split()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--teacher-tokens", type=int, default=32)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--reference-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument("--reference-device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "fidelity.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {
            "source": {
                "repo_id": "nvidia/Nemotron-3.5-Parse",
                "revision": SOURCE_REVISION,
                "weights_sha256": sha256(args.source / "model.safetensors"),
                "dtype": args.reference_dtype,
                "device": args.reference_device,
            },
            "settings": {
                "prompt": PROMPT,
                "temperature": 0,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "max_new_tokens": args.max_tokens,
                "teacher_tokens": args.teacher_tokens,
                "mtp": False,
            },
            "platform": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "mlx",
                        "mlx-vlm",
                        "torch",
                        "transformers",
                        "timm",
                        "numpy",
                        "Pillow",
                    ]
                },
            },
            "pages": [],
            "models": {},
        }
        import torch
        from transformers import AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        torch.set_num_threads(8)
        reference = (
            AutoModel.from_pretrained(
                args.source,
                trust_remote_code=True,
                dtype=getattr(torch, args.reference_dtype),
            )
            .to(args.reference_device)
            .eval()
        )
        # Load the NVIDIA processor directly, independent of MLX's processor registry.
        processor = get_class_from_dynamic_module(
            "hf_nemotron_parse_processor.NemotronParseProcessor", args.source
        ).from_pretrained(args.source, trust_remote_code=True)
        for index, image_path in enumerate(args.images):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=image, text=PROMPT, add_special_tokens=False, return_tensors="pt"
            )
            source_inputs = inputs
            inputs = {k: v.to(args.reference_device) for k, v in inputs.items()}
            started = time.monotonic()
            with torch.inference_mode():
                ids = reference.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_tokens,
                    repetition_penalty=1.1,
                )[0, inputs["input_ids"].shape[1] :]
                text = processor.decode(ids, skip_special_tokens=True)
                teacher = torch.cat(
                    [inputs["input_ids"], ids[: args.teacher_tokens][None]], dim=1
                )
                teacher_inputs = {
                    **inputs,
                    "input_ids": teacher,
                    "attention_mask": torch.ones_like(teacher),
                }
                logits = (
                    reference(**teacher_inputs, use_cache=False)
                    .logits[0]
                    .float()
                    .cpu()
                    .numpy()
                )
            np.savez(
                args.output / f"page-{index}.npz",
                logits=logits,
                teacher=teacher.cpu().numpy(),
                **{k: v.numpy() for k, v in source_inputs.items()},
            )
            report["pages"].append(
                {
                    "path": str(image_path.resolve()),
                    "sha256": sha256(image_path),
                    "size": list(image.size),
                    "reference": {
                        "text": text,
                        "tokens": ids.tolist(),
                        "seconds": time.monotonic() - started,
                        "ended_with_eos": ids[-1].item() == 2,
                        "reached_token_limit": len(ids) >= args.max_tokens,
                        "structure": structure(text),
                    },
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print("reference", image_path.name, len(ids), "tokens", flush=True)
        del reference
    if args.reference_only:
        return

    import mlx.core as mx
    from huggingface_hub import snapshot_download

    from mlx_vlm import generate, load

    for specification in args.model:
        repo, revision = specification.rsplit("@", 1)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use an immutable published model revision")
        snapshot = Path(snapshot_download(repo, revision=revision))
        model, processor = load(str(snapshot))
        results = []
        for index, page in enumerate(report["pages"]):
            arrays = np.load(args.output / f"page-{index}.npz")
            image = Image.open(page["path"]).convert("RGB")
            inputs = processor(images=image, text=PROMPT)
            for name in ["pixel_values", "image_sizes", "content_sizes", "input_ids"]:
                np.testing.assert_array_equal(np.asarray(inputs[name]), arrays[name])
            kwargs = {name: mx.array(value) for name, value in inputs.items()}
            kwargs["input_ids"] = mx.array(arrays["teacher"])
            logits = model(**kwargs).logits[0].astype(mx.float32)
            mx.eval(logits)
            actual, expected = np.asarray(logits), arrays["logits"]
            difference = actual - expected
            started = time.monotonic()
            generated = generate(
                model,
                processor,
                PROMPT,
                [image],
                temperature=0,
                repetition_penalty=1.1,
                repetition_context_size=0,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            parsed, target = structure(generated.text), page["reference"]["structure"]
            metrics = {
                "preprocessing_exact": True,
                "logit_rmse": float(np.sqrt(np.mean(difference**2))),
                "logit_max_abs": float(np.max(np.abs(difference))),
                "teacher_argmax_agreement": float(
                    np.mean(actual.argmax(-1) == expected.argmax(-1))
                ),
                "output_exact": generated.text == page["reference"]["text"],
                "text_exact_ignoring_geometry": parsed["text"] == target["text"],
                "text_similarity_ignoring_geometry": SequenceMatcher(
                    None, target["text"], parsed["text"], autojunk=False
                ).ratio(),
                "classes_exact": parsed["classes"] == target["classes"],
                "boxes_exact": parsed["boxes"] == target["boxes"],
                "text": generated.text,
                "tokens": generated.token_ids,
                "structure": parsed,
                "seconds": time.monotonic() - started,
                "finish_reason": generated.finish_reason,
            }
            results.append(metrics)
            print(
                repo,
                Path(page["path"]).name,
                {
                    k: v
                    for k, v in metrics.items()
                    if k not in ["text", "tokens", "structure"]
                },
                flush=True,
            )
        report["models"][repo] = {
            "revision": revision,
            "config": json.loads((snapshot / "config.json").read_text()),
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        del model, processor
        mx.clear_cache()


if __name__ == "__main__":
    main()
