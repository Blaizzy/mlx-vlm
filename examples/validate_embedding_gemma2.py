"""Run all 20 pinned upstream examples, comparing every forward pass with MLX.

Requires the Transformers wheel in gg-hf-em/embeddinggemma-2-eap-extras (dataset),
sentence-transformers >= 6.1, torch, torchaudio, torchcodec, and FFmpeg. See the
EmbeddingGemma 2 model README for setup and numerical acceptance thresholds.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import time
import traceback
from pathlib import Path
from unittest.mock import patch

# M5 defaults to TF32 for float32 matmul; compare full-precision implementations.
# This must precede importing MLX.
os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download, snapshot_download
from mlx.utils import tree_map
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EmbeddingGemma2Model,
    EmbeddingGemma2Processor,
)

from mlx_vlm.embedding_loader import load_embedding_model

MODEL_REVISION = "fc77679a26fcb86250765859d04ce2fcc6cb0b2c"
EXTRAS_REVISION = "f6c512df20896fd06f85d39db10c45a0a9849ef8"
DOCUMENT_SHA256 = "2f5fbff7c01d7542147da398f23e379847f8dfd0617c6a9cda2f244caff9a588"


class AdaptExample(ast.NodeTransformer):
    """Only replace model/media locations and select the CPU reference device."""

    def __init__(self, replacements):
        self.replacements = replacements

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.replacements:
            node.value = self.replacements[node.value]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "SentenceTransformer":
            node.keywords.append(ast.keyword(arg="device", value=ast.Constant("cpu")))
        return node


def to_mlx(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            return mx.array(value.float().numpy()).astype(mx.bfloat16)
        return mx.array(value.numpy())
    return value


def unit_vectors(value):
    value = value.astype(np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-9)


class Validator:
    def __init__(self, model_path, dtype, mlx_model_path=None, tolerances=None):
        self.model_path = model_path
        self.mlx_model_path = mlx_model_path or model_path
        self.dtype = dtype
        self.cast_weights = mlx_model_path is None
        self.tolerances = tolerances or {}
        self.load = AutoModel.from_pretrained
        self.forward = EmbeddingGemma2Model.forward
        self.process = EmbeddingGemma2Processor.__call__
        self.signature = inspect.signature(self.forward)
        self.references = {}
        self.models = {}
        self.checks = []
        self.accuracy_failures = []
        self.prepared = []
        self.example = 0

    def load_reference(self, path, *args, **kwargs):
        config = kwargs.get("config") or AutoConfig.from_pretrained(
            path,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("audio_config", "vision_config")
            },
        )
        key = (config.vision_config is not None, config.audio_config is not None)
        if key not in self.references:
            kwargs.pop("device_map", None)
            kwargs.pop("torch_dtype", None)
            kwargs.update(
                config=config, dtype=torch.float32, attn_implementation="sdpa"
            )
            self.references[key] = self.load(path, *args, **kwargs).cpu().eval()
        return self.references[key]

    def compare(self, reference, inputs, output):
        key = (
            reference.config.vision_config is not None,
            reference.config.audio_config is not None,
        )
        if key not in self.models:
            config = json.loads((self.mlx_model_path / "config.json").read_text())
            if not key[0]:
                config["vision_config"] = None
            if not key[1]:
                config["audio_config"] = None
            model = load_embedding_model(self.mlx_model_path, config=config)
            if self.cast_weights:
                model.update(
                    tree_map(
                        lambda x: x.astype(getattr(mx, self.dtype)), model.parameters()
                    )
                )
            self.models[key] = model
        actual = self.models[key](
            **{k: to_mlx(v) for k, v in inputs.items() if v is not None}
        )
        tokens = np.array(actual.last_hidden_state.astype(mx.float32))
        embeddings = np.array(actual.text_embeds)
        expected = output.last_hidden_state.detach().float().cpu().numpy()
        mask = (
            inputs.get("attention_mask", torch.ones(expected.shape[:2])).cpu().numpy()
        )
        pooled = (expected * mask[..., None]).sum(1) / np.maximum(
            mask.sum(1, keepdims=True), 1e-9
        )
        pooled = unit_vectors(pooled)
        difference = tokens[mask.astype(bool)] - expected[mask.astype(bool)]
        check = {
            "example": self.example,
            "shape": list(expected.shape),
            "modalities": [
                k
                for k in ("pixel_values", "input_features", "pixel_values_videos")
                if inputs.get(k) is not None
            ],
            "embedding_max_abs": float(np.abs(embeddings - pooled).max()),
            "embedding_min_cosine": float(
                (unit_vectors(embeddings) * pooled).sum(-1).min()
            ),
            "token_max_abs": float(np.abs(difference).max()),
            "token_relative_l2": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(expected[mask.astype(bool)]), 1e-9)
            ),
        }
        # Test every documented Matryoshka dimension on every modality/batch.
        check["prefix_min_cosine"] = min(
            float(
                (unit_vectors(embeddings[..., :dim]) * unit_vectors(pooled[..., :dim]))
                .sum(-1)
                .min()
            )
            for dim in (128, 256, 512)
        )
        self.checks.append(check)
        assert np.isfinite(tokens).all() and np.isfinite(embeddings).all(), check
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=-1), 1, atol=2e-6)
        full_precision = self.dtype == "float32"
        min_cosine = self.tolerances.get(
            "min_cosine", 0.999999 if full_precision else 0.999
        )
        max_error = self.tolerances.get("max_error", 1e-5 if full_precision else 0.01)
        max_token_error = self.tolerances.get(
            "max_token_error", 1e-4 if full_precision else 0.2
        )
        # Unnormalized token magnitudes are more sensitive to BF16 rounding than
        # the normalized sentence vectors; retain and bound both measurements.
        failures = [
            metric
            for metric, passed in (
                ("embedding_max_abs", check["embedding_max_abs"] <= max_error),
                ("embedding_min_cosine", check["embedding_min_cosine"] >= min_cosine),
                ("prefix_min_cosine", check["prefix_min_cosine"] >= min_cosine),
                ("token_relative_l2", check["token_relative_l2"] <= max_token_error),
            )
            if not passed
        ]
        check["failed_metrics"] = failures
        if failures:
            self.accuracy_failures.append(
                {"example": self.example, "metrics": failures}
            )
        print("CHECK", json.dumps(check), flush=True)

    def checked_forward(self, model, *args, **kwargs):
        bound = self.signature.bind(model, *args, **kwargs)
        inputs = {
            k: v for k, v in bound.arguments.items() if k not in ("self", "kwargs")
        }
        inputs.update(bound.arguments.get("kwargs", {}))
        with torch.no_grad():
            result = self.forward(model, *args, **kwargs)
        self.compare(model, inputs, result)
        return result

    def tracked_process(self, processor, *args, **kwargs):
        result = self.process(processor, *args, **kwargs)
        self.prepared.append(result)
        return result

    def check_preprocessing_examples(self):
        if self.example in (17, 19, 20):
            reference = self.load_reference(str(self.model_path))
            for batch in self.prepared:
                reference(**batch)
        if self.example in (14, 15):
            inputs = AutoTokenizer.from_pretrained(self.model_path)(
                ["test input"], return_tensors="pt"
            )
            for reference in list(self.references.values()):
                if reference.config.audio_config is None:
                    reference(**inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Existing pinned checkpoint; otherwise download it",
    )
    parser.add_argument(
        "--extras-path",
        type=Path,
        help="Existing extras snapshot; otherwise download the assets",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--mlx-model-path",
        type=Path,
        help="Validate a converted/quantized checkpoint without casting its stored weights",
    )
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--max-token-error", type=float)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("embeddinggemma2-validation")
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    model_path = args.model_path or Path(
        snapshot_download(
            "gg-hf-em/embeddinggemma-2",
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.md"],
        )
    )
    extras = args.extras_path or Path(
        snapshot_download(
            "gg-hf-em/embeddinggemma-2-eap-extras",
            repo_type="dataset",
            revision=EXTRAS_REVISION,
            allow_patterns=["assets/*"],
        )
    )
    source = (model_path / "embedding_gemma2_documentation.md").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != DOCUMENT_SHA256:
        raise ValueError(
            "Documentation changed: review it before executing its examples"
        )
    blocks = re.findall(r"```python\n(.*?)```", source, re.S)
    assert len(blocks) == 20
    replacements = {
        "google/embeddinggemma-2": str(model_path),
        "path/to/audio.wav": str(extras / "assets/speech.wav"),
        "path/to/video.mp4": str(extras / "assets/sample_video.mp4"),
    }
    for filename in ("pipeline-cat-chonk.jpeg", "coco_sample.png"):
        replacements[
            f"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/{filename}"
        ] = hf_hub_download(
            "huggingface/documentation-images",
            filename,
            repo_type="dataset",
        )
    tolerances = {
        key: value
        for key, value in {
            "min_cosine": args.min_cosine,
            "max_error": args.max_error,
            "max_token_error": args.max_token_error,
        }.items()
        if value is not None
    }
    validator = Validator(model_path, args.dtype, args.mlx_model_path, tolerances)
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_revision": MODEL_REVISION,
        "extras_revision": EXTRAS_REVISION,
        "document_sha256": DOCUMENT_SHA256,
        "dtype": args.dtype,
        "reference_dtype": "float32",
        "mlx_model_path": str(validator.mlx_model_path),
        "quantization": json.loads(
            (validator.mlx_model_path / "config.json").read_text()
        ).get("quantization"),
        "tolerance_overrides": tolerances,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "mlx": mx.__version__,
        "examples": results,
        "checks": validator.checks,
    }

    # Plain functions retain descriptor binding to the reference model/processor.
    def forward(model, *a, **kw):
        return validator.checked_forward(model, *a, **kw)

    def process(processor, *a, **kw):
        return validator.tracked_process(processor, *a, **kw)

    with (
        patch.object(
            AutoModel, "from_pretrained", side_effect=validator.load_reference
        ),
        patch.object(EmbeddingGemma2Model, "forward", forward),
        patch.object(EmbeddingGemma2Processor, "__call__", process),
    ):
        for index, block in enumerate(blocks, 1):
            validator.example = index
            validator.prepared.clear()
            start, before = time.monotonic(), len(validator.checks)
            before_failures = len(validator.accuracy_failures)
            print("EXAMPLE", index, flush=True)
            result = {"example": index, "status": "passed"}
            try:
                tree = ast.fix_missing_locations(
                    AdaptExample(replacements).visit(ast.parse(block))
                )
                exec(compile(tree, f"upstream-example-{index}", "exec"), {})
                validator.check_preprocessing_examples()
                assert len(validator.checks) > before, "No paired forward was checked"
                failures = validator.accuracy_failures[before_failures:]
                assert not failures, f"Accuracy thresholds failed: {failures}"
            except Exception as error:
                traceback.print_exc()
                result.update(status="failed", error=str(error))
            result.update(
                forwards=len(validator.checks) - before,
                seconds=time.monotonic() - start,
            )
            results.append(result)
            (args.output_dir / f"docs-{args.dtype}.json").write_text(
                json.dumps(report, indent=2)
            )
            print("RESULT", json.dumps(result), flush=True)
    failed = sum(result["status"] != "passed" for result in results)
    print(
        f"SUMMARY: {len(results) - failed} passed; {failed} failed; {len(validator.checks)} comparisons"
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
