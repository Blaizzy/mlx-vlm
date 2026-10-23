"""Convert Hi3D v1.0 and its DINOv2 / CLIP encoders into one MLX bundle.

Tensors keep their released dtypes (float16 transformer layers, float32
norms, modulation, decoder and encoders). Only key names change: the
encoders are remapped to the ``dinov2`` and ``llava`` vision layouts and the
CLIP text tower is dropped.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import DecoderConfig, ModelConfig, TransformerConfig

HI3D_REPO = "hi3d-official/Hi3Dv1.0"
DINO_REPO = "facebook/dinov2-large"
CLIP_REPO = "openai/clip-vit-large-patch14"
SHARD_BYTES = 5 << 30
# The Hi3D release ships the Apache-2.0 text with its FlexiCubes sources.
APACHE = "sparc3d/ops/flexicubes/LICENSE.txt"
CLIP_LICENSE = """MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DINO_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "mlp_ratio",
    "layer_norm_eps",
    "image_size",
    "patch_size",
    "layerscale_value",
)
CLIP_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "image_size",
    "patch_size",
    "projection_dim",
    "layer_norm_eps",
)

MODIFICATIONS = """
MLX conversion (mlx-vlm ``hi3d``)
---------------------------------
This bundle is an unofficial MLX conversion of hi3d-official/Hi3Dv1.0
(revision {hi3d_revision}). The Hi3D weights in model-*.safetensors are
unmodified in value and dtype; their tensor names carry the prefixes
``transformer_512.``, ``transformer_1024.`` and ``vae.``, and the constant
``inv_grid_coord`` buffers were dropped. The bundle also contains, under
``conditioner.``, the vision towers of facebook/dinov2-large (revision
{dino_revision}, Apache-2.0, see LICENSE-APACHE) and
openai/clip-vit-large-patch14 (revision {clip_revision}, MIT, see
LICENSE-CLIP), renamed to the mlx-vlm layouts with their patch-embedding
convolutions transposed to channels-last. conversion.json lists every source
file and transform.
"""


def _resolve(source, revision=None):
    """A local directory, or a Hub snapshot of ``source`` at ``revision``."""
    path = Path(source)
    if path.is_dir():
        return path, None
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            source,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "LICENSE",
                "NOTICE",
                "*.md",
                APACHE,
            ],
        )
    )
    return root, root.name


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transformer_config(path):
    values = json.loads(path.read_text())
    values.pop("use_fp16", None)
    return TransformerConfig.from_dict(values)


def _source(name, root, revision, files):
    return {
        "repository": name,
        "revision": revision,
        "files": {f: _sha256(root / f) for f in files},
    }


def convert(output, hi3d=HI3D_REPO, dino=DINO_REPO, clip=CLIP_REPO, revision=None):
    from .hi3d import Model

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hi3d_root, hi3d_revision = _resolve(hi3d, revision)
    dino_root, dino_revision = _resolve(dino)
    clip_root, clip_revision = _resolve(clip)

    scheduler = json.loads((hi3d_root / "scheduler/scheduler_config.json").read_text())
    vae = json.loads((hi3d_root / "vae/config.json").read_text())["model"]
    decoder = dict(vae["decoder_cfg"])
    decoder.pop("type")
    dino_config = json.loads((dino_root / "config.json").read_text())
    clip_config = json.loads((clip_root / "config.json").read_text())["vision_config"]
    config = ModelConfig(
        transformer_512=_transformer_config(hi3d_root / "transformer_512/config.json"),
        transformer_1024=_transformer_config(
            hi3d_root / "transformer_1024/config.json"
        ),
        decoder=DecoderConfig.from_dict(decoder),
        dino_config={"model_type": "dinov2", **{k: dino_config[k] for k in DINO_KEYS}},
        clip_config={
            "model_type": "clip_vision_model",
            **{k: clip_config[k] for k in CLIP_KEYS},
        },
        num_inference_steps=scheduler["num_inference_steps"],
        sigma_min=scheduler["sigma_min"],
        time_rescale_factor=scheduler["time_rescale_factor"],
    )
    model = Model(config)

    weights, skipped = {}, []
    for part in ("transformer_512", "transformer_1024", "vae"):
        for key, value in mx.load(str(hi3d_root / part / "model.safetensors")).items():
            if key.endswith("inv_grid_coord"):
                skipped.append(f"{part}.{key}")
                continue
            weights[f"{part}.{key}"] = value
    dino_weights = model.conditioner.dino.sanitize(
        mx.load(str(dino_root / "model.safetensors"))
    )
    weights.update({f"conditioner.dino.{k}": v for k, v in dino_weights.items()})
    clip_weights = {}
    for key, value in mx.load(str(clip_root / "model.safetensors")).items():
        if key.startswith("vision_model."):
            clip_weights[key] = value
        else:
            skipped.append(f"clip:{key}")
    clip_weights = model.conditioner.clip.sanitize(clip_weights)
    weights.update({f"conditioner.clip.{k}": v for k, v in clip_weights.items()})

    expected = dict(tree_flatten(model.parameters()))
    if set(expected) != set(weights):
        missing = sorted(set(expected) - set(weights))[:5]
        extra = sorted(set(weights) - set(expected))[:5]
        raise ValueError(f"Unmapped tensors: missing {missing}, extra {extra}")
    for key, value in weights.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"{key}: {value.shape} != {expected[key].shape}")

    shards, current, size = [], {}, 0
    for key in sorted(weights):
        nbytes = weights[key].nbytes
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = weights[key]
        size += nbytes
    shards.append(current)
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(output / name), shard, metadata={"format": "mlx"})
        weight_map.update({key: name for key in shard})
    total = sum(v.nbytes for v in weights.values())
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
        + "\n"
    )
    config.save(output / "config.json")

    dtypes = {}
    for key, value in weights.items():
        part = (
            key.split(".")[0]
            if not key.startswith("conditioner.")
            else key.split(".")[1]
        )
        dtypes.setdefault(part, {}).setdefault(str(value.dtype).split(".")[-1], 0)
        dtypes[part][str(value.dtype).split(".")[-1]] += 1
    hi3d_files = [
        f"{part}/{name}"
        for part in ("transformer_512", "transformer_1024", "vae")
        for name in ("config.json", "model.safetensors")
    ] + ["scheduler/scheduler_config.json"]
    report = {
        "sources": [
            _source(HI3D_REPO, hi3d_root, hi3d_revision, hi3d_files),
            _source(
                DINO_REPO,
                dino_root,
                dino_revision,
                ["config.json", "model.safetensors"],
            ),
            _source(
                CLIP_REPO,
                clip_root,
                clip_revision,
                ["config.json", "model.safetensors"],
            ),
        ],
        "tensors": len(weights),
        "bytes": total,
        "dtypes": dtypes,
        "skipped": skipped,
        "transforms": {
            "conditioner.dino": "Hugging Face Dinov2Model keys to mlx-vlm dinov2 layout; "
            "q/k/v merged into qkv; patch conv (O, I, H, W) -> (O, H, W, I)",
            "conditioner.clip": "vision_model.* only; position_ids dropped; patch conv "
            "(O, I, H, W) -> (O, H, W, I)",
            "transformer_512, transformer_1024, vae": "values and dtypes unchanged; "
            "spconv weights (O, kx, ky, kz, I) used as-is",
        },
    }
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")

    shutil.copyfile(hi3d_root / "LICENSE", output / "LICENSE")
    shutil.copyfile(
        hi3d_root / "THIRD_PARTY_LICENSES.md", output / "THIRD_PARTY_LICENSES.md"
    )
    shutil.copyfile(hi3d_root / APACHE, output / "LICENSE-APACHE")
    (output / "LICENSE-CLIP").write_text(CLIP_LICENSE)
    notice = (hi3d_root / "NOTICE").read_text()
    notice += MODIFICATIONS.format(
        hi3d_revision=hi3d_revision or "local",
        dino_revision=dino_revision or "local",
        clip_revision=clip_revision or "local",
    )
    (output / "NOTICE").write_text(notice)

    loaded = Model.from_pretrained(output)
    for key, value in tree_flatten(loaded.parameters()):
        if value.dtype != weights[key].dtype:
            raise ValueError(f"{key} reloaded as {value.dtype}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Bundle directory to write")
    parser.add_argument("--hi3d", default=HI3D_REPO, help="Hi3D repo id or directory")
    parser.add_argument("--revision", default=None, help="Hi3D revision")
    parser.add_argument(
        "--dino", default=DINO_REPO, help="DINOv2-L repo id or directory"
    )
    parser.add_argument(
        "--clip", default=CLIP_REPO, help="CLIP ViT-L/14 repo id or directory"
    )
    args = parser.parse_args()
    convert(args.output, args.hi3d, args.dino, args.clip, args.revision)


if __name__ == "__main__":
    main()
