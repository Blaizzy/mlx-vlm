"""Convert released inference checkpoints with MLX and the standard library."""

import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx

from .checkpoint import to_safetensors
from .config import ModelConfig

COMPONENTS = (
    "ss_generator",
    "slat_generator",
    "ss_decoder",
    "slat_decoder_gs",
    "slat_decoder_gs_4",
    "slat_decoder_mesh",
)


def map_tensor(component, name, tensor):
    if component.endswith("generator"):
        if name.startswith("_base_models.generator.reverse_fn.backbone."):
            name = (
                component
                + "."
                + name.removeprefix("_base_models.generator.reverse_fn.backbone.")
            )
        elif name.startswith("_base_models.condition_embedder."):
            name = (
                component.replace("generator", "condition_embedder")
                + "."
                + name.removeprefix("_base_models.condition_embedder.")
            )
        elif name.startswith("_base_models.decoder."):
            # Duplicate occupancy decoder bundled with the structure generator.
            return None
        else:
            raise ValueError(f"Unmapped inference tensor: {name}")
    else:
        name = component + "." + name
    if name.startswith("ss_decoder.") and tensor.ndim == 5:
        tensor = tensor.transpose(0, 2, 3, 4, 1)
    elif name.endswith("patch_embed.proj.weight"):
        tensor = tensor.transpose(0, 2, 3, 1)
    if mx.issubdtype(tensor.dtype, mx.floating):
        tensor = tensor.astype(mx.bfloat16)
    return name, tensor


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    if (source / "checkpoints").is_dir():
        source = source / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    index, total_size, report = {}, 0, {}
    for i, component in enumerate(COMPONENTS, 1):
        checkpoint = source / f"{component}.ckpt"
        safe = source / f"{component}.safetensors"
        with tempfile.TemporaryDirectory(prefix="sam3d-convert-") as scratch:
            if checkpoint.exists():
                safe = Path(scratch) / "source.safetensors"
                count = to_safetensors(checkpoint, safe)
            elif safe.exists():
                count = None
            else:
                raise FileNotFoundError(checkpoint)
            weights = {}
            for name, tensor in mx.load(str(safe)).items():
                mapped = map_tensor(component, name, tensor)
                if mapped is not None:
                    key, value = mapped
                    weights[key] = value
            mx.async_eval(weights)
            shard = f"model-{i:05d}-of-{len(COMPONENTS):05d}.safetensors"
            mx.save_safetensors(
                str(destination / shard),
                weights,
                metadata={"format": "mlx", "dtype": "bfloat16"},
            )
            index.update({k: shard for k in weights})
            total_size += sum(v.nbytes for v in weights.values())
            report[component] = {
                "source_tensors": count,
                "converted_tensors": len(weights),
                "shard": shard,
            }
            print(f"{component}: {len(weights)} tensors -> {shard}", flush=True)
            del weights
            mx.clear_cache()
    ModelConfig().save(destination / "config.json")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": index}, indent=2
        )
        + "\n"
    )
    (destination / "conversion.json").write_text(
        json.dumps(
            {
                "source": "facebook/sam-3d-objects",
                "dtype": "bfloat16",
                "components": report,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def add_depth(source, destination, dtype=None):
    """Bundle the converted MoGe-3 weights in ``source`` as the depth model.

    ``source`` is an MLX MoGe-3 directory (``config.json`` plus safetensors)
    such as a download of ``mlx-community/moge-3-vitl-mlx-fp32``. Tensors keep
    their stored dtype unless ``dtype`` names an MLX floating type.
    """
    from ..moge3.config import ModelConfig as DepthConfig
    from ..moge3.moge3 import Model as DepthModel
    from .sam3d_objects import read_weights

    source, destination = Path(source), Path(destination)
    depth_config = json.loads((source / "config.json").read_text())
    if depth_config.get("model_type") != "moge3":
        raise ValueError("--moge-checkpoint must be a converted MoGe-3 model directory")
    model = DepthModel(DepthConfig.from_dict(depth_config))
    weights = model.sanitize(read_weights(source))
    if dtype is not None:
        cast = getattr(mx, dtype)
        weights = {
            k: v.astype(cast) if mx.issubdtype(v.dtype, mx.floating) else v
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    weights = {"depth_model." + k: v for k, v in weights.items()}
    mx.save_safetensors(
        str(destination / "moge.safetensors"), weights, metadata={"format": "mlx"}
    )
    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {
        k: v for k, v in index["weight_map"].items() if not k.startswith("depth_model.")
    }
    index["weight_map"].update({k: "moge.safetensors" for k in weights})
    index["metadata"]["total_size"] = sum(
        p.stat().st_size for p in destination.glob("*.safetensors")
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    path = destination / "config.json"
    config = json.loads(path.read_text())
    config["depth_model"] = depth_config
    path.write_text(json.dumps(config, indent=2) + "\n")
    dtypes = sorted({str(v.dtype).rsplit(".", 1)[-1] for v in weights.values()})
    report = destination / "conversion.json"
    if report.exists():
        details = json.loads(report.read_text())
        details.setdefault("components", {})["depth_model"] = {
            "backbone": model.config.encoder.backbone,
            "dtypes": dtypes,
            "converted_tensors": len(weights),
            "shard": "moge.safetensors",
        }
        report.write_text(json.dumps(details, indent=2) + "\n")
    print(f"MoGe-3: {len(weights)} tensors added ({', '.join(dtypes)})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--moge-checkpoint",
        help="Optional converted MoGe-3 directory (mlx-community/moge-3-vitl-mlx-fp32)",
    )
    parser.add_argument(
        "--moge-dtype",
        choices=("float32", "bfloat16", "float16"),
        help="Cast the MoGe-3 tensors; by default they keep their stored dtype",
    )
    args = parser.parse_args()
    if args.source:
        convert(args.source, args.output)
    if args.moge_checkpoint:
        add_depth(args.moge_checkpoint, args.output, args.moge_dtype)
    if not args.source and not args.moge_checkpoint:
        parser.error("Provide --source and/or --moge-checkpoint")
    from .bundle import write_card

    write_card(args.output)


if __name__ == "__main__":
    main()
