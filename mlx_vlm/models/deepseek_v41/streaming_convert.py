"""Streaming fp8-source converter for DeepSeek-V4.1-Flash quants.

Reads the native release shard by shard (never staging the 763 GB), dequantizes
each tensor bit-exactly, requantizes per profile, and writes fixed-size output
shards with an index. Resume-safe via a state file; upload separately with
``hf upload``. Runs on the conversion host (512 GB Studio).

Profiles (``--profile``):

* ``4bit`` -- experts 4-bit affine g64, attention/shared 8-bit, engram 4-bit.
* ``nvfp4`` -- experts NVFP4 g16 (native-adjacent), rest like ``4bit``.
* ``engram6`` -- like ``4bit`` but engram tables at 6-bit (near-fp8 quality).

Routing: FP4-packed routed experts and FP8 matmuls dequantize first; norms,
sinks, biases, hyper-connections, gates, vision, and span vectors pass through;
head/embed stay bf16. Giant engram tables process in row chunks.
"""

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

from .config import ModelConfig
from .dequant import dequant_fp4, dequant_fp8, dequant_fp8_rows, is_fp4_expert

SOURCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
SHARD_SIZE_BYTES = 5 * 1024**3
ROW_CHUNK = 4_000_000

PROFILES = {
    "4bit": {
        "expert": (4, "affine", 64),
        "dense": (8, "affine", 64),
        "engram": (4, "affine", 64),
    },
    "nvfp4": {
        "expert": (4, "nvfp4", 16),
        "dense": (8, "affine", 64),
        "engram": (4, "affine", 64),
    },
    "engram6": {
        "expert": (4, "affine", 64),
        "dense": (8, "affine", 64),
        "engram": (6, "affine", 64),
    },
}


def quantize_like(w: mx.array, bits: int, mode: str, group: int):
    """mx.quantize returning (weight, scales) with optional biases."""
    result = mx.quantize(w, group_size=group, bits=bits, mode=mode)
    if len(result) == 3:
        return result[0], result[1], result[2]
    weight, scales = result
    return weight, scales, None


def _emit_quantized(name: str, w32: mx.array, bits: int, mode: str, group: int):
    weight, scales, biases = quantize_like(w32, bits, mode, group)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    out = {name: weight, f"{stem}.scales": scales}
    if biases is not None:
        out[f"{stem}.biases"] = biases
    return out


def convert_tensor(name: str, weight: mx.array, scale, profile: dict):
    """Dequantize one source tensor and requantize per profile."""
    if "engram.embed" in name:
        bits, mode, group = profile["engram"]
        w32 = dequant_fp8_rows(weight, scale).astype(mx.float32)
        return _emit_quantized(name, w32, bits, mode, group)
    if is_fp4_expert(name):
        bits, mode, group = profile["expert"]
        w32 = dequant_fp4(weight, scale).astype(mx.float32)
        return _emit_quantized(name, w32, bits, mode, group)
    bits, mode, group = profile["dense"]
    w32 = dequant_fp8(weight, scale).astype(mx.float32)
    return _emit_quantized(name, w32, bits, mode, group)


def passthrough_fp32(name: str, tensor: mx.array):
    dtype = mx.float32 if tensor.dtype == mx.float32 else mx.bfloat16
    return {name: tensor.astype(dtype)}


def convert_shard(source_file: Path, profile: dict):
    """Convert one source shard, one tensor at a time, row-chunking giants."""
    out = {}
    with safe_open(str(source_file), framework="mlx") as f:
        keys = list(f.keys())
        store = None

        def load(key):
            nonlocal store
            try:
                return mx.array(f.get_tensor(key))
            except (AttributeError, RuntimeError, TypeError):
                if store is None:
                    store = dict(mx.load(str(source_file)))
                return store.get(key)

        for key in keys:
            if key.endswith(".scale"):
                continue
            tensor = load(key)
            if tensor is None:
                continue
            if "engram.embed" in key and tensor.shape[0] > ROW_CHUNK:
                scale_all = load(f"{key[:-len('.weight')]}.scale")
                merged = {}
                for start in range(0, tensor.shape[0], ROW_CHUNK):
                    end = min(start + ROW_CHUNK, tensor.shape[0])
                    part = convert_tensor(
                        key, tensor[start:end], scale_all[start:end], profile
                    )
                    for k, v in part.items():
                        merged.setdefault(k, []).append(v)
                    mx.clear_cache()
                for k, v in merged.items():
                    out[k] = mx.concatenate(v, axis=0) if len(v) > 1 else v[0]
                del merged, scale_all
            elif (
                f"{key[:-len('.weight')]}.scale" in keys
                if key.endswith(".weight")
                else False
            ):
                scale = load(f"{key[:-len('.weight')]}.scale")
                out.update(convert_tensor(key, tensor, scale, profile))
                del scale
            elif tensor.dtype in (mx.bfloat16, mx.float32):
                out.update(passthrough_fp32(key, tensor))
            else:
                raise ValueError(f"Unroutable tensor {key} dtype {tensor.dtype}.")
            del tensor
            mx.clear_cache()
        if store is not None:
            store.clear()
    gc.collect()
    return out


def output_config() -> dict:
    """Our load config."""
    config = ModelConfig().to_dict()
    config["model_type"] = "deepseek_v41"
    return config


def quantization_recipe(profile_name: str) -> dict:
    """Global recipe marker; per-module fidelity follows the profile table."""
    expert_bits, expert_mode, expert_group = PROFILES[profile_name]["expert"]
    return {"group_size": expert_group, "bits": expert_bits, "mode": expert_mode}


def already_done(state_path: Path):
    if state_path.exists():
        return set(json.loads(state_path.read_text()))
    return set()


def mark_done(state_path: Path, done: set):
    state_path.write_text(json.dumps(sorted(done)))


def stream_convert(
    profile_name: str,
    output: str,
    *,
    token: str | None = None,
    revision: str | None = None,
    max_shards: int | None = None,
    workdir: str | None = None,
) -> Path:
    """Download, convert, and write output shards one source shard at a time."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / ".convert-state.json"
    done = already_done(state_path)
    work = Path(workdir or f"{output}.work")
    work.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in list_repo_files(SOURCE_REPO, revision=revision, token=token)
        if f.startswith("model-") and f.endswith(".safetensors")
    ]
    files = sorted(files)
    if max_shards is not None:
        files = files[:max_shards]

    pending: dict[str, mx.array] = {}
    pending_bytes = 0
    output_index = 0
    weight_map: dict[str, str] = {}
    total_size = 0

    def flush():
        nonlocal output_index, total_size, pending, pending_bytes
        if not pending:
            return
        output_index += 1
        filename = f"model-{output_index:05d}-of-00048.safetensors"
        mx.save_safetensors(
            str(output_path / filename), pending, metadata={"format": "mlx"}
        )
        for key, value in pending.items():
            weight_map[key] = filename
            total_size += value.nbytes
        pending = {}
        pending_bytes = 0
        gc.collect()
        mx.clear_cache()

    for filename in files:
        if filename in done:
            continue
        local = hf_hub_download(
            SOURCE_REPO, filename, revision=revision, token=token, local_dir=str(work)
        )
        converted = convert_shard(Path(local), PROFILES[profile_name])
        Path(local).unlink()
        for key, value in converted.items():
            pending[key] = value
            pending_bytes += value.nbytes
            if pending_bytes >= SHARD_SIZE_BYTES:
                flush()
        del converted
        gc.collect()
        done.add(filename)
        mark_done(state_path, done)
    flush()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    config = output_config()
    config["quantization"] = quantization_recipe(profile_name)
    config["quantization_config"] = quantization_recipe(profile_name)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(
            SOURCE_REPO,
            name,
            revision=revision,
            token=token,
            local_dir=str(output_path),
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream DeepSeek-V4.1 fp8 source into an MLX quant."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--workdir", default=None)
    return parser


def main():
    args = vars(build_parser().parse_args())
    args["profile_name"] = args.pop("profile")
    output = stream_convert(**args)
    print(f"Wrote build to {output}")


if __name__ == "__main__":
    main()
