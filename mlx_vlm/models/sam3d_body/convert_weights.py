"""Convert SAM 3D Body PyTorch checkpoint + JIT MHR model to MLX safetensors.

Usage:
    python -m mlx_vlm.models.sam3d_body.convert_weights \
        --checkpoint /tmp/sam3d-weights/model.ckpt \
        --mhr-model /tmp/sam3d-weights/assets/mhr_model.pt \
        --output /tmp/sam3d-mlx-weights/

Notes:
    The QKV split, backbone prefix remaps, Conv2d layout conversion, and MHR
    JIT prefix remaps performed below are kept here for backward compatibility
    with existing `/tmp/sam3d-mlx-weights/` layouts. The canonical key naming
    lives in `Model.sanitize()` (see model.py), which is invoked automatically
    by `mlx_vlm.utils.load()`. Either path produces the same load result.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from .config import SAM3DConfig

# Max shard size in bytes (5 GB)
MAX_SHARD_SIZE = 5 * 1024**3

# bfloat16 -> float16 for safetensors (safetensors doesn't support bfloat16 natively)
DTYPE_MAP = {
    torch.bfloat16: np.float16,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int32: np.int32,
    torch.int64: np.int32,  # downcast int64 to int32
    torch.bool: np.bool_,
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy with appropriate dtype."""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(np.float16)
    if tensor.dtype == torch.int64:
        return tensor.numpy().astype(np.int32)
    if tensor.dtype == torch.bool:
        # safetensors doesn't support bool; store as uint8
        return tensor.numpy().astype(np.uint8)
    return tensor.numpy()


def transpose_conv2d(weight: np.ndarray) -> np.ndarray:
    """PyTorch Conv2d (O,I,H,W) -> MLX (O,H,W,I)."""
    return np.transpose(weight, (0, 2, 3, 1))


def transpose_conv_transpose2d(weight: np.ndarray) -> np.ndarray:
    """PyTorch ConvTranspose2d (I,O,H,W) -> MLX (O,H,W,I)."""
    return np.transpose(weight, (1, 2, 3, 0))


# --------------------------------------------------------------------------
# Backbone QKV splitting
# --------------------------------------------------------------------------

QKV_PATTERN = re.compile(
    r"backbone\.encoder\.blocks\.(\d+)\.attn\.qkv\.(weight|bias|bias_mask)"
)


def split_qkv_weights(state_dict: dict) -> dict:
    """Find fused QKV weights in backbone and split them into q/k/v."""
    new_weights = {}
    consumed = set()

    for key in list(state_dict.keys()):
        m = QKV_PATTERN.match(key)
        if not m:
            continue
        block_idx = m.group(1)
        param_type = m.group(2)
        tensor = state_dict[key]
        D = tensor.shape[0] // 3
        q, k, v = tensor[:D], tensor[D : 2 * D], tensor[2 * D :]

        prefix = f"backbone.blocks.{block_idx}.attention"
        if param_type == "bias_mask":
            new_weights[f"{prefix}.q_bias_mask"] = q
            new_weights[f"{prefix}.k_bias_mask"] = k
            new_weights[f"{prefix}.v_bias_mask"] = v
        else:
            new_weights[f"{prefix}.q_proj.{param_type}"] = q
            new_weights[f"{prefix}.k_proj.{param_type}"] = k
            new_weights[f"{prefix}.v_proj.{param_type}"] = v
        consumed.add(key)

    return new_weights, consumed


# --------------------------------------------------------------------------
# Key mapping for main checkpoint
# --------------------------------------------------------------------------

# Patterns for backbone block keys (non-QKV)
BACKBONE_BLOCK_PATTERN = re.compile(r"backbone\.encoder\.blocks\.(\d+)\.(.+)")


def map_backbone_block_key(block_idx: str, rest: str) -> str:
    """Map a backbone block sub-key to MLX name."""
    prefix = f"backbone.blocks.{block_idx}"

    # attn.proj -> attention.o_proj
    if rest.startswith("attn.proj."):
        return f"{prefix}.attention.o_proj.{rest[len('attn.proj.'):]}"

    # norm1, norm2, ls1, ls2, mlp stay the same
    return f"{prefix}.{rest}"


# Direct prefix mappings (old_prefix -> new_prefix)
# Keys matching these have old_prefix stripped and new_prefix prepended
BACKBONE_SIMPLE_MAPS = {
    "backbone.encoder.cls_token": "backbone.cls_token",
    "backbone.encoder.storage_tokens": "backbone.storage_tokens",
    "backbone.encoder.patch_embed.proj.weight": "backbone.patch_embed.projection.weight",
    "backbone.encoder.patch_embed.proj.bias": "backbone.patch_embed.projection.bias",
    "backbone.encoder.rope_embed.periods": "backbone.rope_embed.periods",
    "backbone.encoder.norm.weight": "backbone.norm.weight",
    "backbone.encoder.norm.bias": "backbone.norm.bias",
}


def convert_main_checkpoint(state_dict: dict) -> dict:
    """Convert main checkpoint keys to MLX format. Returns {mlx_key: numpy_array}."""
    result = {}

    # Split QKV first
    qkv_weights, qkv_consumed = split_qkv_weights(state_dict)
    for k, v in qkv_weights.items():
        arr = to_numpy(v)
        result[k] = arr

    for key, tensor in state_dict.items():
        if key in qkv_consumed:
            continue

        # Simple backbone mappings
        if key in BACKBONE_SIMPLE_MAPS:
            mlx_key = BACKBONE_SIMPLE_MAPS[key]
            arr = to_numpy(tensor)
            # Conv2d weight transposition for patch_embed
            if arr.ndim == 4 and "patch_embed" in key:
                arr = transpose_conv2d(arr)
            result[mlx_key] = arr
            continue

        # Backbone block keys (non-QKV)
        m = BACKBONE_BLOCK_PATTERN.match(key)
        if m:
            mlx_key = map_backbone_block_key(m.group(1), m.group(2))
            result[mlx_key] = to_numpy(tensor)
            continue

        # Mask downscaling convolutions
        if "mask_downscaling" in key and tensor.ndim == 4:
            arr = to_numpy(tensor)
            arr = transpose_conv2d(arr)
            result[key] = arr
            continue

        # ray_cond_emb conv weights
        if "ray_cond_emb" in key and "conv.weight" in key and tensor.ndim == 4:
            arr = to_numpy(tensor)
            arr = transpose_conv2d(arr)
            result[key] = arr
            continue

        # Everything else: keep the key as-is, just convert to numpy
        arr = to_numpy(tensor)
        result[key] = arr

    return result


# --------------------------------------------------------------------------
# MHR JIT model conversion
# --------------------------------------------------------------------------


def convert_mhr_model(mhr_model) -> dict:
    """Extract buffers and parameters from the JIT MHR model."""
    result = {}

    # Named buffers
    for name, tensor in mhr_model.named_buffers():
        # character_torch.* -> mhr.character.*
        mlx_name = name.replace("character_torch.", "mhr.character.")
        # face_expressions_model.* -> mhr.face_expressions.*
        mlx_name = mlx_name.replace("face_expressions_model.", "mhr.face_expressions.")
        # pose_correctives_model.* -> mhr.pose_correctives.*
        mlx_name = mlx_name.replace("pose_correctives_model.", "mhr.pose_correctives.")

        arr = to_numpy(tensor)
        # Skip empty tensors (shape contains 0)
        if 0 in arr.shape:
            continue
        result[mlx_name] = arr

    # Named parameters (pose_correctives_model weights)
    for name, tensor in mhr_model.named_parameters():
        mlx_name = name.replace("pose_correctives_model.", "mhr.pose_correctives.")
        arr = to_numpy(tensor)
        if 0 in arr.shape:
            continue
        result[mlx_name] = arr

    return result


# --------------------------------------------------------------------------
# Sharded saving
# --------------------------------------------------------------------------


def save_sharded(weights: dict, output_dir: Path):
    """Save weights as safetensors, splitting into shards if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total size
    total_bytes = sum(v.nbytes for v in weights.values())
    print(f"Total weight size: {total_bytes / 1024**3:.2f} GB")

    if total_bytes <= MAX_SHARD_SIZE:
        # Single file
        path = output_dir / "model.safetensors"
        save_file(weights, str(path))
        print(f"Saved single shard: {path} ({path.stat().st_size / 1024**3:.2f} GB)")

        # Write index
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Multiple shards
        shards = []
        current_shard = {}
        current_size = 0
        shard_idx = 1

        for key in sorted(weights.keys()):
            arr = weights[key]
            if current_size + arr.nbytes > MAX_SHARD_SIZE and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
                shard_idx += 1
            current_shard[key] = arr
            current_size += arr.nbytes

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        weight_map = {}

        for i, shard in enumerate(shards, 1):
            fname = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            path = output_dir / fname
            save_file(shard, str(path))
            print(
                f"Saved shard {i}/{num_shards}: {path} ({path.stat().st_size / 1024**3:.2f} GB)"
            )
            for key in shard:
                weight_map[key] = fname

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM 3D Body weights to MLX safetensors"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model.ckpt")
    parser.add_argument("--mhr-model", required=True, help="Path to mhr_model.pt (JIT)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    mhr_path = Path(args.mhr_model)
    output_dir = Path(args.output)

    # Load main checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Handle wrapped checkpoints (state_dict or model key)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    print(f"Checkpoint keys: {len(state_dict)}")

    # Load MHR JIT model
    print(f"Loading MHR model: {mhr_path}")
    mhr_model = torch.jit.load(str(mhr_path), map_location="cpu")

    # Convert main checkpoint
    print("Converting main checkpoint...")
    weights = convert_main_checkpoint(state_dict)
    print(f"  Main checkpoint -> {len(weights)} MLX keys")

    # Convert MHR model
    print("Converting MHR model...")
    mhr_weights = convert_mhr_model(mhr_model)
    print(f"  MHR model -> {len(mhr_weights)} MLX keys")

    # Merge
    weights.update(mhr_weights)
    print(f"Total MLX keys: {len(weights)}")

    # Check for key collisions between main and MHR
    main_keys = set(weights.keys()) - set(mhr_weights.keys())
    mhr_keys = set(mhr_weights.keys())
    collisions = main_keys & mhr_keys
    if collisions:
        print(f"WARNING: {len(collisions)} key collisions between main and MHR:")
        for k in sorted(collisions)[:10]:
            print(f"  {k}")

    # Save
    print("Saving safetensors...")
    save_sharded(weights, output_dir)

    # Save config
    config = SAM3DConfig()
    config_path = output_dir / "config.json"
    config.save(config_path)
    print(f"Saved config: {config_path}")

    # Summary
    print("\n=== Conversion Summary ===")
    prefixes = {}
    for k in sorted(weights.keys()):
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    for p, count in sorted(prefixes.items()):
        print(f"  {p}: {count} keys")

    # Report dtypes
    dtypes = {}
    for k, v in weights.items():
        dt = str(v.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
    print("\nDtype distribution:")
    for dt, count in sorted(dtypes.items()):
        print(f"  {dt}: {count} keys")

    print("\nDone.")


if __name__ == "__main__":
    main()
