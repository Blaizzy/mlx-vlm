"""Optional parity check: compares MLX DFlash against the HF reference on a
single synthetic forward pass. Requires torch + transformers + an HF checkpoint.

Usage:
    python -m mlx_vlm.models.qwen3_5_dflash.parity_check \
        --drafter z-lab/Qwen3.5-4B-DFlash
"""

import argparse

import mlx.core as mx
import numpy as np

from .load import load_dflash_drafter


def _mlx_forward(model, noise_np, target_np):
    noise = mx.array(noise_np)
    target = mx.array(target_np)
    out = model(noise, target)
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def _hf_forward(drafter_path, noise_np, target_np):
    import torch
    from transformers import AutoConfig, AutoModel

    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(drafter_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        drafter_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()

    B, L, H = noise_np.shape
    T = target_np.shape[1]
    position_ids = torch.arange(T + L, dtype=torch.long).unsqueeze(0).expand(B, -1)

    out = model(
        position_ids=position_ids,
        noise_embedding=torch.from_numpy(noise_np).float(),
        target_hidden=torch.from_numpy(target_np).float(),
    )
    return out.float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter", default="z-lab/Qwen3.5-4B-DFlash")
    p.add_argument("--ctx", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-hf", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)

    model = load_dflash_drafter(args.drafter, dtype=mx.float32)
    cfg = model.config
    B, L, H = 1, cfg.block_size, cfg.hidden_size
    T = args.ctx

    noise_np = np.random.randn(B, L, H).astype(np.float32) * 0.02
    target_np = (
        np.random.randn(B, T, len(cfg.target_layer_ids) * H).astype(np.float32) * 0.02
    )

    mlx_out = _mlx_forward(model, noise_np, target_np)
    print(f"MLX   out: shape={mlx_out.shape} mean={mlx_out.mean():.5f} std={mlx_out.std():.5f}")

    if args.skip_hf:
        return
    try:
        hf_out = _hf_forward(args.drafter, noise_np, target_np)
    except Exception as e:  # noqa: BLE001
        print(f"[skip HF] {type(e).__name__}: {e}")
        return
    print(f"HF    out: shape={hf_out.shape} mean={hf_out.mean():.5f} std={hf_out.std():.5f}")
    diff = np.abs(mlx_out - hf_out)
    print(f"|mlx - hf| max={diff.max():.4e} mean={diff.mean():.4e}")


if __name__ == "__main__":
    main()
