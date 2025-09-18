import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., dim]
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int = 1536, hidden: int = 4224):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)  # up
        self.fc3 = nn.Linear(in_features, hidden, bias=False)  # gate
        self.fc2 = nn.Linear(hidden, in_features, bias=False)  # down

    def __call__(self, x: mx.array) -> mx.array:
        up = self.fc1(x)
        gate = self.fc3(x)
        sigmoid = 1.0 / (1.0 + mx.exp(-up))
        silu = up * sigmoid
        return self.fc2(silu * gate)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=1536, patch=14, eps=1e-6):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch, stride=patch, bias=False
        )
        self.norm = RMSNorm(embed_dim, eps=eps)
        self.patch = patch

    def __call__(self, x: mx.array):
        """
        x: [B,3,H,W]
        returns: tokens [B*Hp*Wp, embed_dim], Hp, Wp
        """
        x_nhwc = x.transpose(0, 2, 3, 1)  # [B, H, W, C]
        y = self.proj(x_nhwc)  # [B, Hp, Wp, D]
        B, Hp, Wp, D = y.shape
        y = y.reshape(B * Hp * Wp, D)
        y = self.norm(y)
        return y, int(Hp), int(Wp)

# ---- 2D Rotary Positional Embeddings ----

def build_2d_rotary_cos_sin(H: int, W: int, head_dim_half: int, theta: float = 10000.0):
    """
    Returns cos, sin with shape [H*W, 2*head_dim_half]
    Very simple 2D blend: produce 1D RoPE for H and W, then concatenate.
    """
    import numpy as np

    assert head_dim_half > 0
    # freq vector
    freqs = 1.0 / (theta ** (np.arange(head_dim_half, dtype=np.float32) / head_dim_half))
    # positions
    pos_h = np.arange(H, dtype=np.float32)[:, None]  # [H,1]
    pos_w = np.arange(W, dtype=np.float32)[None, :]  # [1,W]
    ang_h = pos_h * freqs[None, :]  # [H, head_dim_half]
    ang_w = pos_w * freqs[None, :]  # [W, head_dim_half]
    # broadcast to grid and flatten
    cos_h, sin_h = np.cos(ang_h), np.sin(ang_h)  # [H, d/2]
    cos_w, sin_w = np.cos(ang_w), np.sin(ang_w)  # [W, d/2]
    # tile to HxW by outer sum: (h,w) -> (cos_h[h]*cos_w[w], sin_h[h]*sin_w[w])
    # simple concat of row/col parts
    cos = np.concatenate(
        [
            np.repeat(cos_h, W, axis=0),  # [H*W, d/2]
            np.tile(cos_w, (H, 1)),  # [H*W, d/2]
        ],
        axis=-1,
    )  # [H*W, d]
    sin = np.concatenate(
        [
            np.repeat(sin_h, W, axis=0),
            np.tile(sin_w, (H, 1)),
        ],
        axis=-1,
    )
    import mlx.core as mx

    return mx.array(cos), mx.array(sin)


def apply_rotary(q, k, cos, sin):
    """
    q,k: [seq, heads, head_dim]
    cos,sin: [seq, head_dim]  (head_dim must be even)
    Applies RoPE to (q,k) and returns transformed q,k.
    """
    import mlx.core as mx

    hd = q.shape[-1]
    assert hd % 2 == 0, "head_dim must be even for rotary split"
    # split last dimension
    q1, q2 = mx.split(q, 2, axis=-1)
    k1, k2 = mx.split(k, 2, axis=-1)
    cos = cos[..., : q1.shape[-1]]  # [seq, hd/2]
    sin = sin[..., : q1.shape[-1]]
    # (a + jb) * (cos + j sin) = (a cos - b sin) + j(a sin + b cos)
    q_rot = mx.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = mx.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot
