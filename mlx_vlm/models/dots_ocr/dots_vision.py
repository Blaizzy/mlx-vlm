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
    pos_w = np.arange(W, dtype=np.float32)[:, None]  # [W,1]
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

    return mx.array(cos, dtype=mx.float32), mx.array(sin, dtype=mx.float32)


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
    cos = cos[:, None, : q1.shape[-1]]  # [seq, 1, hd/2]
    sin = sin[:, None, : q1.shape[-1]]
    # (a + jb) * (cos + j sin) = (a cos - b sin) + j(a sin + b cos)
    q_rot = mx.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = mx.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


# ---- Attention with fused QKV and optional block-diag mask ----


def build_block_mask_from_cu(cu):
    """
    cu: cumulative lengths [0, L0, L0+L1, ...]
    returns bool mask [total, total] where mask[i,j]==True if in same block
    """

    import numpy as np

    if hasattr(cu, "tolist"):
        cu_np = np.array(cu.tolist(), dtype=np.int64)
    else:
        cu_np = np.array(cu, dtype=np.int64)

    total = int(cu_np[-1]) if len(cu_np) > 0 else 0
    mask = np.zeros((total, total), dtype=bool)
    for start, end in zip(cu_np[:-1], cu_np[1:]):
        s, e = int(start), int(end)
        mask[s:e, s:e] = True

    return mx.array(mask, dtype=mx.bool_)


class VisionAttention(nn.Module):
    def __init__(self, dim=1536, heads=12):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def _sdpa(self, q, k, v, mask):
        # q,k,v: [seq, heads, head_dim]
        scale = self.head_dim ** -0.5
        q_t = q.transpose(1, 0, 2)  # [heads, seq, head_dim]
        k_t = k.transpose(1, 0, 2)
        v_t = v.transpose(1, 0, 2)

        attn = mx.matmul(q_t, k_t.transpose(0, 2, 1)) * scale  # [heads, seq, seq]

        if mask is not None:
            mask_bool = mask.astype(mx.bool_)[None, ...]
            large_neg = mx.array(-1e9, dtype=attn.dtype)
            attn = mx.where(mask_bool, attn, large_neg)

        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v_t)  # [heads, seq, head_dim]
        return out.transpose(1, 0, 2)  # [seq, heads, head_dim]

    def __call__(self, x: mx.array, mask: mx.array | None, cos: mx.array, sin: mx.array):
        # x: [seq, dim]
        seq = x.shape[0]
        qkv = self.qkv(x)  # [seq, 3*dim]
        qkv = qkv.reshape(seq, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rotary(q, k, cos, sin)
        attn_out = self._sdpa(q, k, v, mask)
        attn_out = attn_out.reshape(seq, self.dim)
        return self.proj(attn_out)


class VisionBlock(nn.Module):
    def __init__(self, dim=1536, heads=12, mlp_hidden=4224, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = VisionAttention(dim, heads)
        self.norm2 = RMSNorm(dim, eps)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def __call__(self, x, mask, cos, sin):
        x = x + self.attn(self.norm1(x), mask, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ---- PatchMerger (2x2) + post path ----
class PatchMerger(nn.Module):
    """
    Merges non-overlapping 2x2 token windows spatially:
      x -> RMSNorm -> [concat 4 tokens] -> Linear -> GELU -> Linear
    Input:  x [H*W, D] (single image token sequence)
    Output: y [(H/2)*(W/2), D]
    """

    def __init__(self, dim=1536, merge=2, eps=1e-6):
        super().__init__()
        assert merge == 2, "dots.ocr uses 2x2 merger"
        self.merge = merge
        self.ln = RMSNorm(dim, eps)
        # merger MLP commonly uses bias (per HF weights)
        self.mlp0 = nn.Linear(dim * merge * merge, dim, bias=True)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, H: int, W: int) -> mx.array:
        # x: [H*W, D]
        m = self.merge
        H2, W2 = H // m, W // m
        # pre-norm
        x = self.ln(x)
        # reshape to grid
        x = x.reshape(H, W, -1)[: H2 * m, : W2 * m, :]
        # group 2x2 windows -> concat 4 tokens
        x = x.reshape(H2, m, W2, m, -1)
        x = x.transpose(0, 2, 1, 3, 4)
        x = x.reshape(H2 * W2, m * m * x.shape[-1])
        # MLP with GELU
        x = self.mlp2(self.act(self.mlp0(x)))
        return x


# ---- Vision wrapper ----


class DotsVisionTransformer_MLX(nn.Module):
    """
    End-to-end vision tower:
      PatchEmbed -> [Blocks]*N -> (optional) RMSNorm -> per-image PatchMerger
    __call__(pixels, grid_thw):
      pixels: [B, 3, H, W] (for tests we use B=1)
      grid_thw: list of [1, H', W'] for each image (H',W' are patch grids)
    Returns:
      concatenated merged tokens of shape [sum_i(H'W'/4), D]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        v = cfg.vision
        self.patch = PatchEmbed(3, v.embed_dim, v.patch_size, v.rms_eps)
        self.blocks = [
            VisionBlock(v.embed_dim, v.num_heads, 4224, v.rms_eps)
            for _ in range(v.num_layers)
        ]
        self.post = RMSNorm(v.embed_dim, v.rms_eps) if v.post_norm else None
        self.merger = PatchMerger(v.embed_dim, v.merge_size, v.rms_eps)

    @staticmethod
    def build_cu_from_grid(grid_thw):
        # grid_thw: list of [1,H',W']
        cu = [0]
        for _, H, W in grid_thw:
            cu.append(cu[-1] + H * W)
        return mx.array(cu, dtype=mx.int32)

    def __call__(self, pixels: mx.array, grid_thw: list[list[int]]) -> mx.array:
        tokens, Hp, Wp = self.patch(pixels)
        v = self.cfg.vision
        cos_list, sin_list = [], []
        for _, H, W in grid_thw:
            cos, sin = build_2d_rotary_cos_sin(
                H, W, (v.embed_dim // v.num_heads) // 2, v.rotary_theta
            )
            cos_list.append(cos)
            sin_list.append(sin)
        cos = mx.concatenate(cos_list, axis=0)
        sin = mx.concatenate(sin_list, axis=0)
        cu = self.build_cu_from_grid(grid_thw)
        mask = build_block_mask_from_cu(cu)
        x = tokens
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        if self.post is not None:
            x = self.post(x)
        outs = []
        start = 0
        for _, H, W in grid_thw:
            n = H * W
            xi = x[start : start + n]
            start += n
            outs.append(self.merger(xi, H, W))
        return mx.concatenate(outs, axis=0)
