"""
AnyUp – learned cross-attention upsampler for segmentation masks.
MLX port of the PyTorch AnyUp module from Falcon-Perception.
"""

import mlx.core as mx
import mlx.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, num_groups=8):
        super().__init__()
        p = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=p, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=p, bias=False)
        self._use_shortcut = in_ch != out_ch
        if self._use_shortcut:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def __call__(self, x):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        return h + (self.shortcut(x) if self._use_shortcut else x)


# ---------------------------------------------------------------------------
# Encoder (Conv2d + ResBlocks)
# ---------------------------------------------------------------------------


def _reflect_pad(x, pad):
    """Reflect-pad a (N, H, W, C) tensor by `pad` pixels on each spatial side."""
    # top, bottom
    top = x[:, 1 : pad + 1][:, ::-1]
    bot = x[:, -pad - 1 : -1][:, ::-1]
    x = mx.concatenate([top, x, bot], axis=1)
    # left, right
    left = x[:, :, 1 : pad + 1][:, :, ::-1]
    right = x[:, :, -pad - 1 : -1][:, :, ::-1]
    return mx.concatenate([left, x, right], axis=2)


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        num_blocks=2,
        block_ks=1,
        reflect_padding=False,
    ):
        super().__init__()
        self._reflect = reflect_padding and kernel_size > 1
        pad = kernel_size // 2 if not self._reflect else 0
        self._rpad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.blocks = [ResBlock(out_ch, out_ch, block_ks) for _ in range(num_blocks)]

    def __call__(self, x):
        if self._reflect:
            x = _reflect_pad(x, self._rpad)
        x = self.conv(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# LearnedFeatureUnification
# ---------------------------------------------------------------------------


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = mx.zeros((out_channels, kernel_size, kernel_size, 1))

    def __call__(self, features):
        B, H, W, C = features.shape
        k = self.kernel_size
        p = k // 2
        out_ch = self.out_channels

        # Process each channel independently: NHWC → (B*C, H, W, 1)
        x = features.transpose(0, 3, 1, 2).reshape(B * C, H, W, 1)
        x = mx.pad(x, [(0, 0), (p, p), (p, p), (0, 0)])
        conv_out = mx.conv2d(x, self.basis, stride=1, padding=0)  # (B*C, H, W, out_ch)

        # Normalize by valid element count
        mask = mx.ones((1, H, W, 1))
        mask = mx.pad(mask, [(0, 0), (p, p), (p, p), (0, 0)])
        ones_k = mx.ones((1, k, k, 1))
        denom = mx.conv2d(mask, ones_k, stride=1, padding=0)
        conv_out = conv_out / denom

        # Match PyTorch's grouped-conv channel ordering + view:
        # PyTorch flat channel = c * out_ch + o, then view(B, out_ch, C, H, W)
        conv_out = conv_out.reshape(B, C, H, W, out_ch)
        flat = conv_out.transpose(0, 1, 4, 2, 3).reshape(B, C * out_ch, H, W)
        viewed = flat.reshape(B, out_ch, C, H, W)

        attn = mx.softmax(viewed, axis=1)  # softmax over out_ch
        result = mx.mean(attn, axis=2)  # mean over C → (B, out_ch, H, W)
        return result.transpose(0, 2, 3, 1)  # → (B, H, W, out_ch)


class LFUEncoder(nn.Module):
    def __init__(self, qk_dim, kernel_size_lfu=5, num_blocks=2, block_ks=1):
        super().__init__()
        self.lfu = LearnedFeatureUnification(qk_dim, kernel_size_lfu)
        self.blocks = [ResBlock(qk_dim, qk_dim, block_ks) for _ in range(num_blocks)]

    def __call__(self, x):
        x = self.lfu(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# AnyUp RoPE (2D rotary position encoding)
# ---------------------------------------------------------------------------


class AnyUpRoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.freqs = mx.zeros((2, dim))

    def __call__(self, x, coords):
        # x: (B, N, D), coords: (B, N, 2)
        angle = coords @ self.freqs  # (B, N, D)
        cos_a = mx.cos(angle)
        sin_a = mx.sin(angle)
        x1, x2 = mx.split(x, 2, axis=-1)
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos_a + rotated * sin_a


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------


def _window_mask_chunk(q_start, chunk_size, H, W, h, w, window_ratio):
    """Compute boolean window mask for a chunk of queries.

    Returns: (chunk_size, h*w) — True where attention is allowed.
    """
    q_indices = mx.arange(q_start, q_start + chunk_size)
    q_r = (q_indices // W).astype(mx.float32)
    q_c = (q_indices % W).astype(mx.float32)

    q_r_norm = (q_r + 0.5) / H
    q_c_norm = (q_c + 0.5) / W

    # Window bounds mapped to low-res key space
    r_lo = mx.floor(mx.clip(q_r_norm - window_ratio, 0, 1) * h).astype(mx.int32)
    r_hi = mx.ceil(mx.clip(q_r_norm + window_ratio, 0, 1) * h).astype(mx.int32)
    c_lo = mx.floor(mx.clip(q_c_norm - window_ratio, 0, 1) * w).astype(mx.int32)
    c_hi = mx.ceil(mx.clip(q_c_norm + window_ratio, 0, 1) * w).astype(mx.int32)

    k_r = mx.arange(h)
    k_c = mx.arange(w)

    row_ok = (k_r[None, :] >= r_lo[:, None]) & (
        k_r[None, :] < r_hi[:, None]
    )  # (chunk, h)
    col_ok = (k_c[None, :] >= c_lo[:, None]) & (
        k_c[None, :] < c_hi[:, None]
    )  # (chunk, w)

    mask = (row_ok[:, :, None] & col_ok[:, None, :]).reshape(chunk_size, h * w)
    return mask


class CrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_proj = nn.Linear(qk_dim, qk_dim)
        self.k_proj = nn.Linear(qk_dim, qk_dim)

    def __call__(
        self,
        query,
        key,
        value,
        H=None,
        W=None,
        h=None,
        w=None,
        window_ratio=0.1,
        chunk_size=4096,
    ):
        # query: (B, Q, D_qk), key: (B, KV, D_qk), value: (B, KV, D_v)
        B, Q, _ = query.shape
        _, KV, D_v = value.shape
        v_head_dim = D_v // self.num_heads

        q = self.q_proj(self.norm_q(query))
        k = self.k_proj(self.norm_k(key))

        k = k.reshape(B, KV, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = value.reshape(B, KV, self.num_heads, v_head_dim).transpose(0, 2, 1, 3)

        use_window = H is not None and W is not None and h is not None and w is not None

        outputs = []
        for i in range(0, Q, chunk_size):
            actual_chunk = min(chunk_size, Q - i)
            q_chunk = q[:, i : i + actual_chunk]
            q_chunk = q_chunk.reshape(
                B, actual_chunk, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

            scores = (q_chunk @ k.transpose(0, 1, 3, 2)) * self.scale

            if use_window:
                wmask = _window_mask_chunk(i, actual_chunk, H, W, h, w, window_ratio)
                # (chunk, KV) → (1, 1, chunk, KV) broadcast over B and heads
                scores = mx.where(wmask[None, None], scores, mx.array(float("-inf")))

            weights = mx.softmax(scores, axis=-1)
            out_chunk = (weights @ v).transpose(0, 2, 1, 3)
            out_chunk = out_chunk.reshape(B, actual_chunk, D_v)
            outputs.append(out_chunk)

        return mx.concatenate(outputs, axis=1)


class CrossDecodeBlock(nn.Module):
    def __init__(self, qk_dim, num_heads):
        super().__init__()
        self.cross_attn = CrossAttention(qk_dim, num_heads)
        self.conv = nn.Conv2d(qk_dim, qk_dim, 3, padding=1, bias=False)

    def __call__(self, q, k, v, window_ratio=0.1):
        # q: (B, H, W, C_qk), k: (B, h, w, C_qk), v: (B, h, w, C_v)
        B, H, W, _ = q.shape
        _, h_k, w_k, _ = k.shape
        q = self.conv(q)
        q_flat = q.reshape(B, H * W, -1)
        k_flat = k.reshape(B, h_k * w_k, -1)
        v_flat = v.reshape(B, h_k * w_k, -1)
        out = self.cross_attn(
            q_flat,
            k_flat,
            v_flat,
            H=H,
            W=W,
            h=h_k,
            w=w_k,
            window_ratio=window_ratio,
        )
        return out.reshape(B, H, W, -1)


# ---------------------------------------------------------------------------
# Adaptive average pooling
# ---------------------------------------------------------------------------


def adaptive_avg_pool2d(x, output_size):
    """x: (N, H, W, C) → (N, out_h, out_w, C)"""
    N, H, W, C = x.shape
    out_h, out_w = output_size
    if H == out_h and W == out_w:
        return x
    if H % out_h == 0 and W % out_w == 0:
        kh, kw = H // out_h, W // out_w
        x = x.reshape(N, out_h, kh, out_w, kw, C)
        return mx.mean(x, axis=(2, 4))
    # General case
    rows = []
    for i in range(out_h):
        h0 = (i * H) // out_h
        h1 = ((i + 1) * H) // out_h
        cols = []
        for j in range(out_w):
            w0 = (j * W) // out_w
            w1 = ((j + 1) * W) // out_w
            cols.append(mx.mean(x[:, h0:h1, w0:w1, :], axis=(1, 2), keepdims=True))
        rows.append(mx.concatenate(cols, axis=2))
    return mx.concatenate(rows, axis=1)


# ---------------------------------------------------------------------------
# AnyUp main module
# ---------------------------------------------------------------------------


class AnyUp(nn.Module):
    def __init__(self, input_dim=3, qk_dim=128, num_heads=4):
        super().__init__()
        self.qk_dim = qk_dim
        self.image_encoder = Encoder(
            input_dim, qk_dim, kernel_size=1, reflect_padding=True
        )
        self.key_encoder = Encoder(qk_dim, qk_dim, kernel_size=1, reflect_padding=True)
        self.query_encoder = Encoder(
            qk_dim, qk_dim, kernel_size=1, reflect_padding=True
        )
        self.key_features_encoder = LFUEncoder(qk_dim, kernel_size_lfu=5)
        self.aggregation = Encoder(
            2 * qk_dim, qk_dim, kernel_size=3, reflect_padding=True
        )
        self.cross_decode = CrossDecodeBlock(qk_dim, num_heads)
        self.rope = AnyUpRoPE(qk_dim)

    def __call__(self, images, features):
        """
        Args:
            images: (N, H, W, 3) pixel values in [-1, 1] normalization
            features: (N, h, w, segm_out_dim) low-res segm features from conv_segm
        Returns:
            (N, H, W, segm_out_dim) high-res features
        """
        B, H, W, _ = images.shape
        _, h, w, _ = features.shape

        # Re-normalize: [-1,1] → [0,1] → ImageNet
        mean = mx.array(IMAGENET_MEAN).reshape(1, 1, 1, 3)
        std = mx.array(IMAGENET_STD).reshape(1, 1, 1, 3)
        img = images * 0.5 + 0.5
        img = (img - mean) / std
        img = img.astype(features.dtype)

        # Encode image
        enc = self.image_encoder(img)  # (B, H, W, qk_dim)

        # Apply 2D RoPE
        y_coords = mx.linspace(0.0, 1.0, enc.shape[1])
        x_coords = mx.linspace(0.0, 1.0, enc.shape[2])
        yy = mx.broadcast_to(y_coords[:, None], (enc.shape[1], enc.shape[2]))
        xx = mx.broadcast_to(x_coords[None, :], (enc.shape[1], enc.shape[2]))
        coords = mx.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)
        coords = coords.reshape(1, -1, 2)  # (1, H*W, 2)

        enc_flat = enc.reshape(B, -1, self.qk_dim)
        enc_flat = self.rope(enc_flat, coords)
        enc = enc_flat.reshape(B, enc.shape[1], enc.shape[2], self.qk_dim)

        # Build query, key, value
        q = self.query_encoder(enc)  # (B, H, W, qk_dim)
        k = adaptive_avg_pool2d(self.key_encoder(enc), (h, w))  # (B, h, w, qk_dim)

        # Key features from low-res features
        feat_norm = features / mx.sqrt(
            mx.clip(
                mx.sum(features * features, axis=-1, keepdims=True),
                a_min=1e-12,
                a_max=None,
            )
        )
        k_feat = self.key_features_encoder(feat_norm)  # (B, h, w, qk_dim)

        k = mx.concatenate([k, k_feat], axis=-1)  # (B, h, w, 2*qk_dim)
        k = self.aggregation(k)  # (B, h, w, qk_dim)
        v = features  # (B, h, w, segm_out_dim)

        # Cross-attention upsample
        return self.cross_decode(q, k, v)  # (B, H, W, segm_out_dim)
