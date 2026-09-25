import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self._inv_freq = 1.0 / (
            theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )

    def __call__(self, seqlen: int) -> mx.array:
        return mx.outer(mx.arange(seqlen).astype(mx.float32), self._inv_freq)


class PatchEmbed(nn.Module):
    """Conv3d patch embedding.

    Kernel and stride are equal, so the convolution is a linear map over
    flattened ``in_chans x temporal_patch_size x patch_size x patch_size``
    patches; :meth:`VisionModel.sanitize` reshapes the checkpoint's 5D weight
    to match.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        self.proj = nn.Linear(self.patch_dim, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x.reshape(-1, self.patch_dim))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig, use_sinks: bool, window_size: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.qk_channels
        self.scale = self.head_dim**-0.5
        self.window_size = window_size

        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(config.hidden_size, qkv_dim)
        self.proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.sinks = mx.zeros((self.num_heads,)) if use_sinks else None

    def _mask(self, n: int, full_attn: bool, dtype) -> mx.array:
        mask = None
        if not full_attn and self.window_size > 0:
            idx = mx.arange(n)
            outside = mx.abs(idx[:, None] - idx[None, :]) > self.window_size
            mask = mx.where(outside, mx.array(-mx.inf, dtype), mx.array(0, dtype))
            mask = mask[None, None]
        if self.sinks is not None:
            # constant across queries, so [1, H, 1, n] rather than [1, H, n, n]
            sink = mx.zeros((1, self.num_heads, 1, n), dtype)
            sink[..., 0] = self.sinks.reshape(1, self.num_heads, 1).astype(dtype)
            mask = sink if mask is None else mask + sink
        return mask

    def _attend(self, q, k, v, full_attn: bool) -> mx.array:
        batch_size, n = q.shape[:2]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=self._mask(n, full_attn, q.dtype)
        )
        return out.transpose(0, 2, 1, 3).reshape(batch_size, n, -1)

    def __call__(self, x: mx.array, cos, sin, full_attn: bool, cu_seqlens) -> mx.array:
        n = x.shape[0]
        qkv = self.qkv(x)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        q = qkv[:, :q_dim].reshape(n, self.num_heads, self.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].reshape(n, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim :].reshape(n, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        spans = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        groups = {}
        for index, (start, stop) in enumerate(spans):
            groups.setdefault(stop - start, []).append((index, start, stop))
        chunks = [None] * len(spans)
        for items in groups.values():
            q_batch = mx.stack([q[start:stop] for _, start, stop in items])
            k_batch = mx.stack([k[start:stop] for _, start, stop in items])
            v_batch = mx.stack([v[start:stop] for _, start, stop in items])
            outputs = self._attend(q_batch, k_batch, v_batch, full_attn)
            for output, (index, _, _) in zip(outputs, items):
                chunks[index] = output
        attn = mx.concatenate(chunks, axis=0)
        return self.proj(attn)


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, use_sinks: bool, window_size: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = VisionAttention(config, use_sinks, window_size)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, cos, sin, full_attn: bool, cu_seqlens) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin, full_attn, cu_seqlens)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_pos_emb_vision(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    dtype = q.dtype
    cos = mx.expand_dims(cos, -2).astype(mx.float32)
    sin = mx.expand_dims(sin, -2).astype(mx.float32)
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q.astype(dtype), k.astype(dtype)


class VisionModel(nn.Module):
    """MiMo-V2.6 vision tower (``vision_model_type: mimovl``)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_attn_types = config.vit_window_attn_types or [-1] * config.depth

        self.patch_embed = PatchEmbed(config)
        self.rotary_pos_emb = VisionRotaryEmbedding(config.qk_channels // 2)
        self.blocks = [
            VisionBlock(
                config,
                use_sinks=config.use_sink and i not in config.fullatt_block_indexes,
                window_size=config.visual_token_window_size,
            )
            for i in range(config.depth)
        ]
        self.merger = PatchMerger(config)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.endswith("patch_embed.proj.weight") and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            out[k] = v
        return out

    def _apply_index(self, x: mx.array, index: mx.array) -> mx.array:
        x = x.reshape(-1, self.spatial_merge_unit, *x.shape[1:])
        return x[index].reshape(-1, *x.shape[2:])

    def _window_index(self, grid_thw) -> mx.array:
        out, offset = [], 0
        m = self.spatial_merge_size
        for t, h, w in grid_thw:
            gh, gw = h // m, w // m
            index = mx.arange(t * gh * gw).reshape(t, gh, gw)
            out.append(index.transpose(0, 2, 1).reshape(-1) + offset)
            offset += t * gh * gw
        return mx.concatenate(out, axis=0)

    def _rot_pos_emb(self, grid_thw) -> mx.array:
        m = self.spatial_merge_size
        pos_ids = []
        for t, h, w in grid_thw:
            hpos = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
            hpos = hpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            wpos = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
            wpos = wpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            pos_ids.append(mx.tile(mx.stack([hpos, wpos], axis=-1), (t, 1)))
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid = int(max(max(h, w) for _, h, w in grid_thw))
        full = self.rotary_pos_emb(max_grid)
        return full[pos_ids].reshape(pos_ids.shape[0], -1)

    def __call__(self, pixel_values: mx.array, grid_thw) -> mx.array:
        grid_thw = [tuple(int(v) for v in row) for row in grid_thw]
        x = self.patch_embed(pixel_values)

        rotary = self._rot_pos_emb(grid_thw)
        emb = mx.concatenate([rotary, rotary], axis=-1)
        row_cos, row_sin = mx.cos(emb), mx.sin(emb)
        col_index = self._window_index(grid_thw)
        reverse_index = mx.argsort(col_index)
        col_emb = self._apply_index(emb, col_index)
        col_cos, col_sin = mx.cos(col_emb), mx.sin(col_emb)

        cu_seqlens, total = [0], 0
        for t, h, w in grid_thw:
            for _ in range(t):
                total += h * w
                cu_seqlens.append(total)

        types = self.window_attn_types
        for i, block in enumerate(self.blocks):
            if types[i] == 1 and (i == 0 or types[i - 1] != 1):
                x = self._apply_index(x, col_index)
            elif i > 0 and types[i] != 1 and types[i - 1] == 1:
                x = self._apply_index(x, reverse_index)
            cos, sin = (col_cos, col_sin) if types[i] == 1 else (row_cos, row_sin)
            x = block(x, cos, sin, i in self.fullatt_block_indexes, cu_seqlens)

        return self.merger(x)
