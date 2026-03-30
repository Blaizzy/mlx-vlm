from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import KVCache
from .config import ModelConfig, TextConfig


def precompute_freqs_1d(
    dim: int, end: int, theta: float = 10000.0
) -> Tuple[mx.array, mx.array]:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32)[: dim // 2] / dim))
    t = mx.arange(end).astype(mx.float32)
    freqs = t[:, None] * freqs[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def apply_rotary_emb_1d(
    xq: mx.array, xk: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    dtype = xq.dtype
    *shape_q, d = xq.shape
    *shape_k, _ = xk.shape
    xq_r = xq.astype(mx.float32).reshape(*shape_q, d // 2, 2)
    xk_r = xk.astype(mx.float32).reshape(*shape_k, d // 2, 2)
    xq_0, xq_1 = xq_r[..., 0], xq_r[..., 1]
    xk_0, xk_1 = xk_r[..., 0], xk_r[..., 1]
    c = cos.reshape(1, 1, -1, cos.shape[-1])
    s = sin.reshape(1, 1, -1, sin.shape[-1])
    oq = mx.stack([xq_0 * c - xq_1 * s, xq_0 * s + xq_1 * c], axis=-1)
    ok = mx.stack([xk_0 * c - xk_1 * s, xk_0 * s + xk_1 * c], axis=-1)
    return oq.reshape(*shape_q, d).astype(dtype), ok.reshape(*shape_k, d).astype(dtype)


def compute_golden_freqs(
    freqs_golden: mx.array, pos_hw: mx.array
) -> Tuple[mx.array, mx.array]:
    theta = mx.einsum(
        "bsp,hfp->bshf", pos_hw.astype(mx.float32), freqs_golden.astype(mx.float32)
    )
    return mx.cos(theta), mx.sin(theta)


def apply_golden_rotary_emb(
    x: mx.array, cos_2d: mx.array, sin_2d: mx.array
) -> mx.array:
    dtype = x.dtype
    cos = cos_2d.transpose(0, 2, 1, 3)
    sin = sin_2d.transpose(0, 2, 1, 3)
    x_f = x.astype(mx.float32)
    x_even, x_odd = x_f[..., 0::2], x_f[..., 1::2]
    o_even = x_even * cos - x_odd * sin
    o_odd = x_even * sin + x_odd * cos
    return mx.stack([o_even, o_odd], axis=-1).reshape(x.shape).astype(dtype)


def apply_3d_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    cos_1d: mx.array,
    sin_1d: mx.array,
    cos_2d: Optional[mx.array] = None,
    sin_2d: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    half = xq.shape[-1] // 2
    xq_t, xq_hw = xq[..., :half], xq[..., half:]
    xk_t, xk_hw = xk[..., :half], xk[..., half:]
    xq_t, xk_t = apply_rotary_emb_1d(xq_t, xk_t, cos_1d, sin_1d)
    if cos_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, cos_2d, sin_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, cos_2d, sin_2d)
    dtype_q, dtype_k = xq.dtype, xk.dtype
    return (
        mx.concatenate([xq_t, xq_hw], axis=-1).astype(dtype_q),
        mx.concatenate([xk_t, xk_hw], axis=-1).astype(dtype_k),
    )


def compute_pos_hw(
    input_ids: mx.array,
    image_token_id: int,
    image_grid_hws: Optional[List[Tuple[int, int]]] = None,
) -> mx.array:
    num_tokens = input_ids.shape[-1]
    ids = input_ids.reshape(-1).tolist()
    img_indices = [i for i, t in enumerate(ids) if t == image_token_id]
    if not img_indices:
        return mx.zeros((1, num_tokens, 2))

    all_coords: List[Tuple[float, float]] = []
    if image_grid_hws:
        for gh, gw in image_grid_hws:
            for hi in range(gh):
                for wi in range(gw):
                    h_val = -((gh / gw) ** 0.5) + 2 * ((gh / gw) ** 0.5) * hi / max(
                        gh - 1, 1
                    )
                    w_val = -((gw / gh) ** 0.5) + 2 * ((gw / gh) ** 0.5) * wi / max(
                        gw - 1, 1
                    )
                    all_coords.append((h_val, w_val))

    hw_vals = [[0.0, 0.0]] * num_tokens
    for idx_i, tok_idx in enumerate(img_indices):
        if idx_i < len(all_coords):
            hw_vals[tok_idx] = list(all_coords[idx_i])
    return mx.array(hw_vals).reshape(1, num_tokens, 2)


def create_falcon_ocr_mask(
    input_ids: mx.array,
    image_cls_id: int,
    img_end_id: int,
) -> mx.array:
    """Causal mask with bidirectional attention within image blocks.

    Returns additive float mask ``(1, 1, S, S)``: 0 = attend, -inf = masked.
    """
    ids = input_ids.reshape(-1)
    S = ids.shape[0]

    soi = (ids == image_cls_id).astype(mx.int32)
    eoi = (ids == img_end_id).astype(mx.int32)
    acc_soi = mx.cumsum(soi)
    acc_eoi = mx.cumsum(eoi)
    in_image = (acc_soi - acc_eoi) > 0  # (S,) True inside image blocks
    block_id = acc_soi * in_image.astype(mx.int32)  # 1-indexed block per token

    q = mx.arange(S)
    causal = q[:, None] >= q[None, :]  # (S, S)

    q_in = in_image[:, None]  # (S, 1)
    kv_in = in_image[None, :]  # (1, S)
    q_blk = block_id[:, None]
    kv_blk = block_id[None, :]
    same_image = q_in & kv_in & (q_blk == kv_blk)  # (S, S)

    attend = causal | same_image
    return attend.reshape(1, 1, S, S)


class Attention(nn.Module):

    def __init__(self, args: TextConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = args.rms_norm_eps
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.wqkv = nn.Linear(args.hidden_size, q_size + 2 * kv_size, bias=False)
        self.wo = nn.Linear(q_size, args.hidden_size, bias=False)
        self.sinks = mx.zeros((self.n_heads,))
        self._norm_w_in = mx.ones((args.hidden_size,))
        self._norm_w_qk = mx.ones((self.head_dim,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        cos_1d: Optional[mx.array] = None,
        sin_1d: Optional[mx.array] = None,
        cos_2d: Optional[mx.array] = None,
        sin_2d: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        x_norm = mx.fast.rms_norm(x, self._norm_w_in, eps=self.eps)

        qkv = self.wqkv(x_norm)
        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size : self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size :]

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rms_norm(q, self._norm_w_qk, eps=self.eps)
        k = mx.fast.rms_norm(k, self._norm_w_qk, eps=self.eps)

        # KV must be expanded to n_heads BEFORE rotary because the 2D golden
        # rotary has per-head frequencies (freqs_cis_golden shape: [n_heads, F, 2]).
        # Each repeated copy receives a different spatial rotation
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        if cos_1d is not None:
            q, k = apply_3d_rotary_emb(q, k, cos_1d, sin_1d, cos_2d, sin_2d)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class MLP(nn.Module):

    def __init__(self, args: TextConfig):
        super().__init__()
        self.hidden_dim = args.intermediate_size
        self.eps = args.rms_norm_eps
        self.w13 = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self._norm_w = mx.ones((args.hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = mx.fast.rms_norm(x, self._norm_w, eps=self.eps)
        w13_out = self.w13(x_norm)
        gate = w13_out[..., : self.hidden_dim]
        up = w13_out[..., self.hidden_dim :]
        activated = nn.relu(gate) ** 2 * up
        return self.w2(activated)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        x = x + self.self_attn(x, mask=mask, cache=cache, **kwargs)
        x = x + self.mlp(x)
        return x


class FalconOCRTransformerModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig):
        super().__init__()
        self.args = args
        self.config = config
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        patch_dim = (
            config.vision_config.temporal_patch_size
            * (config.vision_config.spatial_patch_size**2)
            * config.vision_config.channel_size
        )
        self.img_projector = nn.Linear(patch_dim, args.hidden_size, bias=False)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        rope_dim = args.head_dim // 2
        cos_1d, sin_1d = precompute_freqs_1d(
            rope_dim, args.max_position_embeddings, args.rope_theta
        )
        self.cos_1d = cos_1d
        self.sin_1d = sin_1d
        self.freqs_cis_golden = mx.zeros((args.num_attention_heads, rope_dim // 2, 2))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        pos_hw: Optional[mx.array] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        B, L, _ = h.shape

        if position_ids is None:
            offset = 0
            if cache[0] is not None:
                offset = cache[0].offset
                if isinstance(offset, mx.array):
                    offset = offset.item()
            position_ids = mx.arange(offset, offset + L)

        if position_ids.ndim > 1:
            pos_t = position_ids.reshape(-1)[:L]
        else:
            pos_t = position_ids[:L]

        cos_1d = self.cos_1d[pos_t]
        sin_1d = self.sin_1d[pos_t]

        cos_2d, sin_2d = None, None
        if pos_hw is not None:
            cos_2d, sin_2d = compute_golden_freqs(self.freqs_cis_golden, pos_hw)

        for layer, c in zip(self.layers, cache):
            h = layer(
                h,
                mask=mask,
                cache=c,
                cos_1d=cos_1d,
                sin_1d=sin_1d,
                cos_2d=cos_2d,
                sin_2d=sin_2d,
            )

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = FalconOCRTransformerModel(args, config)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._rope_delta = None
        self._position_ids = None
        self._pos_hw = None
        self._full_attn_mask = None

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        kwargs.pop("image_grid_hw", None)
        kwargs.pop("pixel_values", None)

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()

        if inputs_embeds is not None:
            L = inputs_embeds.shape[1]
        elif inputs.ndim > 1:
            L = inputs.shape[1]
        else:
            L = 1

        position_ids = None
        pos_hw = None

        if self._position_ids is not None and inputs_embeds is not None:
            position_ids = self._position_ids[cache_offset : cache_offset + L]
            if self._pos_hw is not None:
                pos_hw = self._pos_hw[:, cache_offset : cache_offset + L, :]
        elif self._rope_delta is not None and cache_offset > 0:
            start = cache_offset + self._rope_delta
            position_ids = mx.arange(start, start + L, dtype=mx.int32)

        if mask is None and self._full_attn_mask is not None and L > 1:
            end = cache_offset + L
            mask = self._full_attn_mask[:, :, cache_offset:end, :end]

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mask=mask,
            position_ids=position_ids,
            pos_hw=pos_hw,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(
            logits=out.astype(self.model.embed_tokens.weight.dtype)
        )

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
