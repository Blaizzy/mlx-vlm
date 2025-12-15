from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig


class HunyuanRotaryEmbedding:
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        # Handle xdrope/dynamic scaling
        self.xdrope_section = config.rope_scaling.get("xdrope_section")
        self.rope_type = config.rope_scaling.get("type")
        alpha = config.rope_scaling.get("alpha")

        if config.rope_scaling is not None and self.rope_type in ["xdrope", "dynamic"]:
            if alpha:
                self.base = self.base * (alpha ** (self.dim / (self.dim - 2)))

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self._inv_freq = inv_freq
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, dtype: mx.Dtype):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = mx.arange(seq_len, dtype=mx.float32)
            freqs = mx.outer(t, self._inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            self._cos_cached = mx.cos(emb).astype(dtype)
            self._sin_cached = mx.sin(emb).astype(dtype)

    def __call__(self, x: mx.array, seq_len: int) -> Tuple[mx.array, mx.array]:
        self._update_cache(seq_len, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_xdrope(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: mx.array,
    xdrope_section: list,
    output_size: tuple,
) -> Tuple[mx.array, mx.array]:
    """Applies XD Rotary Position Embedding."""

    x_dim = len(xdrope_section)
    cos = (
        cos[position_ids, ...]
        .transpose(0, 2, 1, 3)
        .reshape(output_size[0], output_size[2], x_dim, -1)
    )
    sin = (
        sin[position_ids, ...]
        .transpose(0, 2, 1, 3)
        .reshape(output_size[0], output_size[2], x_dim, -1)
    )

    xdrope_section = xdrope_section * 2

    # for xd concat
    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"

    # Convert split sizes to split indices for MLX
    split_indices = [
        sum(xdrope_section[: i + 1]) for i in range(len(xdrope_section) - 1)
    ]
    cos_splits = mx.split(cos, split_indices, axis=-1)
    sin_splits = mx.split(sin, split_indices, axis=-1)

    cos = mx.concatenate(
        [m[:, :, i % x_dim, :] for i, m in enumerate(cos_splits)], axis=-1
    )
    sin = mx.concatenate(
        [m[:, :, i % x_dim, :] for i, m in enumerate(sin_splits)], axis=-1
    )

    # for head repeat
    cos = cos.reshape(output_size[0], 1, output_size[2], -1)
    sin = sin.reshape(output_size[0], 1, output_size[2], -1)

    origin_dtype = q.dtype
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    cos, sin = cos.astype(mx.float32), sin.astype(mx.float32)

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)

    return q_out.astype(origin_dtype), k_out.astype(origin_dtype)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
) -> Tuple[mx.array, mx.array]:
    """Standard rotary position embedding."""
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = HunyuanRotaryEmbedding(config=config)

        self.xdrope_section = None
        if config.rope_scaling is not None:
            self.xdrope_section = config.rope_scaling.get("xdrope_section")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = L
        offset = 0
        if cache is not None:
            offset = cache.offset
            kv_seq_len += offset

        cos, sin = self.rotary_emb(values, seq_len=kv_seq_len)

        # Apply rotary embeddings
        if self.xdrope_section is not None and (cache is None or offset == 0):
            # XD RoPE for prefill (first forward pass)
            output_size = (B, self.n_heads, L, L)
            queries, keys = apply_rotary_pos_emb_xdrope(
                queries,
                keys,
                cos,
                sin,
                position_ids,
                self.xdrope_section,
                output_size,
            )
        else:
            # Standard RoPE for decode (subsequent tokens)
            if cache is not None and offset > 0:
                cos = cos[-L:]
                sin = sin[-L:]
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        # Apply QK normalization if configured
        if self.config.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        # Update cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Apply mask
        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r

        # MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r

        return out


class HunyuanModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:

        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.args = config.text_config
        self.config = config
        self.model_type = self.args.model_type
        self.model = HunyuanModel(self.args)

        if not self.args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.args.hidden_size, self.args.vocab_size, bias=False
            )

    def get_xdrope_input_positions(
        self,
        input_tokens: List[int],
        image_grid_thw: Optional[mx.array],
        image_token_id: int,
        spatial_merge_size: int,
    ) -> mx.array:
        """Compute XD-RoPE position IDs for image-text interleaved inputs."""

        xd_num = len(self.args.rope_scaling["xdrope_section"])

        input_tokens_arr = np.array(input_tokens)
        image_start_indices = np.where(input_tokens_arr == image_token_id)[0].tolist()

        seq_len = len(input_tokens)
        p_index = np.arange(seq_len)
        w_index = np.arange(seq_len)
        h_index = np.arange(seq_len)
        t_index = np.arange(seq_len)

        # Process image positions if we have images
        if image_grid_thw is not None and len(image_start_indices) > 0:
            for image_index in range(len(image_start_indices)):
                # +2: skip first image_token and account for xdrope positions
                pos = int(image_start_indices[image_index]) + 1
                _, h, w = image_grid_thw.flatten().tolist()

                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size

                token_num = (llm_grid_w + 1) * llm_grid_h

                # Ensure we don't go out of bounds
                end_pos = min(pos + token_num, seq_len)
                actual_token_num = end_pos - pos

                if actual_token_num > 0:
                    # w_index: [0, 1, ..., grid_w, 0, 1, ..., grid_w, ...] repeated for each row
                    w_pattern = np.tile(np.arange(llm_grid_w + 1), llm_grid_h)[
                        :actual_token_num
                    ]
                    w_index[pos:end_pos] = w_pattern

                    # h_index: [0, 0, ..., 0, 1, 1, ..., 1, ...] each repeated (grid_w + 1) times
                    h_pattern = np.repeat(np.arange(llm_grid_h), llm_grid_w + 1)[
                        :actual_token_num
                    ]
                    h_index[pos:end_pos] = h_pattern

                    # t_index: image index for temporal dimension
                    t_index[pos:end_pos] = image_index

        # Stack based on number of xdrope dimensions
        if xd_num == 4:
            llm_positions = mx.stack(
                [
                    mx.array(p_index),
                    mx.array(t_index),
                    mx.array(h_index),
                    mx.array(w_index),
                ]
            )
        elif xd_num == 3:
            llm_positions = mx.stack(
                [
                    mx.array(t_index),
                    mx.array(h_index),
                    mx.array(w_index),
                ]
            )
        else:
            # Fallback: just use sequential positions
            llm_positions = mx.stack([mx.array(p_index)] * xd_num)

        return llm_positions

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:

        position_ids = kwargs.pop("position_ids", None)

        # Compute cache offset
        cache_offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                cache_offset = int(offset)

        if position_ids is None and (cache is None or cache_offset == 0):
            if input_ids is not None:
                position_ids = self.get_xdrope_input_positions(
                    input_tokens=input_ids[0].tolist(),
                    image_grid_thw=kwargs.get("image_grid_thw", None),
                    image_token_id=self.config.image_token_id,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                )[None, ...]

        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
