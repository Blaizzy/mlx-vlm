from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    kv_sequence_length,
    scaled_dot_product_attention,
)
from ..rope_utils import (
    apply_rotary_pos_emb_even_odd,
    compute_mrope_frequencies,
    mrope_position_selector,
)
from .config import ModelConfig, TextConfig


def _compute_default_rope_parameters(
    config: Optional[TextConfig] = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:

    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0

    inv_freq = 1.0 / (
        base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
    )
    return inv_freq, attention_factor


class GlmOcrRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        self.rope_type = config.rope_parameters.get("rope_type", "default")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.mrope_section = config.rope_parameters.get("mrope_section", [16, 24, 24])

        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._position_selector = mrope_position_selector(
            "split_select",
            self.mrope_section,
            self._inv_freq.shape[0],
        )

    def __call__(self, x, position_ids):
        freqs = compute_mrope_frequencies(
            position_ids,
            self._inv_freq,
            self.mrope_section,
            style="split_select",
            position_selector=self._position_selector,
        )
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    Matches PyTorch's GLM-OCR implementation exactly.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        cos: Cosine tensor of shape (batch, seq_len, head_dim)
        sin: Sine tensor of shape (batch, seq_len, head_dim)
    """
    return apply_rotary_pos_emb_even_odd(q, k, cos, sin)


class GlmOcrAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope_parameters = args.rope_parameters

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings

        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : kv_sequence_length(keys)]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class GlmOcrMLP(nn.Module):
    def __init__(
        self, config: TextConfig, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class GlmOcrDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = GlmOcrAttention(config)
        self.mlp = GlmOcrMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        r = x

        x = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)

        x = self.post_self_attn_layernorm(x)
        x = r + x

        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.post_mlp_layernorm(x)
        x = r + x
        return x


class GlmOcrTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GlmOcrDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = GlmOcrRotaryEmbedding(config)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:

        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds.astype(self.norm.weight.dtype)

        if position_ids is None:
            position_ids = mx.arange(cache[0].offset, cache[0].offset + h.shape[-2])
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        position_embeddings = self.rotary_emb(h, position_ids)

        if mask is None:
            mask = create_attention_mask(
                h, cache[0] if cache and cache[0] is not None else cache
            )

        if cache is None:
            cache = [None] * self.num_layers

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], position_embeddings)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = GlmOcrTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._rope_deltas = None
        self._position_ids = None

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int],
        temp_merge_size: int = 1,
    ):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        llm_grid_t = int(grid_thw[0]) // temp_merge_size
        llm_grid_h = int(grid_thw[1]) // spatial_merge_size
        llm_grid_w = int(grid_thw[2]) // spatial_merge_size

        image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w
        position_temporal = np.full((image_seq_length,), start_position, dtype=np.int64)
        position_height = np.repeat(
            np.arange(
                start_position,
                start_position + llm_grid_h,
                dtype=np.int64,
            ),
            llm_grid_w * llm_grid_t,
        )
        position_width = np.tile(
            np.arange(
                start_position,
                start_position + llm_grid_w,
                dtype=np.int64,
            ),
            llm_grid_h * llm_grid_t,
        )
        return np.stack([position_temporal, position_height, position_width], axis=0)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids_np = np.zeros((3, batch_size, seq_length), dtype=np.int64)
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            input_ids_list = input_ids.tolist()
            attention_mask_list = (
                attention_mask.tolist()
                if attention_mask is not None
                and attention_mask.shape[-1] == input_ids.shape[-1]
                else None
            )
            image_grids = image_grid_thw.tolist() if image_grid_thw is not None else []
            video_grids = video_grid_thw.tolist() if video_grid_thw is not None else []
            image_index = 0
            video_index = 0
            for batch_idx, current_input_ids in enumerate(input_ids_list):
                if attention_mask_list is not None:
                    valid_indices = [
                        i
                        for i, m in enumerate(attention_mask_list[batch_idx])
                        if m == 1
                    ]
                    current_input_ids = [current_input_ids[i] for i in valid_indices]
                else:
                    valid_indices = list(range(len(current_input_ids)))

                token_types = []
                for token_id in current_input_ids:
                    if token_id == image_token_id:
                        token_types.append(1)
                    elif token_id == video_token_id:
                        token_types.append(2)
                    else:
                        token_types.append(0)

                groups = []
                if token_types:
                    start = 0
                    current_type = token_types[0]
                    for idx, token_type in enumerate(token_types[1:], start=1):
                        if token_type != current_type:
                            groups.append((current_type, start, idx))
                            start = idx
                            current_type = token_type
                    groups.append((current_type, start, len(token_types)))

                current_pos = 0
                video_group_index = 0
                llm_pos_ids_list: list = []
                for modality_type, start_idx, end_idx in groups:
                    if modality_type == 0:
                        text_len = end_idx - start_idx
                        text_positions = np.arange(text_len, dtype=np.int64)[None, :]
                        llm_pos_ids_list.append(
                            np.broadcast_to(text_positions, (3, text_len)) + current_pos
                        )
                        current_pos += text_len
                        continue

                    if modality_type == 2:
                        grid_thw = video_grids[video_index]
                        video_group_index += 1
                        if video_group_index >= int(grid_thw[0]):
                            video_index += 1
                            video_group_index = 0
                    else:
                        grid_thw = image_grids[image_index]
                        image_index += 1

                    temp_merge_size = int(grid_thw[0])
                    llm_pos_ids_list.append(
                        self.get_vision_position_ids(
                            current_pos,
                            grid_thw,
                            temp_merge_size=temp_merge_size,
                        )
                    )
                    current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // (
                        spatial_merge_size
                    )

                if not llm_pos_ids_list:
                    mrope_position_deltas.append(0)
                    continue

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids_np[:, batch_idx, valid_indices] = llm_positions
                mrope_position_deltas.append(
                    int(llm_positions.max()) + 1 - len(current_input_ids)
                )
            return (
                mx.array(position_ids_np, dtype=input_ids.dtype),
                mx.array(mrope_position_deltas, dtype=input_ids.dtype).reshape(-1, 1),
            )
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.zeros_like(position_ids), position_ids
                )
                max_position_ids = position_ids.max(axis=-1, keepdims=True)
                position_ids = mx.broadcast_to(
                    position_ids[None, :, :], (3, *position_ids.shape)
                )
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if pixel_values is not None:
            self._rope_deltas = None

        # Use ``cache._idx`` — the Python-int token counter — instead of
        # syncing on ``cache[0].offset``. See Qwen2.5-VL for details.
        cache_offset = 0
        cache_offsets = None
        if cache and cache[0] is not None:
            c0 = cache[0]
            cache_offset = c0._idx if hasattr(c0, "_idx") else c0.offset
            if (
                isinstance(c0.offset, mx.array)
                and c0.offset.ndim > 0
                and c0.offset.size > 1
            ):
                cache_offsets = c0.offset

        # Check if mask shape matches input shape (for chunked prefill compatibility)
        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            is_prefill = (
                cache is None
                or cache[0] is None
                or (cache_offsets is None and cache_offset == 0)
            )
            seq_length = inputs.shape[1]
            if (
                self._position_ids is not None
                and cache_offsets is None
                and cache_offset + seq_length <= self._position_ids.shape[-1]
            ):
                position_ids = self._position_ids[
                    :, :, cache_offset : cache_offset + seq_length
                ]
            elif is_prefill or self._rope_deltas is None:
                if self._position_ids is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
                else:
                    position_ids = self._position_ids[:, :, :seq_length]
            else:
                # Use the prev pre-calculated rope-deltas to get the correct position ids
                batch_size, seq_length = inputs.shape
                rope_deltas_src = (
                    rope_deltas_kw if rope_deltas_kw is not None else self._rope_deltas
                )
                if cache_offsets is not None:
                    offsets = cache_offsets[:batch_size]
                    rope_deltas = rope_deltas_src
                    if rope_deltas.shape[0] > batch_size:
                        rope_deltas = rope_deltas[:batch_size]
                    delta = (offsets + rope_deltas.squeeze(-1))[:, None]
                else:
                    delta = mx.array(
                        cache_offset + rope_deltas_src if cache is not None else 0
                    )
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            mask=mask,
        )

        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
