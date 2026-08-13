from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    kv_sequence_length,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..mlp import SwiGLUMLP
from .config import TextConfig


class CompassRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        output = x_float * mx.rsqrt(
            mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps
        )
        return (output * self.weight.astype(mx.float32)).astype(x.dtype)


class CompassLayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        mean = mx.mean(x_float, axis=-1, keepdims=True)
        variance = mx.mean((x_float - mean) ** 2, axis=-1, keepdims=True)
        output = (x_float - mean) * mx.rsqrt(variance + self.eps)
        return (output * self.weight.astype(mx.float32)).astype(x.dtype)


def _interleaved_position_selector(mrope_section, freq_dim):
    selector = [0] * freq_dim
    for axis, offset in enumerate((1, 2), start=1):
        for index in range(offset, min(mrope_section[axis] * 3, freq_dim), 3):
            selector[index] = axis
    return mx.array(selector, dtype=mx.int32)


class CompassRotaryEmbedding(nn.Module):
    """Compass RoPE, including its repeated-frequency MRoPE layout."""

    def __init__(self, config: TextConfig, layer_type: Optional[str] = None):
        super().__init__()
        self.dim = config.head_dim
        self.rope_style = config.rope_style
        rope_parameters = config.rope_parameters
        uses_per_layer_rope = isinstance(rope_parameters, dict) and any(
            layer in rope_parameters for layer in config.layer_types
        )
        if uses_per_layer_rope:
            rope_parameters = rope_parameters.get(layer_type)
            self.enabled = rope_parameters is not None
        else:
            self.enabled = config.rope_on_all_layers or layer_type in (
                None,
                "sliding_attention",
            )
        rope_parameters = rope_parameters or {}
        rope_theta = rope_parameters.get(
            "rope_theta",
            (
                config.swa_rope_theta
                if layer_type == "sliding_attention"
                else config.rope_theta
            ),
        )
        self._inv_freq = 1.0 / (
            rope_theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        section = rope_parameters.get("mrope_section")
        if (
            isinstance(section, (list, tuple))
            and len(section) == 3
            and all(isinstance(value, int) and value >= 0 for value in section)
            and sum(section) == self._inv_freq.shape[0]
        ):
            self.mrope_section = list(section)
            self._position_selector = _interleaved_position_selector(
                self.mrope_section, self._inv_freq.shape[0]
            )
        else:
            self.mrope_section = None
            self._position_selector = None

    @property
    def inv_freq(self):
        return self._inv_freq

    @property
    def position_selector(self):
        return self._position_selector

    def __call__(self, x: mx.array, position_ids: mx.array):
        if position_ids.ndim == 3 and self.position_selector is not None:
            positions = mx.take(position_ids, self.position_selector, axis=0).transpose(
                1, 2, 0
            )
            freqs = positions.astype(mx.float32) * self.inv_freq
        else:
            freqs = position_ids.astype(mx.float32)[..., None] * self.inv_freq

        if self.rope_style == "interleave":
            emb = mx.repeat(freqs, 2, axis=-1)
        else:
            emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def make_norm(config: TextConfig):
    if config.norm_type == "rms_norm":
        return CompassRMSNorm(config.hidden_size, config.rms_norm_eps or 1e-6)
    return CompassLayerNorm(config.hidden_size, config.layer_norm_eps)


def _rowwise_batch(module, x: mx.array) -> mx.array:
    """Keep each row's numerics independent of the batch size."""
    if x.ndim >= 3 and x.shape[0] > 1:
        return mx.concatenate([module(x[row : row + 1]) for row in range(x.shape[0])])
    return module(x)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        self.rope_style = config.rope_style

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = CompassRotaryEmbedding(config, self.layer_type)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        position_embeddings=None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        queries = _rowwise_batch(self.q_proj, hidden_states).reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        )
        keys = _rowwise_batch(self.k_proj, hidden_states).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        values = _rowwise_batch(self.v_proj, hidden_states).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rotary_emb.enabled:
            if position_embeddings is None:
                if position_ids is None:
                    position_ids = mx.arange(sequence_length, dtype=mx.int32)[None, :]
                    if self.rotary_emb.position_selector is not None:
                        position_ids = mx.broadcast_to(
                            position_ids[None, ...],
                            (3, batch_size, sequence_length),
                        )
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
            queries, keys = self._apply_precomputed(queries, keys, position_embeddings)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if isinstance(mask, mx.array):
            key_length = kv_sequence_length(keys)
            if mask.shape[-1] != key_length:
                mask = mask[..., -key_length:]

        left_padding = getattr(cache, "left_padding", None)
        padding_rows = left_padding.tolist() if left_padding is not None else []
        if (
            isinstance(keys, mx.array)
            and left_padding is not None
            and (batch_size > 1 or any(int(padding) > 0 for padding in padding_rows))
        ):
            # Batch caches keep rows at a common physical length.  Compact
            # each row to the same logical K/V range used by a single-row
            # cache, then run the identical fused attention kernel.
            offsets = cache.offset.tolist()
            outputs = []
            for row, padding in enumerate(padding_rows):
                padding = int(padding)
                initial_prefill = sequence_length == keys.shape[-2]
                if padding > 0:
                    start = padding
                    end = start + int(offsets[row])
                    row_keys = keys[row : row + 1, ..., start:end, :]
                    row_values = values[row : row + 1, ..., start:end, :]
                    query_start = start if initial_prefill else 0
                    row_queries = queries[row : row + 1, ..., query_start:, :]
                    if sequence_length == 1:
                        row_mask = None
                    elif isinstance(mask, mx.array):
                        row_mask = mask[row : row + 1]
                        row_mask = row_mask[..., query_start:, start:end]
                    else:
                        row_mask = mask
                else:
                    query_start = 0
                    row_queries = queries[row : row + 1]
                    row_keys = keys[row : row + 1]
                    row_values = values[row : row + 1]
                    row_mask = (
                        mask[row : row + 1] if isinstance(mask, mx.array) else mask
                    )
                outputs.append(
                    scaled_dot_product_attention(
                        row_queries,
                        row_keys,
                        row_values,
                        cache=None,
                        scale=self.scale,
                        mask=row_mask,
                    )
                )
                if query_start:
                    outputs[-1] = mx.pad(
                        outputs[-1],
                        [(0, 0), (0, 0), (query_start, 0), (0, 0)],
                    )
            output = mx.concatenate(outputs, axis=0)
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return _rowwise_batch(self.o_proj, output)

    def _apply_precomputed(self, q, k, position_embeddings):
        cos, sin = position_embeddings
        return _apply_rope(q, k, cos, sin, self.rope_style)


def _rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return mx.stack([-x2, x1], axis=-1).flatten(-2)


def _rotate_half_split(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(q, k, cos, sin, rope_style):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    rotate_half = (
        _rotate_half_interleaved if rope_style == "interleave" else _rotate_half_split
    )
    return (
        (q * cos + rotate_half(q) * sin).astype(q.dtype),
        (k * cos + rotate_half(k) * sin).astype(k.dtype),
    )


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.input_layernorm = make_norm(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SwiGLUMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        if config.transformer_block_type != "parallel":
            self.post_attention_layernorm = make_norm(config)

    def __call__(self, x, mask, cache, position_ids, position_embeddings):
        residual = x
        h = self.input_layernorm(x)
        attention = self.self_attn(
            h,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        if hasattr(self, "post_attention_layernorm"):
            h = residual + attention
            return h + _rowwise_batch(self.mlp, self.post_attention_layernorm(h))
        return residual + attention + _rowwise_batch(self.mlp, h)


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        final_eps = config.rms_norm_eps or config.layer_norm_eps
        self.norm = (
            CompassRMSNorm(config.hidden_size, final_eps)
            if config.rms_norm_eps is not None
            else CompassLayerNorm(config.hidden_size, config.layer_norm_eps)
        )
        layer_types = list(dict.fromkeys(config.layer_types))
        self.rotary_embeddings = {
            layer_type: CompassRotaryEmbedding(config, layer_type)
            for layer_type in layer_types
        }
        self.rotary_emb = next(
            (rotary for rotary in self.rotary_embeddings.values() if rotary.enabled),
            next(iter(self.rotary_embeddings.values())),
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds=None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)

        full_indices = [
            i
            for i, layer_type in enumerate(self.config.layer_types)
            if layer_type == "full_attention"
        ]
        sliding_indices = [
            i
            for i, layer_type in enumerate(self.config.layer_types)
            if layer_type == "sliding_attention"
        ]
        global_cache = cache[full_indices[0]] if full_indices else None
        sliding_cache = cache[sliding_indices[0]] if sliding_indices else None
        global_mask = sliding_mask = mask
        if mask is None:
            global_mask = create_attention_mask(h, global_cache)
            sliding_mask = create_attention_mask(
                h, sliding_cache, window_size=self.config.sliding_window
            )

        if position_ids is None:
            position_ids = mx.arange(h.shape[1], dtype=mx.int32)[None, :]
            if self.rotary_emb.position_selector is not None:
                position_ids = mx.broadcast_to(
                    position_ids[None, ...], (3, h.shape[0], h.shape[1])
                )
        position_embeddings = {
            layer_type: (rotary(h, position_ids) if rotary.enabled else None)
            for layer_type, rotary in self.rotary_embeddings.items()
        }

        for layer_idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            layer_mask = (
                global_mask
                if layer.attention_type == "full_attention"
                else sliding_mask
            )
            h = layer(
                h,
                layer_mask,
                layer_cache,
                position_ids,
                position_embeddings[layer.attention_type],
            )
            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                h = self._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
        return self.norm(h)

    @staticmethod
    def _deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
        if visual_pos_masks is None:
            return hidden_states
        flat_mask = visual_pos_masks.reshape(-1)
        indices = mx.array(np.where(flat_mask)[0], dtype=mx.uint32)
        if not indices.shape[0]:
            return hidden_states
        if visual_pos_masks.dtype == mx.bool_:
            selected_visual_embeds = visual_embeds[: indices.shape[0]]
        else:
            visual_indices = flat_mask[indices].astype(mx.uint32) - 1
            selected_visual_embeds = visual_embeds[visual_indices]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_hidden = flat_hidden.at[indices].add(selected_visual_embeds)
        return flat_hidden.reshape(hidden_states.shape)


class LanguageModel(nn.Module):
    supports_logits_to_keep = True

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._rope_deltas = None
        self._position_ids = None

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        is_mrope = self.model.rotary_emb.position_selector is not None
        if not is_mrope:
            if attention_mask is None:
                positions = mx.arange(input_ids.shape[1], dtype=mx.int32)[None, :]
                positions = mx.broadcast_to(positions, input_ids.shape)
                deltas = mx.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
            else:
                positions = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
                positions = mx.where(
                    attention_mask == 0, mx.ones_like(positions), positions
                )
                valid_lengths = attention_mask.astype(input_ids.dtype).sum(
                    axis=-1, keepdims=True
                )
                deltas = positions.max(axis=-1, keepdims=True) + 1 - valid_lengths
            return positions, deltas

        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)
        position_rows = []
        deltas = []
        image_index = 0
        spatial_merge_size = self.config_for_vision.vision_config.spatial_merge_size
        image_token_id = self.config_for_vision.image_token_id
        vision_start_token_id = self.config_for_vision.vision_start_token_id

        for row_ids, row_mask in zip(input_ids.tolist(), attention_mask.tolist()):
            valid_indices = [i for i, keep in enumerate(row_mask) if keep]
            tokens = [row_ids[i] for i in valid_indices]
            vision_tokens = [
                tokens[i + 1]
                for i, token in enumerate(tokens[:-1])
                if token == vision_start_token_id
            ]
            image_count = sum(token == image_token_id for token in vision_tokens)
            pieces = []
            start = 0
            for _ in range(image_count):
                end = tokens.index(image_token_id, start)
                t, h, w = [int(v) for v in image_grid_thw[image_index].tolist()]
                image_index += 1
                text_len = end - start
                start_index = pieces[-1].max().item() + 1 if pieces else 0
                pieces.append(
                    mx.broadcast_to(
                        mx.arange(text_len, dtype=mx.int32)[None, :], (3, text_len)
                    )
                    + start_index
                )
                grid_h = h // spatial_merge_size
                grid_w = w // spatial_merge_size
                t_index = mx.broadcast_to(
                    mx.arange(t, dtype=mx.int32)[:, None], (t, grid_h * grid_w)
                ).reshape(-1)
                h_index = mx.broadcast_to(
                    mx.arange(grid_h, dtype=mx.int32)[None, :, None],
                    (t, grid_h, grid_w),
                ).reshape(-1)
                w_index = mx.broadcast_to(
                    mx.arange(grid_w, dtype=mx.int32)[None, None, :],
                    (t, grid_h, grid_w),
                ).reshape(-1)
                pieces.append(
                    mx.stack([t_index, h_index, w_index]) + text_len + start_index
                )
                start = end + t * grid_h * grid_w

            if start < len(tokens):
                start_index = pieces[-1].max().item() + 1 if pieces else 0
                text_len = len(tokens) - start
                pieces.append(
                    mx.broadcast_to(
                        mx.arange(text_len, dtype=mx.int32)[None, :], (3, text_len)
                    )
                    + start_index
                )
            compact = mx.concatenate(pieces, axis=1)
            padded = [[1] * input_ids.shape[1] for _ in range(3)]
            for compact_index, original_index in enumerate(valid_indices):
                for axis in range(3):
                    padded[axis][original_index] = int(
                        compact[axis, compact_index].item()
                    )
            position_rows.append(mx.array(padded, dtype=mx.int32))
            deltas.append(int(compact.max().item()) + 1 - len(tokens))

        return (
            mx.stack(position_rows, axis=1),
            mx.array(deltas, dtype=input_ids.dtype)[:, None],
        )

    @property
    def config_for_vision(self):
        # Set by the multimodal wrapper after construction.
        return self._vision_config

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        pixel_values = kwargs.pop("pixel_values", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        attention_mask = kwargs.pop("attention_mask", None)
        logits_to_keep = kwargs.pop("logits_to_keep", None)
        n_to_process = kwargs.pop("n_to_process", None)

        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None
        if rope_deltas is not None:
            self._rope_deltas = rope_deltas

        if (
            cache
            and isinstance(attention_mask, mx.array)
            and hasattr(cache[0], "left_padding")
            and cache[0].empty()
        ):
            left_padding = []
            for row in attention_mask.tolist():
                try:
                    left_padding.append(row.index(1))
                except ValueError:
                    left_padding.append(len(row))
            if any(left_padding):
                for layer_cache in cache:
                    prepare = getattr(layer_cache, "prepare", None)
                    if callable(prepare):
                        prepare(left_padding=left_padding)

        cache_offset = 0
        position_offset = 0
        if cache and cache[0] is not None:
            first_cache = cache[0]
            # Keep the shared physical timeline for slicing full-prompt
            # metadata, but use each row's logical token count for RoPE.
            # Dynamic batching may extend a cache with shorter rows, whose
            # added left padding must not advance their positions.
            cache_offset = getattr(first_cache, "_offset", None)
            if cache_offset is None and hasattr(first_cache, "left_padding"):
                cache_offset = getattr(first_cache, "_idx", None)
            if cache_offset is None:
                cache_offset = first_cache.offset
            if isinstance(cache_offset, mx.array):
                cache_offset = int(cache_offset.max().item())
            else:
                cache_offset = int(cache_offset)
            position_offset = getattr(first_cache, "offset", cache_offset)
            if isinstance(position_offset, mx.array):
                position_offset = position_offset.reshape(-1, 1)
            else:
                position_offset = int(position_offset)

        if (
            visual_pos_masks is not None
            and visual_pos_masks.shape[-1] != inputs.shape[-1]
        ):
            # ``generate_step`` forwards full-prompt multimodal metadata to
            # every chunk.  Align both the mask and its packed visual rows to
            # the current cache window before deep-stack injection.
            start = cache_offset
            stop = start + inputs.shape[1]
            full_visual_pos_masks = visual_pos_masks
            visual_pos_masks = full_visual_pos_masks[:, start:stop]
            if (
                deepstack_visual_embeds is not None
                and full_visual_pos_masks.dtype == mx.bool_
            ):
                # Legacy boolean masks carry no visual-row indices, so their
                # packed embeddings still need to be windowed explicitly.
                embed_windows = [[] for _ in deepstack_visual_embeds]
                row_offset = 0
                for row, row_mask in enumerate(full_visual_pos_masks):
                    row_visuals = int(row_mask.sum().item())
                    before = int(row_mask[:start].sum().item())
                    count = int(visual_pos_masks[row].sum().item())
                    for layer, embeds in enumerate(deepstack_visual_embeds):
                        embed_windows[layer].append(
                            embeds[row_offset + before : row_offset + before + count]
                        )
                    row_offset += row_visuals
                deepstack_visual_embeds = [
                    (
                        mx.concatenate(windows, axis=0)
                        if windows
                        else mx.zeros((0, self.config.hidden_size))
                    )
                    for windows in embed_windows
                ]

        if position_ids is not None and position_ids.shape[-1] > inputs.shape[-1]:
            position_ids = position_ids[
                ..., cache_offset : cache_offset + inputs.shape[-1]
            ]

        if position_ids is None:
            if self._rope_deltas is None or cache_offset == 0 or cache is None:
                position_ids, self._rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, mask
                )
                self._position_ids = position_ids
            else:
                delta = position_offset + self._rope_deltas
                position_ids = mx.arange(inputs.shape[1], dtype=mx.int32)[None, :]
                position_ids = mx.broadcast_to(position_ids, inputs.shape) + delta
                if self.model.rotary_emb.position_selector is not None:
                    position_ids = mx.broadcast_to(
                        position_ids[None, ...], (3, inputs.shape[0], inputs.shape[1])
                    )

        hidden_states = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        batch_cache = bool(cache) and hasattr(cache[0], "left_padding")
        batch_last_token_is_real = attention_mask is None or bool(
            mx.all(attention_mask[:, -1]).item()
        )
        if (
            logits_to_keep
            or n_to_process is not None
            or (batch_cache and batch_last_token_is_real)
        ):
            hidden_states = hidden_states[:, -int(logits_to_keep or 1) :, :]
        logits = _rowwise_batch(
            (
                self.model.embed_tokens.as_linear
                if self.config.tie_word_embeddings
                else self.lm_head
            ),
            hidden_states,
        )
        if self.config.logit_scale is not None:
            logits = logits * self.config.logit_scale
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        return weights
