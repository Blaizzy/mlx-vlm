from typing import Optional

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


class Qwen3VLRotaryEmbedding:
    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, rope_scaling=None
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x, position_ids):

        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(
            mx.float32
        )  # shape (3, bs, 1, positions)

        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    Args:
        q (mx.array): The query tensor.
        k (mx.array): The key tensor.
        cos (mx.array): The cosine part of the rotary embedding.
        sin (mx.array): The sine part of the rotary embedding.
        unsqueeze_dim (int, optional): Dimension to unsqueeze. Defaults to 1.
    Returns:
        tuple(mx.array): The rotated query and key tensors.
    """

    cos = mx.expand_dims(cos, axis=unqueeze_dim)
    sin = mx.expand_dims(sin, axis=unqueeze_dim)

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=head_dim, eps=args.rms_norm_eps)

        self.rope_scaling = args.rope_scaling

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            rope_scaling=self.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            kv_seq_len += cache.offset + 1
            position_ids = mx.arange(cache.offset, cache.offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        cos, sin = self.rotary_emb(values, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., :kv_seq_len]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3VLModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        # args for deepstack
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)
        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids)
            # Add deepstack visual embeds
            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                h = self._deepstack_process(
                    h,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        return self.norm(h)

    def _deepstack_process(
        self,
        hidden_states: mx.array,
        visual_pos_masks: mx.array,
        visual_embeds: mx.array,
    ):
        batch_size = hidden_states.shape[0]

        updated_batches = []
        for b in range(batch_size):
            batch_mask = visual_pos_masks[b]
            batch_hidden = hidden_states[b]

            batch_indices = mx.array(np.where(batch_mask)[0], dtype=mx.uint32)

            if len(batch_indices) == 0:
                updated_batches.append(batch_hidden)
                continue

            batch_result = mx.array(batch_hidden)  # avoid modifying in-place
            batch_result = batch_result.at[batch_indices].add(visual_embeds)

            updated_batches.append(batch_result)

        return mx.stack(updated_batches, axis=0)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3VLModel(args)
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        # Calculate RoPE index for image/video tokens
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(
                    attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids)
                )
                image_nums, video_nums = 0, 0
                vision_start_indices = mx.sum(
                    mx.where(
                        input_ids == vision_start_token_id,
                        mx.arange(input_ids.shape[0]),
                        mx.zeros_like(input_ids),
                    )
                )
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(
                        llm_grid_t, 1
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )  # Equivalent to expand()
                    t_index = t_index.flatten()  # Flattens to 1D

                    h_index = mx.arange(llm_grid_h).reshape(
                        1, llm_grid_h, 1
                    )  # Equivalent to .view(1, -1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    h_index = h_index.flatten()  # Flattens to 1D

                    w_index = mx.arange(llm_grid_w).reshape(
                        1, 1, llm_grid_w
                    )  # Equivalent to .view(1, -1)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    w_index = w_index.flatten()  # Flattens to 1D

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(
                        1, text_len
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (3, text_len)
                    )  # Equivalent to expand(3, -1)

                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(
                    -1, keepdims=True
                )[0]
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
        # args for deepstack
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        # Slicing visual_pos_masks when prefilling
        n_to_process = kwargs.get("n_to_process", None)
        if n_to_process is not None:
            visual_pos_masks = visual_pos_masks[:, n_to_process:]

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        # reset rope_deltas when processing a new image/video
        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        if position_ids is None and (mask is None or mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache is not None and cache[0] is not None and (cache_offset == 0))
                or self._rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self._rope_deltas = rope_deltas
            else:
                # Use the prev pre-calculated rope-deltas to get the correct position ids
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)

                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        # Slice delta to match batch
                        delta = delta[:batch_size]

                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
