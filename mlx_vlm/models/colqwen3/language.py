from __future__ import annotations

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


# -----------------------------------------------------------------------------
# RoPE (Qwen3-VL MRoPE)
# -----------------------------------------------------------------------------
class Qwen3VLRotaryEmbedding:
    """
    Qwen3-VL uses "interleaved MRoPE" across (T, H, W) dimensions.
    We follow the original mlx-vlm Qwen3-VL implementation style.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

        rope_scaling = rope_scaling or {}
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    @staticmethod
    def apply_interleaved_mrope(freqs: mx.array, mrope_section: list[int]) -> mx.array:
        """
        Input freqs: (3, bs, seq_len, head_dim//2) -> Output: (bs, seq_len, head_dim//2)
        Reorganizes chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...].
        """
        freqs_t = freqs[0]  # overwrite first dimension with interleaved content
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x: mx.array, position_ids: mx.array):
        # position_ids can be (B, L) or (3, B, L)
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)  # (3, bs, seq, head_dim//2)

        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)  # (bs, seq, head_dim//2)

        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# Decoder blocks
# -----------------------------------------------------------------------------
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

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[-2]
        if cache is not None and position_ids is None:
            kv_seq_len += cache.offset + 1
            pos = mx.arange(cache.offset, cache.offset + L)
            pos = mx.expand_dims(pos, axis=0)
            position_ids = mx.tile(pos, (3, 1, 1))
        elif cache is not None and position_ids is not None:
            kv_seq_len += cache.offset + 1

        cos, sin = self.rotary_emb(values, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., :kv_seq_len]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache, position_ids=position_ids)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3VLModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen3VLDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        # If mask is None, create standard causal mask (mlx_vlm style).
        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=mask, cache=c, position_ids=position_ids)

            # Deepstack: add visual features to hidden states in early layers
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                h = self._deepstack_process(
                    hidden_states=h,
                    visual_pos_masks=visual_pos_masks,
                    visual_embeds=deepstack_visual_embeds[layer_idx],
                )

        return self.norm(h)

    @staticmethod
    def _deepstack_process(
        hidden_states: mx.array,
        visual_pos_masks: mx.array,
        visual_embeds: mx.array,
    ) -> mx.array:
        """
        Adds visual_embeds to hidden_states at positions where visual_pos_masks is True.
        """
        if visual_pos_masks is None:
            return hidden_states

        batch_size = hidden_states.shape[0]
        updated = []

        for b in range(batch_size):
            mask_b = visual_pos_masks[b]
            h_b = hidden_states[b]

            idx = mx.array(np.where(np.array(mask_b))[0], dtype=mx.uint32)
            if len(idx) == 0:
                updated.append(h_b)
                continue

            h_new = mx.array(h_b)  # avoid in-place side effects
            h_new = h_new.at[idx].add(visual_embeds)
            updated.append(h_new)

        return mx.stack(updated, axis=0)


# -----------------------------------------------------------------------------
# Public LanguageModel wrapper
# -----------------------------------------------------------------------------
class LanguageModel(nn.Module):
    """
    Wrapper that provides:
      - forward_hidden(): returns hidden states [B, T, H]
      - __call__(): returns logits (LanguageModelOutput)
    """

    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model = Qwen3VLModel(args)
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    # -----------------------------
    # RoPE index logic
    # -----------------------------
    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        """
        This is the same logic you already had. Kept for correctness with Qwen3-VL.
        Returns:
          position_ids: (3, B, L)
          mrope_position_deltas: (B,) or scalar-ish
        """
        batch_size, seq_length = input_ids.shape

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)

        # default container
        position_ids = mx.ones((3, batch_size, seq_length), dtype=input_ids.dtype)

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids

            image_index, video_index = 0, 0
            for i, ids in enumerate(total_input_ids):
                ids = mx.where(attention_mask[i] == 1, ids, mx.zeros_like(ids))

                vision_start_indices = mx.sum(
                    mx.where(
                        ids == vision_start_token_id,
                        mx.arange(ids.shape[0]),
                        mx.zeros_like(ids),
                    )
                )
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()

                input_tokens = ids.tolist()
                llm_pos_ids_list = []
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
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size

                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)

                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w)).flatten()

                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )

                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))
                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)

                mask_i = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask_i, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask_i.shape[0]))

                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(expanded_mask, expanded_positions, position_ids[:, i : i + 1, :])

                position_ids = mx.concatenate(
                    [position_ids[:, :i, :], new_positions, position_ids[:, i + 1 :, :]],
                    axis=1,
                )

                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas

        # no image/video grids
        if attention_mask is not None:
            pid = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            pid = mx.where(attention_mask == 0, mx.ones_like(pid), pid)
            pid = mx.expand_dims(pid[0], axis=0)
            pid = mx.tile(pid, (3, 1, 1))

            max_pid = pid.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
            mrope_position_deltas = max_pid + 1 - attention_mask.shape[-1]
            return pid, mrope_position_deltas

        pid = mx.arange(seq_length).reshape(1, -1)
        pid = mx.broadcast_to(pid, (3, batch_size, seq_length))
        mrope_position_deltas = mx.zeros([batch_size, 1], dtype=input_ids.dtype)
        return pid, mrope_position_deltas

    # -----------------------------
    # Hidden forward (used by embeddings)
    # -----------------------------
    def forward_hidden(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Returns hidden states [B, T, H].
        IMPORTANT FIX: mask is passed into the underlying model.
        """

        # Slicing visual_pos_masks when prefilling
        n_to_process = kwargs.get("n_to_process", None)
        if n_to_process is not None and visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, n_to_process:]

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        # reset rope deltas when a new image/video comes
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

        # if position_ids not given, compute with Qwen3-VL logic
        if position_ids is None and (mask is None or mask.ndim == 2):
            if (
                (cache is not None and cache[0] is not None and (cache_offset == 0))
                or self._rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, mask
                )
                self._rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = mx.array(cache_offset + self._rope_deltas if cache is not None else 0)

                pid = mx.arange(seq_length).reshape(1, -1)
                pid = mx.broadcast_to(pid, (batch_size, seq_length))

                if delta.ndim == 0:
                    delta = mx.expand_dims(delta, axis=0)
                if delta.shape[0] < batch_size:
                    delta = mx.tile(delta, (batch_size, 1))
                else:
                    delta = delta[:batch_size]

                pid = mx.add(pid, delta)[None, ...]
                position_ids = mx.broadcast_to(pid, (3, batch_size, seq_length))

        h = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,  # <-- CRITICAL: pass mask through
            cache=cache,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return h

    # -----------------------------
    # Logits forward (used by generate)
    # -----------------------------
    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        # Minimal: delegate everything to forward_hidden
        h = self.forward_hidden(
            input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)

        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads