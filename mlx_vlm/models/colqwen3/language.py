from __future__ import annotations

from typing import Optional, Tuple

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
# Rotary embeddings (Qwen3-VL MRoPE)
# -----------------------------------------------------------------------------
class Qwen3VLRotaryEmbedding:
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

    def apply_interleaved_mrope(self, freqs: mx.array, mrope_section) -> mx.array:
        """
        freqs: (3, bs, seq_len, head_dim//2)
        Returns: (bs, seq_len, head_dim//2) with interleaved layout.
        """
        freqs_t = freqs[0]  # overwrite T lane
        for dim, offset in enumerate((1, 2), start=1):  # H, W lanes
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        # position_ids can be (B, T) or (3, B, T)
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        pos_expanded = position_ids[:, :, None, :].astype(mx.float32)  # (3, B, 1, T)

        freqs = inv_freq_expanded @ pos_expanded  # (3, B, dim/2, T)
        freqs = mx.swapaxes(freqs, 2, 3)          # (3, B, T, dim/2)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
) -> Tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads or args.num_attention_heads

        self.head_dim = getattr(args, "head_dim", args.hidden_size // args.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        cache_offset = 0
        if cache is not None:
            # KVCache.offset exists
            off = cache.offset
            if isinstance(off, int):
                cache_offset = off
            elif isinstance(off, mx.array):
                cache_offset = (off if off.ndim == 0 else off[0]).item()
            else:
                cache_offset = int(off)

        kv_seq_len = k.shape[-2] + (cache_offset + 1 if cache is not None else 0)

        if position_ids is None:
            # default position ids
            start = cache_offset
            pos = mx.arange(start, start + L)
            pos = mx.expand_dims(pos, axis=0)      # (1, L)
            pos = mx.tile(pos, (3, 1, 1))          # (3, 1, L)
            position_ids = pos

        cos, sin = self.rotary_emb(v, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., :kv_seq_len]

        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

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
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen3VLDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _deepstack_process(
            self,
            hidden_states: mx.array,
            visual_pos_masks: mx.array,
            visual_embeds: mx.array,
    ):
        """
        hidden_states: [B, T, H]
        visual_pos_masks: [B, T] bool-like
        visual_embeds: [Nvis, H]  (per sample)
        """
        batch_size = hidden_states.shape[0]
        updated_batches = []

        for b in range(batch_size):
            mask_b = visual_pos_masks[b]
            hb = mx.array(hidden_states[b])  # avoid in-place issues

            idx_np = np.where(np.array(mask_b))[0].astype(np.uint32)
            if idx_np.size == 0:
                updated_batches.append(hb)
                continue

            idx_mx = mx.array(idx_np, dtype=mx.uint32)

            ve = visual_embeds
            n_idx = int(idx_np.size)
            n_vis = int(ve.shape[0])

            # ---- align lengths (fixes your 2450 vs 1225 case) ----
            if n_idx != n_vis:
                if n_vis > 0 and (n_idx % n_vis == 0):
                    reps = n_idx // n_vis
                    ve = mx.tile(ve, (reps, 1))  # [n_idx, H]
                else:
                    n = min(n_idx, n_vis)
                    idx_mx = idx_mx[:n]
                    ve = ve[:n]

            hb = hb.at[idx_mx].add(ve)
            updated_batches.append(hb)

        return mx.stack(updated_batches, axis=0)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list[mx.array]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=mask, cache=c, position_ids=position_ids)

            if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                if layer_idx < len(deepstack_visual_embeds):
                    h = self._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])

        return self.norm(h)


# -----------------------------------------------------------------------------
# LanguageModel wrapper (adds RoPE index logic + lm_head)
# -----------------------------------------------------------------------------
class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model = Qwen3VLModel(args)
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    # --- RoPE index helper (kept from your implementation, just cleaned a bit) ---
    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape

        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids

            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)

            position_ids = mx.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype)

            image_index, video_index = 0, 0
            mrope_position_deltas = []

            for i, ids_i in enumerate(total_input_ids):
                ids_i = mx.where(attention_mask[i] == 1, ids_i, mx.zeros_like(ids_i))

                # find vision start indices (works as in original code)
                vision_start_indices = mx.sum(
                    mx.where(
                        ids_i == vision_start_token_id,
                        mx.arange(ids_i.shape[0]),
                        mx.zeros_like(ids_i),
                    )
                )
                vision_tokens = ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()

                input_tokens = ids_i.tolist()
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

        # text-only fallback
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
            position_ids = mx.expand_dims(position_ids[0], axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

            max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, input_ids.shape[0], input_ids.shape[1]))
            mrope_position_deltas = mx.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)

        return position_ids, mrope_position_deltas

    # --- key: hidden-only forward for embedding head usage ---
    def forward_hidden(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list[mx.array]] = None,
        **kwargs,
    ) -> mx.array:
        # prefill slicing
        n_to_process = kwargs.get("n_to_process", None)
        if n_to_process is not None and visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, n_to_process:]

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            off = cache[0].offset
            if isinstance(off, int):
                cache_offset = off
            elif isinstance(off, mx.array):
                cache_offset = (off if off.ndim == 0 else off[0]).item()
            else:
                cache_offset = int(off)

        if position_ids is None and (mask is None or mask.ndim == 2):
            if (
                (cache is not None and cache[0] is not None and cache_offset == 0)
                or self._rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(inputs, image_grid_thw, video_grid_thw, mask)
                self._rope_deltas = rope_deltas
            else:
                batch_size, seq_length = inputs.shape
                delta = mx.array(cache_offset + self._rope_deltas if cache is not None else 0)
                pos = mx.arange(seq_length).reshape(1, -1)
                pos = mx.broadcast_to(pos, (batch_size, seq_length))

                if delta.ndim == 0:
                    delta = mx.expand_dims(delta, axis=0)
                if delta.shape[0] < batch_size:
                    delta = mx.tile(delta, (batch_size, 1))
                else:
                    delta = delta[:batch_size]

                pos = mx.add(pos, delta)[None, ...]
                position_ids = mx.broadcast_to(pos, (3, batch_size, seq_length))

        h = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return h

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list[mx.array]] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        h = self.forward_hidden(
            inputs,
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
        return self.args.num_key_value_heads or self.args.num_attention_heads


