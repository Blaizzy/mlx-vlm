from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig


class Qwen3VLMoeTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=-1)

        topk_indices = mx.argpartition(routing_weights, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        topk_weights = mx.take_along_axis(routing_weights, topk_indices, axis=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / mx.sum(topk_weights, axis=-1, keepdims=True)

        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_indices = topk_indices.astype(mx.int32)

        topk_indices = topk_indices.reshape(batch_size, seq_len, self.top_k)
        topk_weights = topk_weights.reshape(batch_size, seq_len, self.top_k)

        expert_outputs = self.switch_mlp(hidden_states, topk_indices)
        expert_outputs = expert_outputs * mx.expand_dims(topk_weights, axis=-1)

        return mx.sum(expert_outputs, axis=-2)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLMoeRotaryEmbedding:
    def __init__(self, config: TextConfig):
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = mx.array(freqs[0])
        for dim in [1, 2]:
            offset = dim
            length = mrope_section[dim] * 3
            indices = list(range(offset, min(length, freqs.shape[-1]), 3))
            for idx in indices:
                freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.broadcast_to(
                position_ids, (3, position_ids.shape[1], position_ids.shape[2])
            )

        inv_freq_expanded = mx.expand_dims(
            mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0), axis=-1
        )
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded, (3, position_ids.shape[1], self.inv_freq.shape[0], 1)
        )

        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), axis=2)

        freqs = mx.matmul(
            inv_freq_expanded.astype(mx.float32),
            position_ids_expanded,
        ).transpose(0, 1, 3, 2)

        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate((freqs, freqs), axis=-1)

        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3VLMoeTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLMoeTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx)

        use_moe = (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if use_moe:
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = MLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3VLMoeTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3VLMoeTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:

        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x, mask=mask, cache=cache, position_embeddings=position_embeddings
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3VLMoeTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3VLMoeTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLMoeRotaryEmbedding(config=config)

    def _deepstack_process(
        self,
        hidden_states: mx.array,
        visual_pos_masks: mx.array,
        visual_embeds: mx.array,
    ):
        visual_embeds = visual_embeds.astype(hidden_states.dtype)
        mask = mx.expand_dims(visual_pos_masks, axis=-1)
        return mx.where(mask, hidden_states + visual_embeds, hidden_states)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if position_ids is not None:
            position_embeddings = self.rotary_emb(h, position_ids)
        else:
            position_embeddings = None

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_embeddings)

            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                h = self._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.model_type = config.model_type
        self.model = Qwen3VLMoeTextModel(config)
        self.rope_deltas = None

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        spatial_merge_size = self.model_config.vision_config.spatial_merge_size
        image_token_id = self.model_config.image_token_id
        video_token_id = self.model_config.video_token_id
        vision_start_token_id = self.model_config.vision_start_token_id

        if video_grid_thw is not None:
            video_grid_thw_list = []
            for thw in video_grid_thw:
                t = int(thw[0].item())
                for _ in range(t):
                    video_grid_thw_list.append(
                        mx.array([1, thw[1].item(), thw[2].item()])
                    )
            if video_grid_thw_list:
                video_grid_thw = mx.stack(video_grid_thw_list)

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

            for i, input_ids_row in enumerate(total_input_ids):
                input_ids_row = mx.where(
                    attention_mask[i] == 1, input_ids_row, mx.zeros_like(input_ids_row)
                )

                vision_matches = input_ids_row == vision_start_token_id
                vision_start_idx = mx.sum(
                    mx.where(vision_matches, mx.arange(input_ids_row.shape[0]), 0)
                )

                has_vision = mx.sum(input_ids_row == vision_start_token_id).item() > 0

                if has_vision:
                    vision_token = input_ids_row[vision_start_idx + 1]
                    image_nums = (vision_token == image_token_id).item()
                    video_nums = (vision_token == video_token_id).item()
                else:
                    image_nums = video_nums = 0

                input_tokens = input_ids_row.tolist()
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

                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size

                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max().item() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )

                    llm_pos_ids_list.append(
                        mx.arange(text_len).reshape(1, -1).astype(mx.int32) + st_idx
                    )
                    llm_pos_ids_list[-1] = mx.broadcast_to(
                        llm_pos_ids_list[-1], (3, text_len)
                    )

                    t_index = mx.zeros(
                        (llm_grid_t * llm_grid_h * llm_grid_w,), dtype=mx.int32
                    )
                    h_index = mx.arange(llm_grid_h).reshape(1, -1, 1).astype(mx.int32)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    ).reshape(-1)

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, -1).astype(mx.int32)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    ).reshape(-1)

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max().item() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    remaining = (
                        mx.arange(text_len).reshape(1, -1).astype(mx.int32) + st_idx
                    )
                    llm_pos_ids_list.append(mx.broadcast_to(remaining, (3, text_len)))

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask_expanded = mx.expand_dims(attention_mask[i] == 1, axis=0)
                mask_expanded = mx.broadcast_to(
                    mask_expanded, (3, mask_expanded.shape[1])
                )

                position_ids_slice = mx.where(
                    mask_expanded,
                    llm_positions,
                    position_ids[:, i, :],
                )
                position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        mx.expand_dims(position_ids_slice, axis=1),
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                mrope_position_deltas.append(
                    llm_positions.max().item() + 1 - len(total_input_ids[i])
                )

            mrope_position_deltas = mx.array(mrope_position_deltas)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = mx.max(
                    mx.max(position_ids, axis=0), axis=-1, keepdims=True
                )
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1], dtype=input_ids.dtype
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
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)

        if pixel_values is not None:
            self.rope_deltas = None

        if position_ids is None and (mask is None or mask.ndim == 2):
            if (
                (cache is not None and cache[0] is not None and cache[0].offset == 0)
                or self.rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = inputs.shape
                delta = cache[-1].offset + self.rope_deltas if cache is not None else 0
                delta = mx.expand_dims(delta, axis=0)
                position_ids = mx.arange(seq_length).reshape(1, seq_length)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                if cache is not None:
                    delta = mx.repeat(delta, batch_size // delta.shape[0], axis=0)
                position_ids = position_ids + delta
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

        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
