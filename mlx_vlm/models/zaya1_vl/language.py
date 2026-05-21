import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchLinear

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from .config import ModelConfig, TextConfig


def _split_cache(cache):
    if cache is None:
        return None, None
    if isinstance(cache, CacheList):
        return cache[0], cache[1]
    if isinstance(cache, (tuple, list)) and len(cache) == 2:
        return cache[0], cache[1]
    return cache, None


def _cache_is_empty(cache) -> bool:
    return cache is None or (hasattr(cache, "empty") and cache.empty())


def _cache_position_mask(hidden_states: mx.array, cache) -> Optional[mx.array]:
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is None:
        return None

    size = cache.size() if hasattr(cache, "size") else 0
    positions = mx.arange(hidden_states.shape[1]) + size
    return positions[None, :] >= left_padding[:, None]


def _causal_conv1d_stack(
    conv_layers, x: mx.array, state, state_size: int, use_state: bool
):
    if use_state:
        if (
            state is None
            or state.shape[0] != x.shape[0]
            or state.shape[1] != state_size
        ):
            state = mx.zeros((x.shape[0], state_size, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        state_source = conv_input
    else:
        conv_input = mx.pad(x, ((0, 0), (state_size, 0), (0, 0)))
        state_source = x

    y = conv_input
    for conv in conv_layers:
        y = conv(y)

    if state_size == 0:
        next_state = mx.zeros((x.shape[0], 0, x.shape[-1]), dtype=x.dtype)
    else:
        if state_source.shape[1] < state_size:
            state_source = mx.pad(
                state_source,
                ((0, 0), (state_size - state_source.shape[1], 0), (0, 0)),
            )
        next_state = mx.contiguous(state_source[:, -state_size:, :])

    return y, next_state


class ResidualScaling(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones((config.hidden_size,))
        self.hidden_states_bias = mx.zeros((config.hidden_size,))
        if self.not_first_layer:
            self.residual_scale = mx.ones((config.hidden_size,))
            self.residual_bias = mx.zeros((config.hidden_size,))

    def __call__(self, residual: Optional[mx.array], hidden_states: mx.array):
        hidden_states = (
            hidden_states + self.hidden_states_bias
        ) * self.hidden_states_scale
        if self.not_first_layer and residual is not None:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


class CCA(nn.Module):
    def __init__(self, config: TextConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size
        self.cca_time0 = config.cca_time0
        self.cca_time1 = config.cca_time1
        self.total_padding = self.cca_time0 + self.cca_time1 - 2

        self.num_kv_heads = config.num_key_value_heads
        self.num_q_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.latent_k_dim = self.num_kv_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.gqa_groups = self.num_q_heads // self.num_kv_heads
        self.sqrt_head_dim = math.sqrt(self.head_dim)

        self.linear_q = nn.Linear(
            self.hidden_size, self.latent_q_dim, bias=config.attention_bias
        )
        self.linear_k = nn.Linear(
            self.hidden_size, self.latent_k_dim, bias=config.attention_bias
        )
        self.val_proj1 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=config.attention_bias
        )
        self.val_proj2 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=config.attention_bias
        )

        if config.vision_lora:
            r = config.vision_lora_rank_attn
            self.lora_linear_q = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_q_dim, bias=False),
            ]
            self.lora_linear_k = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim, bias=False),
            ]
            self.lora_val_proj1 = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim // 2, bias=False),
            ]
            self.lora_val_proj2 = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim // 2, bias=False),
            ]

        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.conv_qk = [
            nn.Conv1d(
                in_out_ch,
                in_out_ch,
                kernel_size=self.cca_time0,
                groups=in_out_ch,
            ),
            nn.Conv1d(
                in_out_ch,
                in_out_ch,
                kernel_size=self.cca_time1,
                groups=self.num_kv_heads + self.num_q_heads,
            ),
        ]
        self.temp = mx.zeros((self.num_kv_heads,))

    @staticmethod
    def _apply_lora(layers, x):
        return layers[1](layers[0](x))

    def _conv(self, qk_packed0: mx.array, aux_cache, kv_cache):
        x = qk_packed0.transpose(1, 0, 2)
        state = aux_cache[0] if aux_cache is not None else None
        y, state = _causal_conv1d_stack(
            self.conv_qk,
            x,
            state,
            self.total_padding,
            use_state=aux_cache is not None and not _cache_is_empty(kv_cache),
        )
        if aux_cache is not None:
            aux_cache[0] = state
        return y.transpose(1, 0, 2)

    def __call__(
        self,
        hidden_states: mx.array,
        cache=None,
        cca_mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        kv_cache, aux_cache = _split_cache(cache)

        if cca_mask is not None and hidden_states.shape[1] > 1:
            hidden_states = hidden_states * cca_mask[..., None].astype(
                hidden_states.dtype
            )

        hs = hidden_states.transpose(1, 0, 2)
        if hs.shape[0] > 1:
            hs_d = mx.concatenate([mx.zeros_like(hs[:1]), hs[:-1]], axis=0)
        else:
            hs_d = mx.zeros_like(hs)

        q = self.linear_q(hs)
        k = self.linear_k(hs)
        lora_mask = None
        if self.config.vision_lora and image_mask is not None:
            lora_mask = image_mask.transpose(1, 0)[..., None].astype(q.dtype)
            q = q + self._apply_lora(self.lora_linear_q, hs) * lora_mask
            k = k + self._apply_lora(self.lora_linear_k, hs) * lora_mask

        query_pre = q.reshape(*q.shape[:2], self.num_q_heads, self.head_dim)
        key_pre = k.reshape(*k.shape[:2], self.num_kv_heads, self.head_dim)
        key_pre = mx.repeat(key_pre, self.gqa_groups, axis=2)
        qk_mean_q = (query_pre + key_pre) / 2
        qk_mean_k = qk_mean_q.reshape(
            *qk_mean_q.shape[:2], self.num_kv_heads, self.gqa_groups, self.head_dim
        ).mean(axis=3)

        qk_packed0 = mx.concatenate([q, k], axis=-1)
        qk_packed3 = self._conv(qk_packed0, aux_cache, kv_cache)

        query = (
            qk_packed3[..., : self.latent_q_dim].reshape(
                *qk_packed3.shape[:2], self.num_q_heads, self.head_dim
            )
            + qk_mean_q
        )
        key = (
            qk_packed3[..., self.latent_q_dim :].reshape(
                *qk_packed3.shape[:2], self.num_kv_heads, self.head_dim
            )
            + qk_mean_k
        )

        v1 = self.val_proj1(hs)
        if self.config.vision_lora and image_mask is not None:
            v1 = v1 + self._apply_lora(self.lora_val_proj1, hs) * lora_mask

        if (
            aux_cache is not None
            and not _cache_is_empty(kv_cache)
            and aux_cache[1] is not None
        ):
            hs_d = mx.concatenate([aux_cache[1][None, ...], hs[:-1]], axis=0)
        if aux_cache is not None:
            aux_cache[1] = hs[-1]

        v2 = self.val_proj2(hs_d)
        if self.config.vision_lora and image_mask is not None:
            v2 = v2 + self._apply_lora(self.lora_val_proj2, hs_d) * lora_mask

        value = mx.concatenate([v1, v2], axis=-1).reshape(
            *hs.shape[:2], self.num_kv_heads, self.head_dim
        )

        norm_eps = mx.finfo(query.dtype).eps
        query_norm = mx.maximum(
            mx.sqrt(mx.sum(query * query, axis=-1, keepdims=True)), norm_eps
        )
        key_norm = mx.maximum(
            mx.sqrt(mx.sum(key * key, axis=-1, keepdims=True)), norm_eps
        )
        query = query * (self.sqrt_head_dim / query_norm)
        key = key * (self.sqrt_head_dim / key_norm) * self.temp[None, None, :, None]

        query = query.reshape(*query.shape[:2], self.num_q_heads * self.head_dim)
        key = key.reshape(*key.shape[:2], self.num_kv_heads * self.head_dim)
        value = value.reshape(*value.shape[:2], self.num_kv_heads * self.head_dim)
        return (
            query.transpose(1, 0, 2),
            key.transpose(1, 0, 2),
            value.transpose(1, 0, 2),
        )


class ZayaAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.config = config
        self.layer_n = layer_n
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.qkv = CCA(config, layer_n)
        self.rope = initialize_rope(
            int(config.head_dim * config.rope_parameters["partial_rotary_factor"]),
            base=config.rope_parameters["rope_theta"],
            traditional=False,
            scaling_config=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
        )

        if config.vision_lora:
            r = config.vision_lora_rank_attn
            self.lora_linear_o = [
                nn.Linear(self.num_attention_heads * self.head_dim, r, bias=False),
                nn.Linear(r, config.hidden_size, bias=False),
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
    ):
        B, L, _ = hidden_states.shape
        kv_cache, _ = _split_cache(cache)
        q, k, v = self.qkv(hidden_states, cache, cca_mask, image_mask)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : k.shape[-2]]

        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        projected = self.o_proj(out)
        if self.config.vision_lora and image_mask is not None:
            addon = self.lora_linear_o[1](self.lora_linear_o[0](out))
            projected = projected + addon * image_mask[..., None].astype(
                projected.dtype
            )
        return projected


class ZayaRouter(nn.Module):
    def __init__(self, config: TextConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.use_mod = config.zaya_use_mod
        self.num_local_experts = config.num_experts
        self.num_experts = config.num_experts + (1 if self.use_mod else 0)
        self.topk = config.moe_router_topk
        self.layer_number = layer_number
        self.use_eda = config.zaya_use_eda and layer_number != 0

        self.down_proj = nn.Linear(
            config.hidden_size, config.zaya_mlp_expansion, bias=True
        )
        self.rmsnorm_eda = nn.RMSNorm(
            config.zaya_mlp_expansion, eps=config.norm_epsilon
        )
        if self.use_eda:
            self.router_states_scale = mx.ones((config.zaya_mlp_expansion,))

        self.router_mlp = [
            nn.Linear(config.zaya_mlp_expansion, config.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(config.zaya_mlp_expansion, config.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(config.zaya_mlp_expansion, self.num_experts, bias=False),
        ]
        self.balancing_biases = mx.zeros((self.num_experts,), dtype=mx.float32)
        if self.use_mod:
            self.balancing_biases[-1] = -1.0

    def __call__(
        self, hidden_states: mx.array, router_states: Optional[mx.array] = None
    ):
        hs = self.down_proj(hidden_states)
        if self.use_eda and router_states is not None:
            hs = hs + router_states * self.router_states_scale
        next_router_states = hs
        hs = self.rmsnorm_eda(hs)
        for layer in self.router_mlp:
            hs = layer(hs)

        expert_prob = mx.softmax(hs.astype(mx.float32), axis=-1).astype(
            hidden_states.dtype
        )
        biased = expert_prob.astype(mx.float32) + self.balancing_biases
        if self.topk == 1:
            expert_choice = mx.expand_dims(mx.argmax(biased, axis=-1), axis=-1)
            route_prob = mx.take_along_axis(expert_prob, expert_choice, axis=-1)
        else:
            expert_choice = mx.argpartition(biased, kth=-self.topk, axis=-1)[
                ..., -self.topk :
            ]
            route_prob = mx.take_along_axis(expert_prob, expert_choice, axis=-1)
        return route_prob, expert_choice, next_router_states


class ZayaSwitchMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.ffn_hidden_size_out = (
            config.ffn_hidden_size // 2
            if config.gated_linear_unit
            else config.ffn_hidden_size
        )
        self.linear_fc1 = SwitchLinear(
            config.hidden_size,
            self.ffn_hidden_size,
            config.num_experts,
            bias=config.add_bias_linear,
        )
        self.linear_fc2 = SwitchLinear(
            self.ffn_hidden_size_out,
            config.hidden_size,
            config.num_experts,
            bias=config.add_bias_linear,
        )

        if config.vision_lora:
            r = config.vision_lora_rank_mlp
            self.lora_fc1 = [
                SwitchLinear(config.hidden_size, r, config.num_experts, bias=False),
                SwitchLinear(r, self.ffn_hidden_size, config.num_experts, bias=False),
            ]
            self.lora_fc2 = [
                SwitchLinear(
                    self.ffn_hidden_size_out, r, config.num_experts, bias=False
                ),
                SwitchLinear(r, config.hidden_size, config.num_experts, bias=False),
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        expert_choice: mx.array,
        route_prob: mx.array,
        image_mask: Optional[mx.array] = None,
    ):
        skip_mask = expert_choice == self.num_experts
        expert_indices = mx.minimum(expert_choice, self.num_experts - 1)

        routed_hidden_states = hidden_states[..., None, None, :]
        x = self.linear_fc1(
            routed_hidden_states, expert_indices, sorted_indices=False
        ).squeeze(-2)
        if self.config.vision_lora and image_mask is not None:
            addon = self.lora_fc1[0](
                routed_hidden_states, expert_indices, sorted_indices=False
            ).squeeze(-2)
            addon = self.lora_fc1[1](
                addon[..., None, :], expert_indices, sorted_indices=False
            ).squeeze(-2)
            x = x + addon * image_mask[..., None, None].astype(x.dtype)

        if self.config.gated_linear_unit:
            x1, x2 = mx.split(x, 2, axis=-1)
            x = nn.silu(x1) * x2
        elif self.config.activation_func == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.silu(x)

        y = self.linear_fc2(
            x[..., None, :], expert_indices, sorted_indices=False
        ).squeeze(-2)
        if self.config.vision_lora and image_mask is not None:
            addon = self.lora_fc2[0](
                x[..., None, :], expert_indices, sorted_indices=False
            ).squeeze(-2)
            addon = self.lora_fc2[1](
                addon[..., None, :], expert_indices, sorted_indices=False
            ).squeeze(-2)
            y = y + addon * image_mask[..., None, None].astype(y.dtype)

        if self.config.zaya_use_mod:
            y = mx.where(skip_mask[..., None], hidden_states[..., None, :], y)

        y = y * route_prob[..., None]
        return y.sum(axis=-2)


class ZayaBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.router = ZayaRouter(config, layer_n)
        self.experts = ZayaSwitchMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        route_prob, expert_choice, prev_router_hidden_states = self.router(
            hidden_states, prev_router_hidden_states
        )
        output = self.experts(hidden_states, expert_choice, route_prob, image_mask)
        return output, prev_router_hidden_states


class ZayaDecoderATTLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.config = config
        self.self_attn = ZayaAttention(config, layer_n)
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, 2 * layer_n)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
        cca_mask: Optional[mx.array] = None,
    ):
        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states = self.self_attn(hidden_states, mask, cca_mask, image_mask, cache)
        return hidden_states, residual


class ZayaDecoderMLPLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.config = config
        self.zaya_block = ZayaBlock(config, layer_n)
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, 2 * layer_n + 1)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        image_mask: Optional[mx.array] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states, image_mask
        )
        return hidden_states, residual, prev_router_hidden_states


class ZayaDecoderBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_n: int):
        super().__init__()
        self.attn = ZayaDecoderATTLayer(config, layer_n)
        self.mlp = ZayaDecoderMLPLayer(config, layer_n)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
        prev_router_hidden_states: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
    ):
        hidden_states, residual = self.attn(
            hidden_states,
            residual,
            mask=mask,
            image_mask=image_mask,
            cache=cache,
            cca_mask=cca_mask,
        )
        hidden_states, residual, prev_router_hidden_states = self.mlp(
            hidden_states,
            residual,
            image_mask=image_mask,
            prev_router_hidden_states=prev_router_hidden_states,
        )
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            ZayaDecoderBlock(config, layer_n=i) for i in range(config.num_hidden_layers)
        ]
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, config.num_hidden_layers)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_epsilon)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
    ):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)

        first_kv_cache, _ = _split_cache(cache[0]) if cache else (None, None)
        attn_mask = create_attention_mask(h, first_kv_cache)
        padding_mask = _cache_position_mask(h, first_kv_cache)
        if mask is not None and getattr(mask, "ndim", 0) == 2:
            cca_mask = mask if padding_mask is None else mask & padding_mask
        else:
            cca_mask = padding_mask

        residual = None
        prev_router_hidden_states = None
        for layer, layer_cache in zip(self.layers, cache):
            h, residual, prev_router_hidden_states = layer(
                h,
                residual,
                mask=attn_mask,
                image_mask=image_mask,
                cache=layer_cache,
                prev_router_hidden_states=prev_router_hidden_states,
                cca_mask=cca_mask,
            )

        if self.config.scale_residual_merge:
            residual, h = self.res_scale(residual, h)
        residual = h if residual is None else h + residual
        return self.final_norm(residual)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=args.lm_head_bias
            )

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if image_mask is None:
            image_mask = kwargs.pop("visual_pos_masks", None)
        else:
            kwargs.pop("visual_pos_masks", None)

        if image_mask is not None and image_mask.shape[1] != input_ids.shape[1]:
            first_kv_cache, _ = _split_cache(cache[0]) if cache else (None, None)
            start = 0
            if first_kv_cache is not None and hasattr(first_kv_cache, "offset"):
                start = int(first_kv_cache.offset)
            image_mask = image_mask[:, start : start + input_ids.shape[1]]

        out = self.model(input_ids, inputs_embeds, mask, image_mask, cache)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(2)) for _ in self.layers]

    def sanitize(self, weights):
        sanitized_weights = dict(weights)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.mlp.zaya_block.experts"
            for name in ("linear_fc1", "linear_fc2"):
                stacked = []
                for expert_idx in range(self.args.num_experts):
                    key = f"{prefix}.local_experts.{expert_idx}.{name}.weight"
                    if key in sanitized_weights:
                        stacked.append(sanitized_weights.pop(key))
                if stacked:
                    sanitized_weights[f"{prefix}.{name}.weight"] = mx.stack(
                        stacked, axis=0
                    )

            if self.args.vision_lora:
                for lora_name in ("lora_fc1", "lora_fc2"):
                    for sub_idx in (0, 1):
                        stacked = []
                        for expert_idx in range(self.args.num_experts):
                            key = (
                                f"{prefix}.local_experts.{expert_idx}."
                                f"{lora_name}.{sub_idx}.weight"
                            )
                            if key in sanitized_weights:
                                stacked.append(sanitized_weights.pop(key))
                        if stacked:
                            sanitized_weights[
                                f"{prefix}.{lora_name}.{sub_idx}.weight"
                            ] = mx.stack(stacked, axis=0)

        result = {}
        for name, param in sanitized_weights.items():
            if "conv_qk" in name and name.endswith("weight") and param.ndim == 3:
                if param.shape[1] != 2 and param.shape[2] == 2:
                    param = param.transpose(0, 2, 1)
            result[name] = param
        return result
