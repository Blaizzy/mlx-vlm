"""Ming-Image conditioning encoder: prompt -> caption features for the DiT.

The prompt is templated, and a block of 256 learnable query tokens is appended
in the image slots. A single BailingMoeV2 forward produces per-layer hidden
states. Two conditioning streams are derived:

* **cap_feats** (learnable-token branch): the final hidden states at the 256
  query positions -> ``proj_in`` -> a bidirectional Qwen2 connector ->
  ``proj_out`` -> ``[1, 256, 2560]``.
* **cap_feats_2** (direct-VLM branch): hidden states from layers [5, 12, 20]
  concatenated over the real text tokens -> norm -> ``proj_directvlm`` ->
  ``[1, text_len, 3840]``.

The BailingMoeV2 MoE uses a per-modality MultiRouter: the 256 query tokens route
through ``image_gate``, all other tokens through ``gate``; a shared expert is
applied to every token. Only these two of the three routers are needed for a
text-to-image prompt (audio is never present).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.bailing_moe.config import ModelConfig as BailingConfig
from mlx_vlm.models.bailing_moe.language import (
    BailingMoeAttention,
    BailingMoeGate,
    aggregate_expert_outputs,
)
from mlx_vlm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_vlm.models.mlp import SwiGLUMLP
from mlx_vlm.models.switch_layers import SwitchGLU

from .config import MingImageConnectorConfig, MingImageMLLMConfig


def _bailing_config(config: MingImageMLLMConfig) -> BailingConfig:
    return BailingConfig(
        model_type="bailing_moe_v2",
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        moe_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        num_shared_experts=config.num_shared_experts,
        norm_topk_prob=config.norm_topk_prob,
        num_attention_heads=config.num_attention_heads,
        num_experts_per_tok=config.num_experts_per_tok,
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        vocab_size=config.vocab_size,
        first_k_dense_replace=config.first_k_dense_replace,
        rope_scaling=None,
        use_qk_norm=config.use_qk_norm,
        partial_rotary_factor=config.partial_rotary_factor,
        moe_router_enable_expert_bias=config.use_expert_bias,
        routed_scaling_factor=config.routed_scaling_factor,
        score_function=config.score_function,
        n_group=config.n_group,
        topk_group=config.topk_group,
    )


class MingMoeBlock(nn.Module):
    """MoE with a text ``gate`` and an ``image_gate``, selected per token."""

    def __init__(self, args: BailingConfig) -> None:
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.gate = BailingMoeGate(args)
        self.image_gate = BailingMoeGate(args)
        self.shared_experts = SwiGLUMLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts
        )

    def __call__(self, x: mx.array, image_mask: mx.array) -> mx.array:
        text_idx, text_weight = self.gate(x)
        image_idx, image_weight = self.image_gate(x)
        select = image_mask[..., None]
        idx = mx.where(select, image_idx, text_idx)
        weight = mx.where(select, image_weight, text_weight)
        out = aggregate_expert_outputs(self.switch_mlp(x, idx), weight)
        return out + self.shared_experts(x)


class MingDecoderLayer(nn.Module):
    def __init__(self, args: BailingConfig, layer_idx: int) -> None:
        super().__init__()
        self.attention = BailingMoeAttention(args)
        self.is_moe = layer_idx >= args.first_k_dense_replace
        self.mlp = (
            MingMoeBlock(args)
            if self.is_moe
            else SwiGLUMLP(args.hidden_size, args.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self, x: mx.array, mask: mx.array | None, image_mask: mx.array
    ) -> mx.array:
        h = x + self.attention(self.input_layernorm(x), mask)
        normed = self.post_attention_layernorm(h)
        out = self.mlp(normed, image_mask) if self.is_moe else self.mlp(normed)
        return h + out


class MingMLLM(nn.Module):
    """BailingMoeV2 text backbone returning every layer's hidden state."""

    def __init__(self, args: BailingConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MingDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, image_mask: mx.array) -> list[mx.array]:
        mask = create_attention_mask(inputs_embeds, None)
        hidden_states = []
        h = inputs_embeds
        for layer in self.layers:
            hidden_states.append(h)
            h = layer(h, mask, image_mask)
        hidden_states.append(self.norm(h))
        return hidden_states


class Qwen2Attention(nn.Module):
    def __init__(self, config: MingImageConnectorConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array) -> mx.array:
        b, length, _ = x.shape
        q = self.q_proj(x).reshape(b, length, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = (
            self.k_proj(x)
            .reshape(b, length, self.num_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(b, length, self.num_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        q = self.rope(q)
        k = self.rope(k)
        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=None
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class Qwen2Layer(nn.Module):
    def __init__(self, config: MingImageConnectorConfig) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen2Connector(nn.Module):
    """Bidirectional Qwen2 encoder over the 256 caption query tokens."""

    def __init__(self, config: MingImageConnectorConfig) -> None:
        super().__init__()
        self.layers = [Qwen2Layer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MingImageTextEncoder(nn.Module):
    def __init__(
        self, mllm: MingImageMLLMConfig, connector: MingImageConnectorConfig, bridge
    ) -> None:
        super().__init__()
        self.image_patch_token = mllm.image_patch_token
        self.image_start_token = mllm.image_start_token
        self.image_end_token = mllm.image_end_token
        self.query_token_count = bridge.query_token_count
        self.selected_layers = tuple(bridge.selected_hidden_states_layers)
        self.mllm = MingMLLM(_bailing_config(mllm))
        self.connector = Qwen2Connector(connector)
        self.query_tokens = mx.zeros((bridge.query_token_count, bridge.mllm_hidden))
        self.proj_in = nn.Linear(bridge.mllm_hidden, bridge.connector_hidden)
        self.proj_out = nn.Linear(bridge.connector_hidden, bridge.cap_feat_dim)
        self.directvlm_norm = nn.RMSNorm(bridge.directvlm_in)
        self.directvlm_proj = nn.Linear(bridge.directvlm_in, bridge.directvlm_dim)

    def encode(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        ids = input_ids.reshape(-1).tolist()
        start = ids.index(self.image_patch_token)
        count = self.query_token_count
        block = {self.image_patch_token, self.image_start_token, self.image_end_token}
        text_positions = [i for i, token in enumerate(ids) if token not in block]

        embeds = self.mllm.word_embeddings(input_ids)
        embeds = mx.concatenate(
            [
                embeds[:, :start],
                self.query_tokens[None].astype(embeds.dtype),
                embeds[:, start + count :],
            ],
            axis=1,
        )
        image_mask = mx.array([token == self.image_patch_token for token in ids])[None]
        hidden_states = self.mllm(embeds, image_mask)

        query_hidden = hidden_states[-1][:, start : start + count]
        cap_feats = self.proj_out(self.connector(self.proj_in(query_hidden)))

        stacked = mx.concatenate(
            [hidden_states[layer] for layer in self.selected_layers], axis=-1
        )
        text = stacked[:, mx.array(text_positions)]
        cap_feats_2 = self.directvlm_proj(self.directvlm_norm(text))
        return cap_feats, cap_feats_2


__all__ = ["MingImageTextEncoder"]
