"""Ming-Image conditioning encoder: prompt -> caption features for the DiT."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.bailing_moe.config import ModelConfig as BailingConfig
from mlx_vlm.models.bailing_moe.language import (
    BailingMoeAttention,
    BailingMoeGate,
    aggregate_expert_outputs,
)
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.mlp import SwiGLUMLP
from mlx_vlm.models.qwen2.config import ModelConfig as Qwen2Config
from mlx_vlm.models.qwen2.language import TransformerBlock as Qwen2Block
from mlx_vlm.models.switch_layers import SwitchGLU

from .config import MingImageConnectorConfig


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


class Qwen2Connector(nn.Module):
    """Bidirectional Qwen2 encoder (shared engine blocks) over the query tokens."""

    def __init__(self, config: MingImageConnectorConfig) -> None:
        super().__init__()
        args = Qwen2Config(
            model_type="qwen2",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            vocab_size=1,
        )
        self.layers = [Qwen2Block(args) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=None)
        return self.norm(x)


class MingImageTextEncoder(nn.Module):
    def __init__(self, config: MingImageConfig) -> None:
        super().__init__()
        bridge = config.bridge
        self.image_patch_token = config.image_patch_token
        self.image_start_token = config.image_start_token
        self.image_end_token = config.image_end_token
        self.query_token_count = bridge.query_token_count
        self.selected_layers = tuple(bridge.selected_hidden_states_layers)
        self.mllm = MingMLLM(config.mllm)
        self.connector = Qwen2Connector(config.connector)
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
