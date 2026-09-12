"""Native Qwen3.5 MTP architecture; all request state lives in SpeculativeCache."""

import re
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, KVCache
from ....models.linear import linear
from ....models.qwen3_5.language import Qwen3_5DecoderLayer
from ....models.qwen3_5_moe.language import Qwen3_5MoeDecoderLayer
from .config import Qwen3_5MTPConfig


class Qwen3_5MTPDraftModel(nn.Module):
    def __init__(self, config: Qwen3_5MTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Qwen3_5MTPConfig.text_config must be set")

        hidden_size = text_config.hidden_size
        mtp_layers = int(getattr(text_config, "mtp_num_hidden_layers", 1))
        layer_config = replace(
            text_config,
            num_hidden_layers=mtp_layers,
            full_attention_interval=1,
        )
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            hidden_size, eps=text_config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        layer_cls = (
            Qwen3_5MoeDecoderLayer
            if "moe" in getattr(text_config, "model_type", "")
            else Qwen3_5DecoderLayer
        )
        self.layers = [
            layer_cls(args=layer_config, layer_idx=0) for _ in range(mtp_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

    def make_cache(self, left_padding=None):
        return [
            KVCache() if left_padding is None else BatchKVCache(left_padding)
            for _ in self.layers
        ]

    def __call__(self, tokens, hidden, cache, position, target_model, lengths=None):
        positions = mx.array(position, dtype=mx.int32).reshape(-1, 1)
        positions = positions + mx.arange(tokens.shape[1], dtype=mx.int32)[None]
        embedding = target_model.model.embed_tokens(tokens)
        embedding = embedding * getattr(target_model.model, "embed_scale", 1.0)
        hidden = linear(
            self.fc,
            mx.concatenate(
                [
                    self.pre_fc_norm_embedding(embedding),
                    self.pre_fc_norm_hidden(hidden),
                ],
                axis=-1,
            ),
        )
        for layer, entry in zip(self.layers, cache):
            hidden = layer(
                hidden,
                mask=create_attention_mask(hidden, entry),
                cache=entry,
                position_ids=positions,
            )
        hidden = self.norm(hidden)
        last = (
            hidden[:, -1:]
            if lengths is None
            else mx.take_along_axis(
                hidden, mx.maximum(mx.array(lengths), 1)[:, None, None] - 1, axis=1
            )
        )
        head = (
            target_model.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else target_model.lm_head
        )
        return linear(head, last), hidden

    def sanitize(self, weights: dict) -> dict:
        out = {}
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        )
        for key, value in weights.items():
            is_hf_layout = key.startswith("mtp.")
            if is_hf_layout:
                key = key[len("mtp.") :]
            if is_hf_layout and any(key.endswith(suffix) for suffix in norm_suffixes):
                if value.ndim == 1 and mx.issubdtype(value.dtype, mx.floating):
                    value = value + 1.0
            out[key] = value
        expert_prefixes = [
            key[: -len(".experts.gate_up_proj")]
            for key in out
            if key.endswith(".experts.gate_up_proj")
        ]
        for prefix in expert_prefixes:
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            gate_up_weight = out.pop(gate_up_key)
            gate_weight, up_weight = mx.split(gate_up_weight, 2, axis=-2)
            out[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
            out[f"{prefix}.switch_mlp.up_proj.weight"] = up_weight

            gate_up_scales_key = f"{gate_up_key}_scales"
            if gate_up_scales_key in out:
                gate_scales, up_scales = mx.split(
                    out.pop(gate_up_scales_key), 2, axis=-2
                )
                out[f"{prefix}.switch_mlp.gate_proj.scales"] = gate_scales
                out[f"{prefix}.switch_mlp.up_proj.scales"] = up_scales

            down_key = f"{prefix}.experts.down_proj"
            out[f"{prefix}.switch_mlp.down_proj.weight"] = out.pop(down_key)
            if f"{down_key}_scales" in out:
                out[f"{prefix}.switch_mlp.down_proj.scales"] = out.pop(
                    f"{down_key}_scales"
                )

        pattern = re.compile(
            r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\."
            r"(weight|scales|biases)$"
        )
        groups = {}
        for key in out:
            if match := pattern.fullmatch(key):
                expert_prefix, expert, projection, suffix = match.groups()
                groups.setdefault((expert_prefix, projection, suffix), {})[
                    int(expert)
                ] = key

        for (expert_prefix, projection, suffix), expert_keys in groups.items():
            experts = sorted(expert_keys)
            if experts != list(range(len(experts))):
                raise ValueError(
                    f"Qwen MTP expert indexes are not contiguous for {expert_prefix}: "
                    f"{experts}."
                )
            base = expert_prefix[: -len(".experts")]
            out[f"{base}.switch_mlp.{projection}.{suffix}"] = mx.stack(
                [out.pop(expert_keys[expert]) for expert in experts]
            )

        return out
