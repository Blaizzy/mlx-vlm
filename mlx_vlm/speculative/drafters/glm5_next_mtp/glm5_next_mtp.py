from dataclasses import replace
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.cache import (
    BatchKVCache,
    BatchPoolingCache,
    CacheList,
    KVCache,
    PoolingCache,
)
from ....models.glm5_next.language import Glm5NextAttention, Glm5NextMoE
from ....models.linear import linear
from .config import Glm5NextMTPConfig


class Glm5NextMTPBlock(nn.Module):
    """The checkpoint's decoder layer after the 45 hyperconnected target layers."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Glm5NextAttention(config, layer_idx)
        self.mlp = Glm5NextMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, cache=None, padding_mask=None, last_only: bool = False):
        residual = x
        attention, _ = self.self_attn(
            self.input_layernorm(x),
            cache=cache,
            padding_mask=padding_mask,
            last_only=last_only,
        )
        if last_only and x.shape[1] > 1:
            residual = residual[:, -1:]
        x = residual + attention
        return x + self.mlp(self.post_attention_layernorm(x))


class Glm5NextMTPDraftModel(nn.Module):
    """GLM-5.3-Flash's native MTP head. All request state belongs to the cache."""

    def __init__(self, config: Glm5NextMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Glm5NextMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        layer_idx = text_config.num_hidden_layers
        layer_config = replace(
            text_config,
            num_hidden_layers=layer_idx + 1,
            layer_types=[*text_config.layer_types, "deepseek_sparse_attention"],
            mlp_layer_types=[*text_config.mlp_layer_types, "sparse"],
            indexer_types=[*text_config.indexer_types, "full"],
        )
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.mtp_block = Glm5NextMTPBlock(layer_config, layer_idx)
        self.shared_head_norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

    @property
    def quant_predicate(self):
        return lambda _path, _module: True

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[CacheList]:
        indexer = self.mtp_block.self_attn.indexer
        if left_padding is None:
            kv_cache = KVCache
            pool_cache = PoolingCache(indexer.index_kpool)
        else:
            kv_cache = lambda: BatchKVCache(left_padding)
            pool_cache = BatchPoolingCache(indexer.index_kpool, left_padding)
        return [CacheList(kv_cache(), kv_cache(), pool_cache, kv_cache())]

    def __call__(self, tokens, hidden, cache, position, target_model, lengths=None):
        """Predict from shifted tokens and target (or previous MTP) features."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.args.hidden_size:
            raise ValueError(
                "MTP hidden states must have shape [batch, tokens, hidden_size]."
            )
        positions = mx.array(position, dtype=mx.int32).reshape(-1, 1)
        positions = positions + mx.arange(tokens.shape[1], dtype=mx.int32)[None]
        embeddings = target_model.model.embed_tokens(tokens)
        embeddings = mx.where(positions[..., None] == 0, 0, embeddings)
        hidden = linear(
            self.eh_proj,
            mx.concatenate([self.enorm(embeddings), self.hnorm(hidden)], axis=-1),
        )
        hidden = self.shared_head_norm(
            self.mtp_block(hidden, cache[0], padding_mask=positions >= 0)
        )
        head = (
            target_model.model.embed_tokens.as_linear
            if target_model.args.tie_word_embeddings
            else target_model.lm_head
        )
        # Only the final prediction seeds the next draft step.
        last = (
            hidden[:, -1:]
            if lengths is None
            else mx.take_along_axis(
                hidden, mx.maximum(mx.array(lengths), 1)[:, None, None] - 1, axis=1
            )
        )
        return linear(head, last), hidden

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = dict(weights)

        mlp_prefix = "mtp_block.mlp.shared_experts"
        for suffix in ("weight", "scales", "biases"):
            source_keys = [
                f"{mlp_prefix}.{projection}.{suffix}"
                for projection in ("gate_proj", "up_proj")
            ]
            if all(key in weights for key in source_keys):
                weights[f"{mlp_prefix}.gate_up_proj.{suffix}"] = mx.concatenate(
                    [weights.pop(key) for key in source_keys], axis=0
                )

        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key0 = f"mtp_block.mlp.experts.0.{projection}.{suffix}"
                if key0 not in weights:
                    continue
                values = [
                    weights.pop(f"mtp_block.mlp.experts.{expert}.{projection}.{suffix}")
                    for expert in range(self.args.n_routed_experts)
                ]
                weights[f"mtp_block.mlp.switch_mlp.{projection}.{suffix}"] = mx.stack(
                    values
                )

        attn_prefix = "mtp_block.self_attn"
        for suffix in ("weight", "scales", "biases", "bias"):
            source_keys = [
                f"{attn_prefix}.{projection}.{suffix}"
                for projection in ("q_a_proj", "kv_a_proj_with_mqa")
            ]
            if all(key in weights for key in source_keys):
                weights[f"{attn_prefix}.qkv_a_proj.{suffix}"] = mx.concatenate(
                    [weights.pop(key) for key in source_keys], axis=0
                )

        kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
        if kv_b_key in weights:
            value = weights.pop(kv_b_key)
            quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
            if quantized:
                scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases", None)
                bits = value.shape[-1] * 32 // self.args.kv_lora_rank
                group_size = self.args.kv_lora_rank // scales.shape[-1]
                mode = "mxfp8" if biases is None and bits == 8 else "affine"
                dequantize_kwargs = {
                    "bits": bits,
                    "group_size": group_size,
                    "mode": mode,
                }
                value = (
                    mx.dequantize(value, scales, **dequantize_kwargs)
                    if biases is None
                    else mx.dequantize(value, scales, biases, **dequantize_kwargs)
                )
            value = value.reshape(
                self.args.num_attention_heads,
                self.args.qk_nope_head_dim + self.args.v_head_dim,
                self.args.kv_lora_rank,
            )
            wk = mx.contiguous(value[:, : self.args.qk_nope_head_dim].swapaxes(-1, -2))
            wv = mx.contiguous(value[:, self.args.qk_nope_head_dim :])
            if quantized:
                wk, wk_scales, *wk_biases = mx.quantize(
                    wk, bits=bits, group_size=group_size, mode=mode
                )
                wv, wv_scales, *wv_biases = mx.quantize(
                    wv, bits=bits, group_size=group_size, mode=mode
                )
                weights[f"{attn_prefix}.embed_q.scales"] = wk_scales
                weights[f"{attn_prefix}.unembed_out.scales"] = wv_scales
                if wk_biases:
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_biases[0]
                if wv_biases:
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_biases[0]
            weights[f"{attn_prefix}.embed_q.weight"] = wk
            weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights
