from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.cache import CacheList, HierarchyCache, KVCache, PoolingCache
from ....models.glm5_next.language import Glm5NextAttention, Glm5NextMoE
from ..deepseek_v4_mtp.deepseek_v4_mtp import DeepseekV4MTPDraftModel
from .config import Glm5NextMTPConfig


def _clone_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


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

    def __call__(self, x, cache=None):
        residual = x
        attention, _ = self.self_attn(self.input_layernorm(x), cache=cache)
        x = residual + attention
        return x + self.mlp(self.post_attention_layernorm(x))


class Glm5NextMTPDraftModel(DeepseekV4MTPDraftModel):
    """Native GLM-5-Next MTP drafter backed by checkpoint decoder layer 45."""

    prefer_requested_block_size = False

    def __init__(self, config: Glm5NextMTPConfig):
        nn.Module.__init__(self)
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

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[CacheList] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position = 0
        self._round_appended = 0
        self._kv_valid_len = 0
        self._position = 0
        self._draft_round = 0
        self._round_cache_snapshot = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    @property
    def quant_predicate(self):
        return lambda _path, _module: True

    def make_cache(self) -> List[CacheList]:
        indexer = self.mtp_block.self_attn.indexer
        caches = [KVCache(), KVCache(), PoolingCache(indexer.index_kpool)]
        if indexer.hisa_block > 0:
            caches.append(HierarchyCache(indexer.hisa_block))
        caches.append(KVCache())
        return [CacheList(*caches)]

    def reset(self, target_model) -> List[CacheList]:
        caches = super().reset(target_model)
        self._round_cache_snapshot = None
        return caches

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            for subcache in cache.caches:
                # A batch MTP round can update the accepted seed before the
                # first draft forward initializes every KV cache. Only arrays
                # need to be synchronized for sampler-state isolation.
                if getattr(subcache, "keys", False) is None:
                    continue
                state.append(subcache.state)
        return state

    def validate_target_compatibility(self, target_model) -> None:
        language_model = getattr(target_model, "language_model", target_model)
        target_args = getattr(language_model, "args", None)
        target_type = getattr(target_args, "model_type", None)
        if target_type is not None and target_type not in (
            "glm5_next",
            "glm5_next_text",
        ):
            raise ValueError(
                "GLM-5-Next MTP must be paired with a GLM-5-Next target; "
                f"got model_type={target_type!r}."
            )

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del shared_kv_states, left_padding
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if position is None:
            position = kv_valid_len
        self._kv_valid_len = kv_valid_len
        self._position = position
        if not self._cache or all(cache.empty() for cache in self._cache):
            self._next_position = kv_valid_len

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        if hidden.ndim != 3 or hidden.shape[-1] != self.args.hidden_size:
            raise ValueError(
                "GLM-5-Next MTP expects target hidden shape "
                "[batch, tokens, hidden_size]."
            )
        return hidden

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[CacheList]],
    ) -> Tuple[mx.array, mx.array]:
        del tokens
        hidden = self._target_hidden(hidden)
        position_ids = self._position_ids(length=token_embed.shape[1])
        token_embed = mx.where(position_ids[..., None] == 0, 0, token_embed)
        h = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )
        h = self.mtp_block(h, None if cache is None else cache[0])
        h = self.shared_head_norm(h)
        return h, h

    def draft_block(self, *args, **kwargs) -> mx.array:
        block_size = kwargs.get("block_size")
        if block_size is None and len(args) >= 4:
            block_size = args[3]
        if block_size is not None and int(block_size) > 2:
            self._round_cache_snapshot = [
                (_clone_tree(cache.state), _clone_tree(cache.meta_state))
                for cache in self._cache
            ]
            self._round_snapshot_position = _clone_tree(self._next_position)
        else:
            self._round_cache_snapshot = None
        return super().draft_block(*args, **kwargs)

    def _restore_untrimmable_round(self, accepted: int) -> None:
        rejected_appended = max(0, self._round_appended - int(accepted))
        if not rejected_appended or self._round_cache_snapshot is None:
            return
        if all(cache.is_trimmable() for cache in self._cache):
            return
        for cache, (state, meta_state) in zip(self._cache, self._round_cache_snapshot):
            cache.meta_state = _clone_tree(meta_state)
            cache.state = _clone_tree(state)
        self._next_position = _clone_tree(self._round_snapshot_position)
        self._round_appended = 0

    def accept_verified_tokens(self, *args, **kwargs) -> None:
        accepted = kwargs.get("accepted")
        if accepted is None and len(args) >= 3:
            accepted = args[2]
        self._restore_untrimmable_round(int(accepted))
        super().accept_verified_tokens(*args, **kwargs)
        self._round_cache_snapshot = None

    def accept_verified_tokens_batch(self, *args, **kwargs) -> None:
        accepted = kwargs.get("accepted")
        if accepted is None and len(args) >= 3:
            accepted = args[2]
        accepted_values = [int(value) for value in accepted]
        if len(set(accepted_values)) != 1:
            raise ValueError(
                "GLM-5-Next MTP batched cache update requires uniform acceptance."
            )
        self._restore_untrimmable_round(accepted_values[0])
        super().accept_verified_tokens_batch(*args, **kwargs)
        self._round_cache_snapshot = None

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]
        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])

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
