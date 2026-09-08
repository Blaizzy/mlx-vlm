from dataclasses import replace
from typing import Dict, List, Optional, Tuple

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
from ...cache_state import start_speculative_cache
from ...common import _prepare_ragged_mtp_replay
from ..mtp_base import AutoregressiveMTPDraftModel
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

    def __call__(self, x, cache=None, last_only: bool = False):
        residual = x
        attention, _ = self.self_attn(
            self.input_layernorm(x),
            cache=cache,
            last_only=last_only,
        )
        if last_only and x.shape[1] > 1:
            residual = residual[:, -1:]
        x = residual + attention
        return x + self.mlp(self.post_attention_layernorm(x))


class Glm5NextMTPDraftModel(AutoregressiveMTPDraftModel):
    """Native GLM-5-Next MTP drafter backed by checkpoint decoder layer 45."""

    prefer_requested_block_size = False
    requires_uniform_batch_acceptance = False
    supports_ragged_batch_acceptance = True
    supports_left_padded_prefill = True

    def __init__(self, config: Glm5NextMTPConfig):
        super().__init__(config)
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

        self._round_transaction = None
        self._round_initial = None

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

    def reset(
        self, target_model, left_padding: Optional[List[int]] = None
    ) -> List[CacheList]:
        self.abort_draft_round()
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache(left_padding)
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
        self._round_appended = 0
        self._round_transaction = None
        return self._cache

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
        del shared_kv_states, left_padding, position
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if not self._cache or all(cache.empty() for cache in self._cache):
            self._next_position = kv_valid_len

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        if hidden.ndim != 3 or hidden.shape[-1] != self.args.hidden_size:
            raise ValueError(
                "GLM-5-Next MTP expects target hidden shape "
                "[batch, tokens, hidden_size]."
            )
        return hidden

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        left_padding: Optional[List[int]] = None,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)

        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        self._next_position = (
            0 if left_padding is None else -mx.array(left_padding, dtype=mx.int32)
        )
        logits_hidden, pre_hc_hidden = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], ...],
            token_dtype,
        )
        self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
        self._seed_hidden = pre_hc_hidden[:, -1:, ...]

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[CacheList]],
        *,
        last_only=False,
    ) -> Tuple[mx.array, mx.array]:
        del tokens
        hidden = self._target_hidden(hidden)
        position_ids = self._position_ids(length=token_embed.shape[1])
        token_embed = mx.where(position_ids[..., None] == 0, 0, token_embed)
        h = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )
        h = self.mtp_block(
            h,
            None if cache is None else cache[0],
            last_only=last_only,
        )
        h = self.shared_head_norm(h)
        return h, h

    def draft_block(
        self,
        last_bonus,
        hidden,
        cache,
        block_size,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        self._round_initial = (self._next_position, self._seed_token, self._seed_hidden)
        # A cached seed contributes a token without appending a draft state.
        seeded = self._seed_token is not None and self._seed_hidden is not None
        steps = block_size - 1 - int(seeded)
        self._round_transaction = (
            start_speculative_cache(self._cache, steps) if steps > 0 else None
        )
        try:
            return super().draft_block(
                last_bonus, hidden, cache, block_size, sampler, token_dtype, greedy
            )
        except BaseException:
            self.abort_draft_round()
            raise

    def abort_draft_round(self):
        if self._round_transaction is not None:
            self._round_transaction.abort()
            self._round_transaction = None
        if self._round_initial is not None:
            self._next_position, self._seed_token, self._seed_hidden = (
                self._round_initial
            )
            self._round_initial = None
        self._round_appended = 0

    def accept_verified_tokens(
        self,
        verify_hidden,
        draft_tokens,
        accepted,
        new_tokens,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        self.accept_verified_tokens_batch(
            verify_hidden,
            draft_tokens,
            [accepted],
            [new_tokens],
            sampler,
            token_dtype,
            greedy,
        )

    def accept_verified_tokens_batch(
        self,
        verify_hidden,
        draft_tokens,
        accepted,
        new_tokens,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        kept = [min(int(value), self._round_appended) for value in accepted]
        if self._round_transaction is not None:
            self._round_transaction.commit(kept)
            self._round_transaction = None
        self._round_initial = None
        trims = [self._round_appended - value for value in kept]
        self._next_position -= (
            trims[0] if len(trims) == 1 else mx.array(trims, dtype=mx.int32)
        )
        self._round_appended = 0

        tokens, hiddens, lengths, right_padding = _prepare_ragged_mtp_replay(
            verify_hidden,
            draft_tokens,
            accepted,
            new_tokens,
            kept,
            token_dtype,
        )
        if tokens is None:
            return
        if any(right_padding):
            for cache in self._cache:
                cache.prepare(right_padding=right_padding, lengths=lengths)
        last_only = len(accepted) == 1
        logits_hidden, draft_hidden = self._forward_tokens(
            tokens,
            hiddens,
            token_dtype,
            last_only=last_only,
        )
        if any(right_padding):
            for cache in self._cache:
                cache.finalize()
            self._next_position -= mx.array(right_padding, dtype=mx.int32)
        if not last_only:
            last = mx.array(lengths, dtype=mx.int32)[:, None, None] - 1
            logits_hidden = mx.take_along_axis(logits_hidden, last, axis=1)
            draft_hidden = mx.take_along_axis(draft_hidden, last, axis=1)
        self._set_seed_from_hidden(logits_hidden, sampler, greedy)
        self._seed_hidden = draft_hidden

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]
        value = self._next_position
        if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
            self._next_position = value[keep]

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
