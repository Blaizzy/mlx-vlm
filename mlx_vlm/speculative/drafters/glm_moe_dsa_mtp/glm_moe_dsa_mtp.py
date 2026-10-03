from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, CacheList, KVCache
from ....models.glm_moe_dsa.language import GlmMoeDsaMTP
from .config import GlmMoeDsaMTPConfig


class GlmMoeDsaMTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True
    supports_ragged_batch_acceptance = False

    def __init__(self, config: GlmMoeDsaMTPConfig):
        super().__init__()
        self.config = config
        if config.text_config is None:
            raise ValueError("GlmMoeDsaMTPConfig.text_config must be set")
        self.mtp = GlmMoeDsaMTP(config.text_config)
        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        model_type = getattr(target, "model_type", None)
        if model_type != "glm_moe_dsa":
            raise ValueError(
                "glm_moe_dsa_mtp requires a glm_moe_dsa target model; "
                f"got {model_type!r}."
            )

    def bind(self, target_model) -> "GlmMoeDsaMTPDraftModel":
        target = getattr(target_model, "language_model", target_model)
        if not hasattr(target, "model") or not hasattr(target.model, "embed_tokens"):
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self._input_embed = target.model.embed_tokens
        self._lm_head_fn = target.lm_head
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        if left_padding is not None:
            return [CacheList(BatchKVCache(left_padding), BatchKVCache(left_padding))]
        return [CacheList(KVCache(), KVCache())]

    def reset(
        self, target_model, left_padding: Optional[List[int]] = None
    ) -> List[Any]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache(left_padding)
        self._seed_token = None
        self._seed_hidden = None
        self._round_appended = 0
        return self._cache

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            state.append(cache.state)
        return state

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del shared_kv_states, kv_offset, position, kv_valid_len, left_padding

    def _mask_for(self, x: mx.array, cache) -> Optional[mx.array]:
        if cache is None or x.shape[1] == 1:
            return None
        return create_attention_mask(x, cache[0], return_array=True)

    def _forward_tokens(
        self, tokens: mx.array, hidden: mx.array, token_dtype: mx.Dtype
    ) -> mx.array:
        token_embed = self._input_embed(tokens.astype(token_dtype))
        cache = self._cache[0] if self._cache else None
        return self.mtp(hidden, token_embed, self._mask_for(token_embed, cache), cache)

    def _set_seed_from_hidden(self, hidden: mx.array, sampler, greedy: bool) -> None:
        logits = self._lm_head_fn(hidden)
        self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        self._seed_hidden = hidden

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)
        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        drafted_hidden = self._forward_tokens(
            shifted, hidden[:, : shifted.shape[1], :], token_dtype
        )
        self._set_seed_from_hidden(drafted_hidden[:, -1:, :], sampler, greedy)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        keep_appended = min(int(accepted), self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)

        token_chunks = []
        hidden_chunks = []
        for draft_index in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_index : draft_index + 1])
            hidden_chunks.append(verify_hidden[:, draft_index : draft_index + 1, :])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])
        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            drafted_hidden = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(drafted_hidden[:, -1:, :], sampler, greedy)
        self._round_appended = 0

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        self.accept_verified_tokens(
            verify_hidden,
            draft_tokens,
            int(accepted[0]),
            new_tokens[0],
            sampler,
            token_dtype,
            greedy,
        )

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError("bind(target_model) must run before draft_block().")
        if isinstance(last_bonus, int):
            token = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            token = last_bonus[:, None].astype(token_dtype)

        previous_hidden = hidden
        tokens: List[mx.array] = []
        self._round_appended = 0
        if self._seed_token is not None and self._seed_hidden is not None:
            token = self._seed_token.astype(token_dtype)
            previous_hidden = self._seed_hidden
            tokens.append(token)
            self._seed_token = None
            self._seed_hidden = None

        while len(tokens) < block_size - 1:
            previous_hidden = self._forward_tokens(token, previous_hidden, token_dtype)
            self._round_appended += 1
            logits = self._lm_head_fn(previous_hidden)
            token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(token)
        return mx.concatenate(tokens, axis=1)
