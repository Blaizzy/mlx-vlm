from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, CacheList, KVCache
from ....models.glm5_next.mtp import Glm5NextMTP, load_mtp_weights
from .config import Glm5NextMTPConfig


class Glm5NextMTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True
    supports_ragged_batch_acceptance = False

    def __init__(self, config: Glm5NextMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Glm5NextMTPConfig.text_config must be set")
        # The GLM-5-Next nextn (layer-45) block: enorm/hnorm -> eh_proj -> DSA+MoE ->
        # shared_head_norm. Drafts token t+2 from the target hidden h(t+1) and the
        # embedding of the accepted token t+1; the (shared) lm_head is applied here.
        self.mtp = Glm5NextMTP(text_config)

        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "Glm5NextMTPDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )

        self._input_embed = inner.embed_tokens
        self._input_embed_scale = float(getattr(inner, "embed_scale", 1.0))

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        # One sparse (MLA + lightning-indexer) self-attention -> one CacheList of two
        # KV caches (MLA latents + indexer keys), matching Glm5NextSparseAttention.
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
        # The nextn block keeps its own KV/indexer cache (MLA is NoPE, so there is no
        # position to import); the target's shared K/V is not reused.
        del shared_kv_states, kv_offset, position, kv_valid_len, left_padding

    def _mask_for(self, x: mx.array, cache) -> Optional[mx.array]:
        if cache is None or x.shape[1] == 1:
            return None
        mla = cache[0]
        ref = mla if (mla is not None and not mla.empty()) else None
        return create_attention_mask(x, ref, return_array=True)

    def _forward_tokens(
        self, tokens: mx.array, hidden: mx.array, token_dtype: mx.Dtype
    ) -> mx.array:
        token_embed = (
            self._input_embed(tokens.astype(token_dtype)) * self._input_embed_scale
        )
        cache = self._cache[0] if self._cache else None
        mask = self._mask_for(token_embed, cache)
        return self.mtp(hidden, token_embed, mask, cache)

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
        h = self._forward_tokens(shifted, hidden[:, : shifted.shape[1], :], token_dtype)
        self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

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
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])

        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
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
        # Uniform acceptance (requires_uniform_batch_acceptance=True): every row advances
        # by the same count, but each row has its own accepted drafts + next token, so the
        # appended tokens/hidden stay [B, n] (not the single-row [1, n]).
        a = int(accepted[0])
        keep = min(a, self._round_appended)
        trim = self._round_appended - keep
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep, a):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])
        if all(len(nt) for nt in new_tokens):
            nt_col = mx.array(
                [[int(nt[-1])] for nt in new_tokens], dtype=token_dtype
            )  # [B, 1]
            token_chunks.append(nt_col)
            hidden_chunks.append(verify_hidden[:, a : a + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
        self._round_appended = 0

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache_filter = getattr(cache, "filter", None)
            if callable(cache_filter):
                cache_filter(keep)
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
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []
        self._round_appended = 0

        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._seed_token.astype(token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            self._seed_token = None
            self._seed_hidden = None

        while len(tokens) < block_size - 1:
            h_prev = self._forward_tokens(tok, h_prev, token_dtype)
            self._round_appended += 1
            logits = self._lm_head_fn(h_prev)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok)

        return mx.concatenate(tokens, axis=1)

    def sanitize(self, weights: dict) -> dict:
        # A split drafter checkpoint already carries the mtp.* tree; raw layer-45
        # tensors (a manual load) are mapped through load_mtp_weights.
        if any(k.startswith("language_model.model.layers.") for k in weights):
            mapped = load_mtp_weights(self.config.text_config, weights)
            return {f"mtp.{k}": v for k, v in mapped.items()}
        return weights
