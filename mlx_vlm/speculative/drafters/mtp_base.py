"""Autoregressive MTP session behavior shared by checkpoint-specific drafters."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn


class AutoregressiveMTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._input_embed = None
        self._lm_head_fn = None
        self._cache = []
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
        self._round_appended = 0
        self._draft_round = 0
        self.accept_lens = []
        self.draft_lens = []

    def bind(self, target_model):
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

        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def reset(self, target_model) -> list:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache()
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
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
        del shared_kv_states, left_padding, position
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if not self._cache or self._cache[0].offset == 0:
            self._next_position = kv_valid_len

    def _position_ids(self, length: int = 1) -> mx.array:
        start = self._next_position
        pos = mx.arange(length, dtype=mx.int32)
        if isinstance(start, int):
            return (pos + start)[None, :]
        if isinstance(start, mx.array):
            return start.astype(mx.int32)[:, None] + pos[None, :]
        return mx.array(start, dtype=mx.int32)[:, None] + pos[None, :]

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
        **kwargs,
    ) -> Tuple[mx.array, mx.array]:
        token_embed = self._input_embed(tokens.astype(token_dtype))
        logits_hidden, pre_hc_hidden = self._forward_hidden(
            token_embed,
            hidden[:, : tokens.shape[1], ...],
            tokens,
            self._cache,
            **kwargs,
        )
        steps = int(tokens.shape[1])
        self._next_position += steps
        return logits_hidden, pre_hc_hidden

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
        self._next_position = 0
        logits_hidden, pre_hc_hidden = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], ...],
            token_dtype,
        )
        self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
        self._seed_hidden = pre_hc_hidden[:, -1:, ...]

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
            self._next_position -= trim

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, ...])

        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(
                verify_hidden[:, int(accepted) : int(accepted) + 1, ...]
            )

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            logits_hidden, pre_hc_hidden = self._forward_tokens(
                tokens, hiddens, token_dtype
            )
            self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
            self._seed_hidden = pre_hc_hidden[:, -1:, ...]
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
        """Extend a drafter cache after uniform batched verification."""
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden,
                draft_tokens,
                int(accepted[0]),
                new_tokens[0],
                sampler,
                token_dtype,
                greedy,
            )
            return

        accepted_set = {int(a) for a in accepted}
        if len(accepted_set) != 1:
            raise ValueError("This MTP drafter requires uniform batch acceptance.")
        accepted_i = accepted_set.pop()

        keep_appended = min(accepted_i, self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position -= trim

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, accepted_i):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, ...])

        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )
            token_chunks.append(bonus)
            hidden_chunks.append(verify_hidden[:, accepted_i : accepted_i + 1, ...])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            logits_hidden, pre_hc_hidden = self._forward_tokens(
                tokens, hiddens, token_dtype
            )
            self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
            self._seed_hidden = pre_hc_hidden[:, -1:, ...]
        self._round_appended = 0

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)

        for cache in self._cache:
            if cache.keys is not None:
                cache.keys = cache.keys[keep]
                cache.values = cache.values[keep]

        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]

        value = self._next_position
        if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
            self._next_position = value[keep]

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
            logits_hidden, h_prev = self._forward_tokens(tok, h_prev, token_dtype)
            self._round_appended += 1
            logits = self._lm_head_fn(logits_hidden)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok)

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)
