from dataclasses import replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.qwen3_5.language import Qwen3_5DecoderLayer
from .config import Qwen3_5MTPConfig


class Qwen3_5MTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True

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
        self.layers = [
            Qwen3_5DecoderLayer(args=layer_config, layer_idx=0)
            for _ in range(mtp_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None
        self._cache: List[KVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position: Any = 0
        self._round_appended = 0
        self._kv_valid_len: Any = 0
        self._position: Any = 0
        self._draft_round = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "Qwen3_5MTPDraftModel":
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

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def reset(self, target_model) -> List[KVCache]:
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
        if not self._cache or self._cache[0].offset == 0:
            self._next_position = kv_valid_len

    def _draft_start_position(self):
        return self._next_position

    def _position_ids(self, step: int = 0, length: int = 1) -> mx.array:
        start = self._draft_start_position()
        pos = mx.arange(length, dtype=mx.int32) + step
        if isinstance(start, int):
            return (pos + start)[None, :]
        if isinstance(start, mx.array):
            return start.astype(mx.int32)[:, None] + pos[None, :]
        return mx.array(start, dtype=mx.int32)[:, None] + pos[None, :]

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[KVCache]],
        position_ids: mx.array,
    ) -> mx.array:
        h = mx.concatenate(
            [
                self.pre_fc_norm_embedding(token_embed),
                self.pre_fc_norm_hidden(hidden),
            ],
            axis=-1,
        )
        h = self.fc(h)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            mask = (
                create_attention_mask(h, layer_cache)
                if layer_cache is not None
                else ("causal" if h.shape[1] > 1 else None)
            )
            h = layer(h, mask=mask, cache=layer_cache, position_ids=position_ids)
        return self.norm(h)

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        token_embed = (
            self._input_embed(tokens.astype(token_dtype)) * self._input_embed_scale
        )
        steps = int(tokens.shape[1])
        position_ids = self._position_ids(length=steps)
        h = self._forward_hidden(token_embed, hidden, self._cache, position_ids)
        self._next_position = (
            self._next_position + steps
            if isinstance(self._next_position, int)
            else self._next_position + steps
        )
        return h

    def _forward_token(
        self,
        tok: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        return self._forward_tokens(tok, hidden, token_dtype)

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
        h = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], :],
            token_dtype,
        )
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
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )

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
            h_prev = self._forward_token(tok, h_prev, token_dtype)
            self._round_appended += 1
            logits = self._lm_head_fn(h_prev)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok)

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)

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
            if key.startswith("mtp."):
                key = key[len("mtp.") :]
            if any(key.endswith(suffix) for suffix in norm_suffixes):
                if value.ndim == 1 and mx.issubdtype(value.dtype, mx.floating):
                    value = value + 1.0
            out[key] = value
        return out
