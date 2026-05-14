from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask, scaled_dot_product_attention
from ....models.cache import KVCache
from .config import Eagle3Config, TextConfig


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: TextConfig, input_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5

        in_dim = input_size or config.hidden_size
        self.q_proj = nn.Linear(
            in_dim,
            self.n_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            in_dim,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            in_dim,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_offset: Any = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if position_offset is None:
            position_offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=position_offset)
        keys = self.rope(keys, offset=position_offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Eagle3FirstLayer(nn.Module):
    """EAGLE-3 first layer: attention sees concat(token_embed, draft_hidden)."""

    def __init__(self, config: TextConfig, norm_before_residual: bool):
        super().__init__()
        self.config = config
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, input_size=2 * config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def __call__(
        self,
        embeds: mx.array,
        hidden: mx.array,
        mask: Optional[mx.array],
        cache: Optional[KVCache],
        position_offset: Any,
    ) -> mx.array:
        embeds = self.input_layernorm(embeds)
        hidden_normed = self.hidden_norm(hidden)
        residual = hidden_normed if self.norm_before_residual else hidden

        h = mx.concatenate([embeds, hidden_normed], axis=-1)
        h = self.self_attn(h, mask=mask, cache=cache, position_offset=position_offset)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def __call__(
        self,
        hidden: mx.array,
        mask: Optional[mx.array],
        cache: Optional[KVCache],
        position_offset: Any,
    ) -> mx.array:
        h = self.self_attn(
            self.input_layernorm(hidden),
            mask=mask,
            cache=cache,
            position_offset=position_offset,
        )
        hidden = hidden + h
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Eagle3DraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        text_config = config.transformer_layer_config
        self.hidden_size = text_config.hidden_size
        self.target_hidden_size = int(config.target_hidden_size or self.hidden_size)
        self.draft_vocab_size = int(config.draft_vocab_size)
        self.target_vocab_size = int(text_config.vocab_size)
        self.uses_draft_vocab = self.draft_vocab_size != self.target_vocab_size

        self.embed_tokens = nn.Embedding(self.target_vocab_size, self.hidden_size)
        self.fc = nn.Linear(3 * self.target_hidden_size, self.hidden_size, bias=False)
        self.layers = [
            Eagle3FirstLayer(text_config, config.norm_before_residual),
            *[
                DecoderLayer(text_config)
                for _ in range(1, int(text_config.num_hidden_layers))
            ],
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)
        if config.norm_before_fc:
            self.input_norm = nn.RMSNorm(
                3 * self.target_hidden_size, eps=text_config.rms_norm_eps
            )
        else:
            self.input_norm = None

        if config.tie_word_embeddings and not self.uses_draft_vocab:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                self.hidden_size, self.draft_vocab_size, bias=False
            )

        self.d2t = (
            mx.zeros((self.draft_vocab_size,), dtype=mx.int32)
            if self.uses_draft_vocab
            else None
        )

        self._cache: List[KVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position: Any = 1
        self._round_appended = 0
        self._adaptive_block_size: Optional[int] = None

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "Eagle3DraftModel":
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

        if inner is not None:
            target_embed = inner.embed_tokens
            weight = getattr(target_embed, "weight", None)
            if weight is not None and tuple(weight.shape) == tuple(
                self.embed_tokens.weight.shape
            ):
                self.embed_tokens = target_embed
        return self

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache()
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 1
        self._round_appended = 0
        self._adaptive_block_size = None
        return self._cache

    def _prepare_target_hidden(self, hidden: mx.array) -> mx.array:
        if hidden.shape[-1] == self.hidden_size:
            return hidden
        if self.input_norm is not None:
            hidden = self.input_norm(hidden)
        return self.fc(hidden)

    def _logits(self, hidden: mx.array) -> mx.array:
        hidden = self.norm(hidden)
        if self.lm_head is None:
            return self.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def _draft_to_target(self, draft_ids: mx.array, token_dtype: mx.Dtype) -> mx.array:
        draft_ids = draft_ids.astype(mx.int32)
        if self.d2t is not None:
            draft_ids = draft_ids + self.d2t[draft_ids]
        return draft_ids.astype(token_dtype)

    def _sample(self, logits: mx.array, sampler, token_dtype: mx.Dtype, greedy: bool):
        draft_ids = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        return self._draft_to_target(draft_ids, token_dtype)

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        hidden = self._prepare_target_hidden(hidden)
        embeds = self.embed_tokens(tokens.astype(token_dtype))

        h = hidden
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = self._cache[layer_idx] if self._cache else None
            mask = create_attention_mask(h, layer_cache) if layer_cache else "causal"
            if isinstance(self._next_position, int):
                position_offset = self._next_position
            else:
                position_offset = self._next_position
            if layer_idx == 0:
                h = layer(embeds, h, mask, layer_cache, position_offset)
            else:
                h = layer(h, mask, layer_cache, position_offset)

        steps = int(tokens.shape[1])
        self._next_position = (
            self._next_position + steps
            if isinstance(self._next_position, int)
            else self._next_position + steps
        )
        return h

    def _set_seed_from_hidden(self, hidden: mx.array, sampler, token_dtype, greedy):
        logits = self._logits(hidden)
        self._seed_token = self._sample(logits, sampler, token_dtype, greedy)
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
        self._next_position = 1
        h = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], :],
            token_dtype,
        )
        self._set_seed_from_hidden(h[:, -1:, :], sampler, token_dtype, greedy)

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
            tok = self._sample(self._logits(h_prev), sampler, token_dtype, greedy)
            tokens.append(tok)

        return mx.concatenate(tokens, axis=1)

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
        trim = self._round_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )

        accepted = int(accepted)
        token_chunks = []
        hidden_chunks = []
        if accepted > 0:
            token_chunks.append(draft_tokens[:, :accepted])
            hidden_chunks.append(verify_hidden[:, :accepted, :])

        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, accepted : accepted + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, token_dtype, greedy)
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
            raise ValueError(
                "EAGLE-3 batched cache update requires uniform acceptance."
            )
        accepted_i = accepted_set.pop()

        trim = self._round_appended
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
        if accepted_i > 0:
            token_chunks.append(draft_tokens[:, :accepted_i])
            hidden_chunks.append(verify_hidden[:, :accepted_i, :])

        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )
            token_chunks.append(bonus)
            hidden_chunks.append(verify_hidden[:, accepted_i : accepted_i + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, token_dtype, greedy)
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
        if isinstance(self._next_position, mx.array) and self._next_position.ndim > 0:
            self._next_position = self._next_position[keep]

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for key, value in weights.items():
            if key == "t2d" or key.startswith("verifier_"):
                continue
            if "rotary_emb.inv_freq" in key:
                continue
            if key == "input_norm.weight" and self.input_norm is None:
                continue
            if key == "d2t":
                value = value.astype(mx.int32)
            if key == "lm_head.weight" and self.lm_head is None:
                continue
            out[key] = value
        return out
