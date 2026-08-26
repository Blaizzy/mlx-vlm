from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.qwen4_exp.config import SPARSE_ATTENTION
from ....models.qwen4_exp.language import (
    Qwen3_5RotaryEmbedding,
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
)
from .config import Qwen4ExpMTPConfig


class Qwen4ExpMTPDraftModel(nn.Module):
    """The Qwen4-Exp next-token-plus-one head.

    One decoder layer, shaped like the trunk's sparse-attention layers but run
    **dense**: a QSA indexer would need a block cache of its own, and below the
    indexer budget the selection is a no-op anyway, so the draft skips it and the
    indexer weights are dropped at split time.

    The draft has no PLE either, and it shares the target's embeddings and LM
    head, so the whole thing is one layer plus the input fusion.

    Input fusion, unlike DeepSeek-V4's: the hyper-connection row is normed *as a
    whole* before being split into streams -- hence a ``hc_count * hidden_size``
    wide ``pre_fc_norm_hidden`` -- and the embedding term is broadcast across the
    streams. ``fc_embedding(e) + fc_hidden(h)`` is one projection of the
    concatenation, which is how the reference stores it.
    """

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: Qwen4ExpMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Qwen4ExpMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        self.hc_count = text_config.hc_count
        hc_hidden_size = self.hc_count * hidden_size

        # A single sparse-attention layer, with the indexer and the PLE switched
        # off through the config so neither allocates nor expects weights.
        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            layer_types=[SPARSE_ATTENTION],
            ple_layer_ids=[],
            indexer_n_heads=None,
            indexer_kv_heads=None,
            indexer_head_dim=None,
            indexer_budget=None,
            indexer_compress_ratio=None,
        )

        self.pre_fc_norm_embedding = nn.RMSNorm(
            hidden_size, eps=text_config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(
            hc_hidden_size, eps=text_config.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layers = [Qwen4ExpDecoderLayer(layer_config, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            text_config, use_combine=False
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(
                text_config.head_dim
                * text_config.rope_parameters["partial_rotary_factor"]
            ),
            max_position_embeddings=text_config.max_position_embeddings,
            base=text_config.rope_parameters["rope_theta"],
            mrope_section=text_config.rope_parameters["mrope_section"],
        )

        self._input_embed = None
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

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The router and the per-stream injection gates are tiny and scale
            # everything downstream, so they stay at 8 bits like the trunk's.
            if path.endswith(("mlp.gate", "shared_expert_gate", "block_inject_weight")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def bind(self, target_model) -> "Qwen4ExpMTPDraftModel":
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

    def make_cache(self) -> List[KVCache]:
        # A plain KV cache: the draft attention is dense, so there is no indexer
        # and nothing for a `Qwen4ExpAttentionCache`'s pooling half to hold.
        return [KVCache()]

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

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            # `prefill_from_target_hidden` bails out when the runtime hands over
            # post-mixer hidden, which leaves the cache cold for the first round --
            # and `KVCache.state` dereferences `keys` without a None check.
            if not cache.empty():
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
        del shared_kv_states, left_padding
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if position is None:
            position = kv_valid_len
        self._kv_valid_len = kv_valid_len
        self._position = position
        if not self._cache or self._cache[0].offset == 0:
            self._next_position = kv_valid_len

    def _position_ids(self, length: int = 1) -> mx.array:
        """3-row M-RoPE positions for `length` draft tokens.

        Draft tokens are always text, so all three rows advance together; the
        trunk's rope deltas are already folded into the start position handed
        over by `set_shared_kv`.
        """
        start = self._next_position
        pos = mx.arange(length, dtype=mx.int32)
        if isinstance(start, int):
            rows = (pos + start)[None, :]
        elif isinstance(start, mx.array):
            rows = start.astype(mx.int32)[:, None] + pos[None, :]
        else:
            rows = mx.array(start, dtype=mx.int32)[:, None] + pos[None, :]
        if rows.ndim == 2:
            rows = rows[None]
        return mx.broadcast_to(rows, (3, *rows.shape[1:]))

    def _target_streams(self, hidden: mx.array) -> mx.array:
        """The trunk's hyper-connection streams, flattened to one row."""
        expected = self.hc_count * self.args.hidden_size
        if hidden.ndim == 4:
            hidden = hidden.reshape(*hidden.shape[:2], -1)
        if hidden.ndim != 3 or hidden.shape[-1] != expected:
            raise ValueError(
                "Qwen4-Exp MTP expects the target's pre-mixer hyper-connection "
                f"streams, i.e. [batch, tokens, {expected}], got "
                f"{list(hidden.shape)}."
            )
        return hidden

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[KVCache]],
    ) -> Tuple[mx.array, mx.array]:
        hidden = self._target_streams(hidden)
        B, L, _ = hidden.shape

        # One norm over the whole row, then split into streams.
        streams = self.pre_fc_norm_hidden(hidden).reshape(
            B, L, self.hc_count, self.args.hidden_size
        )
        embed = self.fc_embedding(self.pre_fc_norm_embedding(token_embed))
        # fc_embedding(e) + fc_hidden(h) == eh_proj(concat(e, h)).
        h = (embed[:, :, None, :] + self.fc_hidden(streams)).reshape(B, L, -1)

        if cache is None:
            cache = [None]
        position_ids = self._position_ids(L)
        position_embeddings = self.rotary_emb(h, position_ids)
        mask = create_attention_mask(h, cache[0])
        h = self.layers[0](
            h,
            mask=mask,
            cache=cache[0],
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        # `h` is what a chained head would read back, exactly as the trunk hands
        # its streams over before the mixer collapses them.
        return self.hyper_connection_mixer(h), h

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> Tuple[mx.array, mx.array]:
        token_embed = self._input_embed(tokens.astype(token_dtype))
        logits_hidden, streams = self._forward_hidden(
            token_embed,
            hidden[:, : tokens.shape[1], ...],
            self._cache,
        )
        self._next_position = self._next_position + int(tokens.shape[1])
        return logits_hidden, streams

    def _forward_token(
        self,
        tok: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> Tuple[mx.array, mx.array]:
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
        if hidden.shape[-1] != self.hc_count * self.args.hidden_size:
            # The runtime's MTP prefill asks the trunk for `return_hidden` without
            # a layer capture, which yields the post-mixer hidden -- and the mixer
            # is lossy, so the streams cannot be recovered from it. Skip seeding
            # rather than guess: the first draft round then starts from the bonus
            # token with a cold cache, which is correct, only one round slower.
            # Handing the streams over here needs `capture_layer_ids` in
            # `speculative.utils.speculative_prefill_kwargs`.
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)

        # The head predicts from the token the target just produced, so the
        # inputs are shifted by one and the bonus token closes the sequence.
        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        self._next_position = 0
        logits_hidden, streams = self._forward_tokens(
            shifted,
            hidden[:, : shifted.shape[1], ...],
            token_dtype,
        )
        self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
        self._seed_hidden = streams[:, -1:, ...]

    def _extend(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        bonus: Optional[mx.array],
        sampler,
        token_dtype: mx.Dtype,
        greedy: bool,
    ) -> None:
        keep_appended = min(int(accepted), self._round_appended)
        trim = self._round_appended - keep_appended
        if trim > 0:
            for cache in self._cache:
                cache.trim(trim)
            self._next_position = self._next_position - trim

        token_chunks = []
        hidden_chunks = []
        for draft_idx in range(keep_appended, int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, ...])

        if bonus is not None:
            token_chunks.append(bonus)
            hidden_chunks.append(
                verify_hidden[:, int(accepted) : int(accepted) + 1, ...]
            )

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            logits_hidden, streams = self._forward_tokens(tokens, hiddens, token_dtype)
            self._set_seed_from_hidden(logits_hidden[:, -1:, :], sampler, greedy)
            self._seed_hidden = streams[:, -1:, ...]
        self._round_appended = 0

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
        bonus = (
            mx.array([[int(new_tokens[-1])]], dtype=token_dtype) if new_tokens else None
        )
        self._extend(
            verify_hidden,
            draft_tokens,
            int(accepted),
            bonus,
            sampler,
            token_dtype,
            greedy,
        )

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
        """Extend the drafter cache after a batched verify."""
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
                "Qwen4-Exp MTP batched cache update requires uniform acceptance."
            )

        bonus = None
        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )
        self._extend(
            verify_hidden,
            draft_tokens,
            accepted_set.pop(),
            bonus,
            sampler,
            token_dtype,
            greedy,
        )

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

        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])

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
            logits_hidden, h_prev = self._forward_token(tok, h_prev, token_dtype)
            self._round_appended += 1
            logits = self._lm_head_fn(logits_hidden)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok)

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = {
            (key[len("mtp.") :] if key.startswith("mtp.") else key): value
            for key, value in weights.items()
        }
        # The draft attention is dense, so the indexer never runs and its
        # weights are dropped rather than carried as dead parameters.
        weights = {
            key: value
            for key, value in weights.items()
            if ".self_attn.indexer." not in key
        }

        for prefix in (f"layers.{i}.mlp" for i in range(len(self.layers))):
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.up_proj.weight" in weights:
                for name in ("up_proj", "down_proj", "gate_proj"):
                    weights[f"{prefix}.switch_mlp.{name}.weight"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{name}.weight")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights
