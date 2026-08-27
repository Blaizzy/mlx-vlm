from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.longcat_flash_sparse.language import LongcatFlashMLA, LongcatFlashMLP
from .config import LongcatFlashSparseMTPConfig


class LongcatSparseMTPLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = LongcatFlashMLA(args, is_index_owner=False)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = LongcatFlashMLP(args, is_expert=False)

    def __call__(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        mask = (
            create_attention_mask(x, cache, return_array=True)
            if x.shape[1] > 1
            else None
        )
        attn_out, _ = self.self_attn(
            self.input_layernorm(x), mask=mask, latent_cache=cache
        )
        h = x + attn_out
        return h + self.mlp(self.post_attention_layernorm(h))


class LongcatFlashSparseMTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = False

    def __init__(self, config: LongcatFlashSparseMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("LongcatFlashSparseMTPConfig.text_config must be set")
        self.args = text_config

        hidden_size = text_config.hidden_size
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layers = [LongcatSparseMTPLayer(text_config)]
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[KVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0
        self._draft_round = 0
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "LongcatFlashSparseMTPDraftModel":
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

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def reset(
        self, target_model, left_padding: Optional[List[int]] = None
    ) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache(left_padding)
        self._seed_token = None
        self._seed_hidden = None
        self._round_appended = 0
        return self._cache

    def set_shared_kv(
        self,
        shared_kv_states,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del shared_kv_states, kv_offset, position, kv_valid_len, left_padding

    def _forward_hidden(self, token_embed, hidden, cache):
        h = self.eh_proj(
            mx.concatenate([self.hnorm(hidden), self.enorm(token_embed)], axis=-1)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, cache=layer_cache)
        return self.norm(h)

    def _forward_tokens(self, tokens, hidden, token_dtype):
        token_embed = self._input_embed(tokens.astype(token_dtype))
        return self._forward_hidden(token_embed, hidden, self._cache)

    def _forward_token(self, tok, hidden, token_dtype):
        return self._forward_tokens(tok, hidden, token_dtype)

    def _set_seed_from_hidden(self, hidden, sampler, greedy):
        logits = self._lm_head_fn(hidden)
        self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        self._seed_hidden = hidden

    def prefill_from_target_hidden(
        self,
        input_ids,
        hidden,
        bonus_token,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
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
        verify_hidden,
        draft_tokens,
        accepted,
        new_tokens,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
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

    def draft_block(
        self,
        last_bonus,
        hidden,
        cache,
        block_size,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
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
        num_heads = self.args.num_attention_heads
        head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
        for k, v in weights.items():
            key = k
            if key.startswith("model.mtp."):
                key = key[len("model.mtp.") :]
            elif key.startswith("mtp."):
                key = key[len("mtp.") :]
            # single mtp layer: strip the layers.0 prefix onto flat drafter names
            key = key.replace("layers.0.", "", 1)
            # the checkpoint wraps enorm/hnorm as a ".m" submodule
            key = key.replace("enorm.m.", "enorm.").replace("hnorm.m.", "hnorm.")
            # the mtp attention + ffn live under transformer_layer / self_attn
            key = key.replace("transformer_layer.mlp.", "layers.0.mlp.")
            if key.startswith("self_attn."):
                key = "layers.0." + key
            if key.startswith("input_layernorm.") or key.startswith(
                "post_attention_layernorm."
            ):
                key = "layers.0." + key
            # drop the (inactive) mtp indexer + the shared mtp embed_tokens
            if ".indexer." in key or key == "embed_tokens.weight":
                continue
            out[key] = v

        # split the mtp attention kv_b_proj into embed_q / unembed_out (MLA absorb)
        prefix = "layers.0.self_attn"
        if f"{prefix}.kv_b_proj.weight" in out:
            w = out.pop(f"{prefix}.kv_b_proj.weight")
            w = w.reshape(num_heads, head_dim, -1)
            out[f"{prefix}.embed_q.weight"] = mx.contiguous(
                w[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
            )
            out[f"{prefix}.unembed_out.weight"] = mx.contiguous(
                w[:, self.args.qk_nope_head_dim :, :]
            )
        return out


Model = LongcatFlashSparseMTPDraftModel
