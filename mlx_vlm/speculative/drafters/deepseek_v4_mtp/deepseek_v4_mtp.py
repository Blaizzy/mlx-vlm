from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ....models.base import create_attention_mask
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import HyperHead
from ....models.deepseek_v4.language import DeepseekV4Block
from .config import DeepseekV4MTPConfig


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if "decoder.ffn.switch_mlp." in k and k.endswith("_proj")
    }
    mxfp8_modules = {
        k: mxfp8
        for k, _ in flat_modules
        if k in ("e_proj", "h_proj")
        or "decoder.ffn.shared_experts." in k
        or "decoder.attn.w" in k
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **mxfp8_modules,
    }


class DeepseekV4MTPDraftModel(nn.Module):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config: DeepseekV4MTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV4MTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            compress_ratios=[0],
            num_hash_layers=0,
        )
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder = DeepseekV4Block(layer_config, layer_idx=0)
        self.hc_head = HyperHead(text_config)
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[RotatingKVCache] = []
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
        quantization_config = make_quantization_config(self)

        def predicate(path, _):
            return quantization_config.get(path, True)

        return predicate

    def bind(self, target_model) -> "DeepseekV4MTPDraftModel":
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

    def make_cache(self) -> List[RotatingKVCache]:
        return [RotatingKVCache(max_size=self.args.sliding_window)]

    def reset(self, target_model) -> List[RotatingKVCache]:
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
        del shared_kv_states, left_padding
        if kv_valid_len is None:
            kv_valid_len = kv_offset
        if position is None:
            position = kv_valid_len
        self._kv_valid_len = kv_valid_len
        self._position = position
        if not self._cache or self._cache[0].offset == 0:
            self._next_position = kv_valid_len

    def _position_ids(self, step: int = 0, length: int = 1) -> mx.array:
        start = self._next_position
        pos = mx.arange(length, dtype=mx.int32) + step
        if isinstance(start, int):
            return (pos + start)[None, :]
        if isinstance(start, mx.array):
            return start.astype(mx.int32)[:, None] + pos[None, :]
        return mx.array(start, dtype=mx.int32)[:, None] + pos[None, :]

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        if (
            hidden.ndim == 3
            and hidden.shape[-1] == self.args.hc_mult * self.args.hidden_size
        ):
            hidden = hidden.reshape(*hidden.shape[:-1], self.args.hc_mult, -1)
        if hidden.ndim != 4:
            raise ValueError(
                "DeepSeek-V4 MTP expects target hidden shape "
                "[batch, tokens, hc_mult, hidden_size]."
            )
        return hidden

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[RotatingKVCache]],
    ) -> Tuple[mx.array, mx.array]:
        hidden = self._target_hidden(hidden)
        B, L, H, D = hidden.shape
        h_flat = hidden.reshape(B * L * H, D)
        h_proj = self.h_proj(self.hnorm(h_flat)).reshape(B, L, H, D)
        e_proj = self.e_proj(self.enorm(token_embed))[:, :, None, :]
        h = e_proj + h_proj

        if cache is None:
            cache = [None]
        mask = create_attention_mask(
            h[:, :, 0, :],
            cache[0],
            window_size=self.args.sliding_window,
            return_array=True,
        )
        h = self.decoder(
            h,
            mask,
            cache[0],
            tokens,
            position_offset=self._next_position,
        )
        logits_hidden = self.norm(self.hc_head(h))
        return logits_hidden, h

    def _forward_tokens(
        self,
        tokens: mx.array,
        hidden: mx.array,
        token_dtype: mx.Dtype,
    ) -> Tuple[mx.array, mx.array]:
        token_embed = self._input_embed(tokens.astype(token_dtype))
        logits_hidden, pre_hc_hidden = self._forward_hidden(
            token_embed,
            hidden[:, : tokens.shape[1], ...],
            tokens,
            self._cache,
        )
        steps = int(tokens.shape[1])
        self._next_position = (
            self._next_position + steps
            if isinstance(self._next_position, int)
            else self._next_position + steps
        )
        return logits_hidden, pre_hc_hidden

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
            self._next_position = (
                self._next_position - trim
                if isinstance(self._next_position, int)
                else self._next_position - trim
            )

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
        """Extend the DeepSeek-V4 MTP drafter cache after batched verify."""
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
                "DeepSeek-V4 MTP batched cache update requires uniform acceptance."
            )
        accepted_i = accepted_set.pop()

        keep_appended = min(accepted_i, self._round_appended)
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
        weights = dict(weights)
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp.0."):
                k = k[len("mtp.0.") :]
            elif k.startswith("mtp."):
                k = k[len("mtp.") :]
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ("ffn.experts." in wk or ".ffn.experts." in wk)
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            if nk.startswith("attn.") or nk.startswith("attn_norm."):
                nk = f"decoder.{nk}"
            elif nk.startswith("ffn.") or nk.startswith("ffn_norm."):
                nk = f"decoder.{nk}"
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
                    nk = nk.replace(f"hc_{sub}_{param}", f"decoder.{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            nk = nk.replace("hc_head_fn", "hc_head.fn")
            nk = nk.replace("hc_head_base", "hc_head.base")
            nk = nk.replace("hc_head_scale", "hc_head.scale")
            remapped[nk] = v
        weights = remapped

        prefix = "decoder.ffn.experts"
        for src, dst in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            for suffix in ("weight", "scales"):
                key0 = f"{prefix}.0.{src}.{suffix}"
                if key0 in weights:
                    stacked = [
                        weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"decoder.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                        stacked
                    )

        prefix = "decoder.attn.wo_a"
        for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
            if key in weights and weights[key].ndim == 2:
                weights[key] = weights[key].reshape(
                    self.args.o_groups, self.args.o_lora_rank, -1
                )

        return weights
