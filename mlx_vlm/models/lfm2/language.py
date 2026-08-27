from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..switch_layers import SwitchGLU
from .config import ModelConfig
from .speculative_verifier import Lfm2ExactSpeculativeVerifier

_EXACT_SPECULATIVE_VERIFIER = Lfm2ExactSpeculativeVerifier()


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries, keys, values = self._prepare_projected_qkv(
            queries, keys, values, cache
        )

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(output)

    def _prepare_projected_qkv(self, queries, keys, values, cache):
        B, L, _ = queries.shape

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        return queries, keys, values


class ShortConv(nn.Module):
    def __init__(self, args, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias
        self.causal = getattr(args, "conv_causal", True)

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: list | None = None,
    ):
        projected = self.in_proj(x)
        return self.out_proj(self._convolve_projected(projected, mask, cache, gdn_sink))

    def _convolve_projected(self, BCx, mask, cache, gdn_sink):
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        if cache is not None:
            previous_state = mx.array(cache[0]) if cache[0] is not None else None
            previous_left_padding = (
                mx.array(cache.left_padding) if cache.left_padding is not None else None
            )
            previous_lengths = (
                mx.array(cache.lengths) if cache.lengths is not None else None
            )
            conv_inputs = Bx
            if cache[0] is None:
                state = mx.zeros(
                    (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=Bx.dtype,
                )
            else:
                state = cache[0]
            Bx = mx.concatenate([state, Bx], axis=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, t)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(Bx, positions, axis=1)
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
            if gdn_sink is not None:
                gdn_sink.append(
                    (
                        self.layer_idx,
                        previous_state,
                        previous_left_padding,
                        previous_lengths,
                        conv_inputs,
                    )
                )
        elif self.causal:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])
        else:
            pad = self.L_cache // 2
            Bx = mx.pad(Bx, [(0, 0), (pad, pad), (0, 0)])

        conv_out = self.conv(Bx)

        y = C * conv_out
        return y


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class GatedMLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_expert_bias = args.use_expert_bias

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)
        if self.use_expert_bias:
            self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, x: mx.array):
        gates = self.gate(x).astype(mx.float32)
        gates = mx.softmax(gates, axis=-1)

        if self.use_expert_bias:
            gates += self.expert_bias

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]

        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True) + 1e-20
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y


def _feed_forward(args, layer_idx: int):
    if getattr(args, "num_experts", 0):
        if layer_idx < args.num_dense_layers:
            return GatedMLP(args.hidden_size, args.intermediate_size)
        return Lfm2MoeSparseMoeBlock(args)
    return MLP(
        dim=args.block_dim,
        ff_dim=args.block_ff_dim,
        multiple_of=args.block_multiple_of,
        auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
        ffn_dim_multiplier=args.block_ffn_dim_multiplier,
    )


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args, layer_idx: int):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = _feed_forward(args, layer_idx)

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: list | None = None,
    ) -> mx.array:
        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
                gdn_sink=gdn_sink,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Lfm2Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: list[int] | None = None,
        hidden_sink: list | None = None,
        gdn_sink: list | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c, gdn_sink=gdn_sink)
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(h)

        return self.embedding_norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        exact_speculative_verify = bool(kwargs.pop("speculative_verify", False))
        if exact_speculative_verify:
            return _EXACT_SPECULATIVE_VERIFIER(
                self,
                inputs,
                cache=cache,
                input_embeddings=inputs_embeds,
                capture_layer_ids=capture_layer_ids,
            )

        hidden_sink: list[mx.array] | None = (
            [] if capture_layer_ids is not None else None
        )

        out = self.model(
            inputs,
            cache,
            inputs_embeds,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(
            logits=out,
            hidden_states=hidden_sink,
            gdn_states=None,
        )

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        if draft_model is None:
            return True
        prefill_kwargs = prefill_kwargs or {}
        if draft_kind == "dflash":
            return prefill_kwargs.get("capture_layer_ids") is not None
        return False

    @staticmethod
    def _restore_conv_cache(
        cache: ArraysCache,
        previous_state: mx.array | None,
        previous_left_padding: mx.array | None,
        previous_lengths: mx.array | None,
        conv_inputs: mx.array,
        valid_lengths: list[int],
        state_size: int,
    ) -> None:
        rows = []
        for row, valid in enumerate(valid_lengths):
            if previous_state is None:
                state = mx.zeros(
                    (1, state_size, conv_inputs.shape[-1]),
                    dtype=conv_inputs.dtype,
                )
            else:
                state = previous_state[row : row + 1]
            committed = conv_inputs[row : row + 1, :valid]
            rows.append(mx.concatenate([state, committed], axis=1)[:, -state_size:])
        cache[0] = mx.concatenate(rows, axis=0)
        cache.left_padding = (
            None
            if previous_left_padding is None
            else previous_left_padding
            - mx.array(valid_lengths, dtype=previous_left_padding.dtype)
        )
        cache.lengths = (
            None
            if previous_lengths is None
            else previous_lengths
            - mx.array(valid_lengths, dtype=previous_lengths.dtype)
        )

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Commit the accepted LFM2 prefix across attention and conv caches."""
        if isinstance(accepted, int):
            accepted = mx.array([accepted], dtype=mx.int32)
        elif not isinstance(accepted, mx.array):
            accepted = mx.array(accepted, dtype=mx.int32)
        if accepted.ndim == 0:
            accepted = accepted.reshape(1)

        accepted_values = [int(value) for value in accepted.tolist()]
        valid_lengths = [value + 1 for value in accepted_values]
        max_accepted = max(accepted_values)
        max_valid = max_accepted + 1
        trim = int(block_size) - max_valid
        is_batch = len(accepted_values) > 1

        for cache in caches:
            if cache is None or isinstance(cache, ArraysCache):
                continue
            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)

            if not is_batch:
                continue
            extra_trim = [max_accepted - value for value in accepted_values]
            prepare = getattr(cache, "prepare", None)
            finalize = getattr(cache, "finalize", None)
            keys = getattr(cache, "keys", None)
            values = getattr(cache, "values", None)
            if any(extra_trim) and callable(prepare) and callable(finalize):
                prepare(right_padding=extra_trim)
                finalize()
                continue

            # Dense caches without row-wise padding retain a common physical
            # width, so clear rejected tails inside the committed window.
            if not hasattr(cache, "_idx"):
                continue
            if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
                raise NotImplementedError(
                    "Ragged LFM2 DSpark rollback currently requires dense KV caches."
                )
            kv_length = int(cache._idx)
            verify_start = kv_length - max_valid
            for row, valid in enumerate(valid_lengths):
                start = verify_start + valid
                if start < kv_length:
                    keys[row, :, start:kv_length, :] = 0
                    values[row, :, start:kv_length, :] = 0

        if gdn_states is None:
            raise RuntimeError(
                "LFM2 speculative rollback requires convolution state captured "
                "with speculative_verify=True."
            )
        for (
            layer_index,
            previous_state,
            previous_left_padding,
            previous_lengths,
            conv_inputs,
        ) in gdn_states:
            cache = caches[layer_index]
            if not isinstance(cache, ArraysCache):
                raise TypeError(
                    "LFM2 convolution rollback expected ArraysCache at layer "
                    f"{layer_index}, got {type(cache).__name__}."
                )
            self._restore_conv_cache(
                cache,
                previous_state,
                previous_left_padding,
                previous_lengths,
                conv_inputs,
                valid_lengths,
                self.args.conv_L_cache - 1,
            )
        return max_accepted

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name and param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.layers
        ]
