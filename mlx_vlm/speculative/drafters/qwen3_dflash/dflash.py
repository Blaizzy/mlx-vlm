from typing import List

import mlx.core as mx
import mlx.nn as nn

from ....models.activations import swiglu
from ....models.cache import BufferedRotatingKVCache, KVCache, RotatingKVCache
from ....models.rope_utils import initialize_rope
from .config import DFlashConfig


def _build_rope(config: DFlashConfig):
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=False,
        scaling_config=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        layer_types = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            if self.sliding_window is None:
                raise ValueError(
                    "DFlash draft config must define sliding_window for sliding layers."
                )
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip

        # Project context and proposal separately so only context KV
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # DFlash denoises the whole proposed block at once, so draft-block
        # self-attention is intentionally non-causal. Sliding layers already
        # limit resident prefix context through the rotating cache above.
        mask = None
        o = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(nn.Module):
    """Qwen3-style gated MLP (matches mlx_lm.models.qwen3.MLP weights)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


# --- DFlash 2 -----------------------------------------------------------------
# Ported from the reference MLX implementation in z-lab/dflash (MIT),
# dflash/model_mlx.py: GroupedDynamicCausalConv, DFlash2DecoderLayer and
# CandidateSelector. Its decoder-layer signature already matches the one above,
# so the round loop in mlx_vlm/speculative/dflash.py is untouched: the selector
# runs inside draft_block and still returns a plain token array.
#
# The drafter only proposes; the target verifies every token. A bug in the conv
# or the selector therefore costs acceptance, never correctness. That is also
# why the reference's distribution-preserving sampling path (q rows) is not
# needed here -- _speculative_walk compares the draft against the target sample.


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """Two-tap causal conv with a static kernel plus an input-dependent part.

    The dynamic part is shared per group (group_size channels each), which is
    why kernel_projection emits 2 * kernel_size * groups values instead of a
    full per-channel matrix.
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else mx.concatenate(
                (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]), axis=1
            )
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(
                hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache):
        residual = x
        x, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(
            self.self_attn(x, x_ctx, rope, cache), kernel
        )
        residual = x
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    """Pick one coherent path through the top-k candidates of a draft block.

    Instead of taking an independent argmax per position, adjacent token pairs
    are scored with a low-rank bilinear form (predecessor codebook x projected
    hidden state x successor codebook) and the best path is traced once,
    left to right.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )

    def select(self, hidden: mx.array, logits: mx.array, anchor_ids: mx.array):
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            selected = mx.argmax(unary[:, position] + edges, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ["full_attention"] * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # DFlash 2 adds per-layer convs and the path selector. Both are driven
        # purely by the drafter config, so a v1 checkpoint builds as before.
        layer_class = (
            DFlash2DecoderLayer if config.conv_kernel_size > 0 else DFlashDecoderLayer
        )
        self.layers = [layer_class(config, i) for i in range(config.num_hidden_layers)]
        self.candidate_selector = (
            CandidateSelector(config) if config.selector_rank > 0 else None
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "DFlashDraftModel":
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
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        window = getattr(self.config, "draft_window_size", None)
        if window is not None and int(window) > 0:
            return [
                BufferedRotatingKVCache(max_size=int(window), buffer_size=64)
                for _ in self.layers
            ]
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "DFlash draft config must define sliding_window for sliding layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_hidden = self._hidden(block, hidden, cache)
        draft_logits = self._logits(draft_hidden[:, 1:])
        if self.candidate_selector is not None:
            # DFlash 2: trace a path through the per-position top-k instead of
            # taking an argmax per position. The anchor is the first block
            # token, i.e. the bonus token that was already accepted.
            return self.candidate_selector.select(
                draft_hidden[:, 1:], draft_logits, block[:, 0]
            ).astype(token_dtype)
        return sampler(draft_logits)

    def _hidden(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        h = self._embed_input_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.norm(h)

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return self.embed_tokens(inputs) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            # DFlash 2 stores the selector codebooks as bare tensors, but they
            # are nn.Embedding here, which expects the .weight suffix.
            if k in (
                "candidate_selector.predecessor_codebook",
                "candidate_selector.successor_codebook",
            ):
                k = f"{k}.weight"
            out[k] = v
        return out


DFlashKVCache = KVCache
