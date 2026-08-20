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
        self.has_explicit_attention_mode = config.is_causal is not None
        self.is_causal = bool(config.is_causal)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding and self.has_explicit_attention_mode:
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
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # DFlash 2 explicitly declares its attention mode. Its non-causal
        # proposal block stays global, while the oldest prefix keys fall out
        # of the sliding window as the proposal position advances. Legacy
        # checkpoints without ``is_causal`` retain the original unmasked path.
        mask = None
        if self.is_causal:
            query = ctx_len + mx.arange(L)[:, None]
            key = mx.arange(ctx_len + L)[None]
            mask = key <= query
        if self.is_sliding and self.has_explicit_attention_mode:
            query = ctx_len + mx.arange(L)[:, None]
            key = mx.arange(ctx_len + L)[None]
            context = (key < ctx_len) & (query - key < self.sliding_window)
            block = key >= ctx_len
            if self.is_causal:
                block = block & (key <= query)
            mask = context | block
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


@mx.compile
def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    """Apply DFlash 2's grouped, token-conditioned causal convolution."""
    batch, length, hidden_size = hidden.shape
    if group_size <= 0 or hidden_size % group_size:
        raise ValueError(
            "DFlash 2 conv_group_size must be positive and divide hidden_size; "
            f"got hidden_size={hidden_size}, conv_group_size={group_size}."
        )
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        if offset == 0:
            values = blocks
        else:
            values = mx.concatenate(
                [mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]], axis=1
            )
        kernel = base[offset].reshape(1, 1, groups, group_size)
        output = output + (kernel + dynamic[:, :, offset]) * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(
                f"DFlash 2 conv_kernel_size must be positive; got {kernel_size}."
            )
        if group_size <= 0 or hidden_size % group_size:
            raise ValueError(
                "DFlash 2 conv_group_size must be positive and divide hidden_size; "
                f"got hidden_size={hidden_size}, conv_group_size={group_size}."
            )
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden: mx.array):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.base_kernel.shape[1], groups
        )
        return (
            _grouped_dynamic_convolve(
                hidden,
                dynamic[..., 0, :, :],
                self.base_kernel[0],
                self.group_size,
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
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
    """Trace a coherent path through per-position top-k candidates."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        if config.selector_rank <= 0 or config.selector_top_k <= 0:
            raise ValueError(
                "DFlash 2 requires positive selector_rank and selector_top_k."
            )
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(
            config.vocab_size, config.selector_rank
        )
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )

    def select(self, hidden, logits, anchor_ids, temperature: float = 0.0):
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        probability_rows = []
        for position in range(projected.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * projected[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            if temperature > 0:
                probabilities = mx.softmax(
                    scores.astype(mx.float32) * (1.0 / temperature), axis=-1
                )
                selected = mx.random.categorical(mx.log(probabilities))
                probability_rows.append(probabilities)
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return (
            mx.stack(path, axis=1),
            candidates,
            mx.stack(probability_rows, axis=1) if probability_rows else None,
        )


class DFlashDraftModel(nn.Module):
    layer_class = DFlashDecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ["full_attention"] * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            self.layer_class(config, i) for i in range(config.num_hidden_layers)
        ]
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
        self.embed_scale = (
            getattr(
                self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
            )
            * self.config.input_embedding_scale
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
        if float(getattr(sampler, "temperature", 0.0)) <= 0:
            return sampler(draft_logits)
        draft_logprobs = draft_logits - mx.logsumexp(
            draft_logits, axis=-1, keepdims=True
        )
        return sampler(draft_logprobs)

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
        logits = self.lm_head(hidden) * self.config.output_multiplier
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
            out[k] = v
        return out


class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer
    prefer_requested_block_size = True
    supports_greedy_selector = True
    supports_rejection_sampling = True

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def propose_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        temperature: float,
        token_dtype: mx.Dtype = mx.int32,
    ):
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            batch = last_bonus.shape[0]
            masks = mx.full((batch, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_hidden = self._hidden(block, hidden, cache)[:, 1:]
        return self.candidate_selector.select(
            draft_hidden,
            self._logits(draft_hidden),
            block[:, 0],
            temperature,
        )

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        *,
        greedy_sampling: bool = False,
    ) -> mx.array:
        temperature = (
            0.0 if greedy_sampling else float(getattr(sampler, "temperature", 1.0))
        )
        tokens, _, _ = self.propose_block(
            last_bonus,
            hidden,
            cache,
            block_size,
            temperature,
            token_dtype,
        )
        return tokens

    def sanitize(self, weights: dict) -> dict:
        out = super().sanitize(weights)
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            if key in out:
                out[f"{key}.weight"] = out.pop(key)
        return out


DFlashKVCache = KVCache
