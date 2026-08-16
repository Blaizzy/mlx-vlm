from typing import Callable, List, Mapping

import mlx.core as mx
import mlx.nn as nn

from ....models.activations import swiglu
from ....models.base import scaled_dot_product_attention
from ....models.cache import KVCache, RotatingKVCache
from ....models.rope_utils import initialize_rope
from .config import (
    DFlashConfig,
    validate_laguna_dflash_target,
    validate_laguna_dflash_weights,
)


def _build_rope(config: DFlashConfig):
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=False,
        scaling_config=config.rope_parameters,
        max_position_embeddings=config.max_position_embeddings,
    )


def _causal_block_mask(query_length: int, context_length: int) -> mx.array:
    context = mx.ones((query_length, context_length), dtype=mx.bool_)
    proposal = mx.tril(mx.ones((query_length, query_length), dtype=mx.bool_))
    return mx.concatenate([context, proposal], axis=-1)


def _trim_dflash_context(context, cache, sliding_window):
    context_length = context.shape[1]
    if context_length >= sliding_window:
        skip = context_length - (sliding_window - 1)
        context = context[:, skip:]
        context_length = context.shape[1]
        cache.offset += skip
    return context, context_length


def _project_dflash_qkv(
    qkv,
    context_qkv,
    batch,
    length,
    context_length,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    queries = qkv[..., :q_size]
    proposal_keys = qkv[..., q_size : q_size + kv_size]
    proposal_values = qkv[..., q_size + kv_size :]
    context_keys = context_qkv[..., q_size : q_size + kv_size]
    context_values = context_qkv[..., q_size + kv_size :]

    queries = queries.reshape(batch, length, num_attention_heads, head_dim).transpose(
        0, 2, 1, 3
    )
    context_keys = context_keys.reshape(
        batch, context_length, num_key_value_heads, head_dim
    ).transpose(0, 2, 1, 3)
    proposal_keys = proposal_keys.reshape(
        batch, length, num_key_value_heads, head_dim
    ).transpose(0, 2, 1, 3)
    context_values = context_values.reshape(
        batch, context_length, num_key_value_heads, head_dim
    ).transpose(0, 2, 1, 3)
    proposal_values = proposal_values.reshape(
        batch, length, num_key_value_heads, head_dim
    ).transpose(0, 2, 1, 3)
    return queries, context_keys, proposal_keys, context_values, proposal_values


def _normalize_dflash_qkv(projected, q_norm, k_norm):
    queries, context_keys, proposal_keys, context_values, proposal_values = projected
    return (
        q_norm(queries.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3),
        k_norm(context_keys.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3),
        k_norm(proposal_keys.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3),
        context_values,
        proposal_values,
    )


def _apply_dflash_attention(attention, x, projected, rope, cache):
    queries, context_keys, proposal_keys, context_values, proposal_values = projected
    offset = cache.offset + int(getattr(cache, "pos_shift", 0))
    queries = rope(queries, offset=offset + context_keys.shape[2])
    context_keys = rope(context_keys, offset=offset)
    proposal_keys = rope(proposal_keys, offset=offset + context_keys.shape[2])
    cached_keys, cached_values = cache.update_and_fetch(context_keys, context_values)
    keys = mx.concatenate([cached_keys, proposal_keys], axis=2)
    values = mx.concatenate([cached_values, proposal_values], axis=2)
    mask = (
        _causal_block_mask(queries.shape[2], cached_keys.shape[2])
        if attention.causal
        else None
    )
    output = scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=attention.scale, mask=mask
    )
    batch, length, _ = x.shape
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    gate = nn.softplus(attention.g_proj(x).astype(mx.float32)).astype(output.dtype)
    output = (
        output.reshape(batch, length, attention.n_heads, attention.head_dim)
        * gate[..., None]
    ).reshape(batch, length, -1)
    return attention.o_proj(output)


class LagunaDFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.causal = config.causal
        dim = config.hidden_size
        qkv_size = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(dim, qkv_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        rope,
        cache: RotatingKVCache,
    ) -> mx.array:
        batch, length, _ = x.shape
        context, context_length = _trim_dflash_context(
            context, cache, self.sliding_window
        )
        qkv = self.qkv_proj(x)
        context_qkv = self.qkv_proj(context)
        projected = _project_dflash_qkv(
            qkv,
            context_qkv,
            batch,
            length,
            context_length,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        )
        projected = _normalize_dflash_qkv(
            projected,
            self.q_norm,
            self.k_norm,
        )
        return _apply_dflash_attention(self, x, projected, rope, cache)


class LagunaDFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LagunaDFlashAttention(config, layer_idx)
        self.mlp = LagunaDFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, context, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), context, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaDFlashMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def sanitize_laguna_dflash_weights(
    weights: Mapping[str, mx.array], config: DFlashConfig
) -> dict[str, mx.array]:
    """Normalize published or HF-prefixed keys and fuse split QKV weights."""
    normalized = {}
    split_qkv: dict[int, dict[str, mx.array]] = {}
    for key, value in weights.items():
        key = key.removeprefix("model.")
        parts = key.split(".")
        if len(parts) >= 5 and parts[0] == "layers" and parts[2] == "self_attn":
            layer = int(parts[1])
            projection = parts[3]
            if projection in {"q_proj", "k_proj", "v_proj"}:
                split_qkv.setdefault(layer, {})[projection] = value
                continue
        if key in normalized:
            raise ValueError(
                f"Duplicate Laguna DFlash weight key after sanitization: {key}"
            )
        normalized[key] = value

    for layer, projections in split_qkv.items():
        missing = {"q_proj", "k_proj", "v_proj"} - set(projections)
        if missing:
            raise ValueError(
                f"Laguna DFlash layer {layer} has incomplete split QKV weights: "
                f"missing={sorted(missing)}."
            )
        key = f"layers.{layer}.self_attn.qkv_proj.weight"
        if key in normalized:
            raise ValueError(
                f"Laguna DFlash has both fused and split QKV weights for layer {layer}."
            )
        normalized[key] = mx.concatenate(
            [projections["q_proj"], projections["k_proj"], projections["v_proj"]],
            axis=0,
        )

    validate_laguna_dflash_weights(normalized, config)
    return normalized


class LagunaDFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.layers = [
            LagunaDFlashDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.aux_hidden_norms = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in config.target_layer_ids
        ]
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model, tokenizer=None) -> "LagunaDFlashDraftModel":
        inner = getattr(target_model, "model", target_model)
        language_model = getattr(target_model, "language_model", None)
        if language_model is not None:
            inner = getattr(language_model, "model", language_model)
        if not hasattr(inner, "embed_tokens"):
            raise AttributeError(
                f"Cannot find target embed_tokens in {type(target_model).__name__}."
            )

        target_config = getattr(language_model, "config", None)
        if target_config is None:
            target_config = getattr(target_model, "config", None)
        target_layers = getattr(inner, "layers", None)
        target_layer_count = len(target_layers) if target_layers is not None else None
        tokenizer_length = len(tokenizer) if tokenizer is not None else None
        if tokenizer_length is None and target_config is not None:
            tokenizer_length = getattr(target_config, "vocab_size", None)
        validate_laguna_dflash_target(
            self.config,
            target_model_config=target_config,
            target_model_layer_count=target_layer_count,
            target_tokenizer_length=tokenizer_length,
        )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(self.embed_tokens, "embed_scale", 1.0)
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        target_inner = getattr(target, "model", target)
        target_layers = getattr(target_inner, "layers", None)
        validate_laguna_dflash_target(
            self.config,
            target_model_config=getattr(target, "config", None),
            target_model_layer_count=(
                len(target_layers) if target_layers is not None else None
            ),
        )

    def make_cache(self) -> List[KVCache]:
        return [
            RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
            for _ in self.layers
        ]

    def reset(self, target_model, tokenizer=None) -> List[KVCache]:
        self.bind(target_model, tokenizer=tokenizer)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        if self.embed_tokens is None:
            raise RuntimeError("bind(target_model) must run before DFlash generation.")
        return self.embed_tokens(inputs) * self.embed_scale

    def _combine_hidden(self, target_hidden: mx.array) -> mx.array:
        expected = len(self.config.target_layer_ids) * self.config.hidden_size
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                "Laguna DFlash target hidden state shape mismatch: expected last "
                f"dimension {expected}, got {target_hidden.shape[-1]}."
            )
        slices = mx.split(target_hidden, len(self.aux_hidden_norms), axis=-1)
        normalized = [norm(value) for norm, value in zip(self.aux_hidden_norms, slices)]
        return self.hidden_norm(self.fc(mx.concatenate(normalized, axis=-1)))

    def _hidden(self, inputs, target_hidden, cache):
        h = self._embed_input_tokens(inputs)
        context = self._combine_hidden(target_hidden)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, context, self.rope, layer_cache)
        return self.norm(h)

    def _logits(self, hidden):
        if self.lm_head is None:
            raise RuntimeError("bind(target_model) must run before DFlash generation.")
        return self.lm_head(hidden)

    def __call__(self, inputs, target_hidden, cache):
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = self.config.mask_token_id
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)], dtype=token_dtype
            )
        else:
            batch = last_bonus.shape[0]
            masks = mx.full((batch, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        return sampler(self._logits(self._hidden(block, hidden, cache)[:, 1:]))

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        return sanitize_laguna_dflash_weights(weights, self.config)

    def load_weights(self, weights, strict=True):
        normalized = sanitize_laguna_dflash_weights(dict(weights), self.config)
        return super().load_weights(list(normalized.items()), strict=strict)


DFlashKVCache = KVCache
Model = LagunaDFlashDraftModel


__all__ = [
    "DFlashKVCache",
    "LagunaDFlashDraftModel",
    "LagunaDFlashAttention",
    "sanitize_laguna_dflash_weights",
    "Model",
]
