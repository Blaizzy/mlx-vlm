from collections.abc import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn

from ....models.activations import swiglu
from ....models.cache import BufferedRotatingKVCache, RotatingKVCache
from ....models.rope_utils import initialize_rope
from .config import (
    MuseGlimmerAssistantConfig,
    validate_muse_glimmer_assistant_weights,
    validate_muse_glimmer_target,
)


def _build_rope(config: MuseGlimmerAssistantConfig):
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=False,
        scaling_config=config.rope_parameters,
        max_position_embeddings=config.max_position_embeddings,
    )


def _bidirectional_sliding_mask(
    query_start: int,
    query_length: int,
    key_start: int,
    key_length: int,
    sliding_window: int,
) -> mx.array:
    """Transformers DFlash mask: ``abs(query_pos - key_pos) <= window``."""

    query_positions = mx.arange(query_start, query_start + query_length)[:, None]
    key_positions = mx.arange(key_start, key_start + key_length)[None, :]
    return mx.abs(query_positions - key_positions) <= sliding_window


class MuseGlimmerAssistantAttention(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        context_hidden_states: mx.array,
        rope,
        cache: RotatingKVCache,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        context_length = context_hidden_states.shape[1]

        if context_length > self.sliding_window:
            skip = context_length - self.sliding_window
            context_hidden_states = context_hidden_states[:, skip:]
            context_length = context_hidden_states.shape[1]
            cache.offset += skip

        cache_offset = cache.offset
        queries = self.q_proj(hidden_states)
        kv_hidden_states = mx.concatenate(
            [context_hidden_states, hidden_states], axis=1
        )
        keys = self.k_proj(kv_hidden_states)
        values = self.v_proj(kv_hidden_states)

        queries = self.q_norm(
            queries.reshape(batch, length, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(
                batch,
                context_length + length,
                self.n_kv_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch,
            context_length + length,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        queries = rope(queries, offset=cache_offset + context_length)
        keys = rope(keys, offset=cache_offset)

        context_keys = keys[:, :, :context_length]
        proposal_keys = keys[:, :, context_length:]
        context_values = values[:, :, :context_length]
        proposal_values = values[:, :, context_length:]

        cached_keys, cached_values = cache.update_and_fetch(
            context_keys, context_values
        )
        keys = mx.concatenate([cached_keys, proposal_keys], axis=2)
        values = mx.concatenate([cached_values, proposal_values], axis=2)

        query_start = cache.offset
        key_start = cache.offset - cached_keys.shape[2]
        mask = _bidirectional_sliding_mask(
            query_start,
            length,
            key_start,
            keys.shape[2],
            self.sliding_window,
        )
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MuseGlimmerAssistantMLP(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
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

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


class MuseGlimmerAssistantDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        self.self_attn = MuseGlimmerAssistantAttention(config)
        self.mlp = MuseGlimmerAssistantMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states, context_hidden_states, rope, cache):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            context_hidden_states,
            rope,
            cache,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MuseGlimmerAssistantContextProjection(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        input_size = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(input_size, config.hidden_size, bias=False)
        self.output_norm_enc = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, context_hidden_states: mx.array) -> mx.array:
        return self.output_norm_enc(self.fc(context_hidden_states))


class MuseGlimmerAssistantDraftModel(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = MuseGlimmerAssistantContextProjection(config)
        self.layers = [
            MuseGlimmerAssistantDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.lm_head = None
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    @staticmethod
    def _target_parts(target_model):
        language_model = getattr(target_model, "language_model", target_model)
        inner = getattr(language_model, "model", language_model)
        return language_model, inner

    def validate_target_compatibility(self, target_model) -> None:
        language_model, inner = self._target_parts(target_model)
        layers = getattr(inner, "layers", None)
        validate_muse_glimmer_target(
            self.config,
            target_model_config=getattr(language_model, "config", None),
            target_model_layer_count=(len(layers) if layers is not None else None),
        )

    def bind(self, target_model) -> "MuseGlimmerAssistantDraftModel":
        language_model, inner = self._target_parts(target_model)
        if not hasattr(inner, "embed_tokens"):
            raise AttributeError(
                f"Cannot find target embed_tokens in {type(target_model).__name__}."
            )
        self.validate_target_compatibility(target_model)
        # Muse's target embedding applies a scale-free RMSNorm during normal
        # decoding. DFlash intentionally consumes the raw lookup table.
        self.embed_tokens = inner.embed_tokens
        self.lm_head = getattr(language_model, "lm_head", None)
        if self.lm_head is None:
            raise AttributeError(
                f"Cannot find target lm_head in {type(target_model).__name__}."
            )
        return self

    def make_cache(self) -> list[RotatingKVCache]:
        window = self.config.draft_window_size
        if window is not None and int(window) > 0:
            return [
                BufferedRotatingKVCache(max_size=int(window), buffer_size=64)
                for _ in self.layers
            ]
        return [
            RotatingKVCache(max_size=self.config.sliding_window, keep=0)
            for _ in self.layers
        ]

    def reset(self, target_model) -> list[RotatingKVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        if self.embed_tokens is None:
            raise RuntimeError("bind(target_model) must run before DFlash generation.")
        return self.embed_tokens(inputs)

    def _hidden(self, inputs, target_hidden, cache):
        hidden_states = self._embed_input_tokens(inputs)
        context_hidden_states = self.encoder(target_hidden)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                context_hidden_states,
                self.rope,
                layer_cache,
            )
        return self.norm(hidden_states)

    def _logits(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is None:
            raise RuntimeError("bind(target_model) must run before DFlash generation.")
        # Transformers' DFlash candidate generator calls the target output
        # embedding directly, without Muse's monotonic output scaling/softcap.
        return self.lm_head(hidden_states)

    def __call__(self, inputs, target_hidden, cache):
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: list[RotatingKVCache],
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = self.config.mask_token_id
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
        return sampler(self._logits(self._hidden(block, hidden, cache)[:, 1:]))

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {
            key.removeprefix("model."): value for key, value in weights.items()
        }
        validate_muse_glimmer_assistant_weights(normalized, self.config)
        return normalized

    def load_weights(self, weights, strict=True):
        normalized = self.sanitize(dict(weights))
        return super().load_weights(list(normalized.items()), strict=strict)


DFlashKVCache = RotatingKVCache
Model = MuseGlimmerAssistantDraftModel


__all__ = [
    "DFlashKVCache",
    "Model",
    "MuseGlimmerAssistantAttention",
    "MuseGlimmerAssistantDraftModel",
]
