from typing import Callable, List, Mapping

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.rope_utils import initialize_rope
from .config import MuseGlimmerAssistantConfig


def _bidirectional_block_mask(query_length: int, context_length: int) -> mx.array:
    """Block queries attend to all context keys and bidirectionally to each
    other within the block.

    With the drafter KV cache capped at ``sliding_window - 1`` and the block at
    most ``sliding_window`` long, the sliding window is always satisfied by
    construction, so a full-ones mask is correct.
    """
    context = mx.ones((query_length, context_length), dtype=mx.bool_)
    proposal = mx.ones((query_length, query_length), dtype=mx.bool_)
    return mx.concatenate([context, proposal], axis=-1)


def _build_rope(config: MuseGlimmerAssistantConfig):
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=False,
        scaling_config={"rope_type": "default", "rope_theta": config.rope_theta},
        max_position_embeddings=config.max_position_embeddings,
    )


class MuseGlimmerAssistantContextProjection(nn.Module):
    """Fuse the target's aux hidden states (concatenated on features) into one
    hidden-state stream for the drafter (``fc`` + ``output_norm_enc``)."""

    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.output_norm_enc = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, context_hidden_states: mx.array) -> mx.array:
        return self.output_norm_enc(self.fc(context_hidden_states))


class MuseGlimmerAssistantAttention(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        dim = config.hidden_size
        qkv_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, qkv_size, bias=False)
        self.k_proj = nn.Linear(dim, kv_size, bias=False)
        self.v_proj = nn.Linear(dim, kv_size, bias=False)
        self.o_proj = nn.Linear(qkv_size, dim, bias=False)
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
        context_length = context.shape[1]

        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        context_keys = self.k_proj(context).reshape(
            batch, context_length, self.n_kv_heads, self.head_dim
        )
        context_values = self.v_proj(context).reshape(
            batch, context_length, self.n_kv_heads, self.head_dim
        )
        proposal_keys = self.k_proj(x).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        proposal_values = self.v_proj(x).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries.transpose(0, 2, 1, 3))
        context_keys = self.k_norm(context_keys.transpose(0, 2, 1, 3))
        proposal_keys = self.k_norm(proposal_keys.transpose(0, 2, 1, 3))
        context_values = context_values.transpose(0, 2, 1, 3)
        proposal_values = proposal_values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        block_offset = offset + context_length
        queries = rope(queries, offset=block_offset)
        context_keys = rope(context_keys, offset=offset)
        proposal_keys = rope(proposal_keys, offset=block_offset)

        if cache is not None:
            context_keys, context_values = cache.update_and_fetch(
                context_keys, context_values
            )
        keys = mx.concatenate([context_keys, proposal_keys], axis=2)
        values = mx.concatenate([context_values, proposal_values], axis=2)

        mask = _bidirectional_block_mask(length, context_keys.shape[2])
        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MuseGlimmerAssistantMLP(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        dim = config.hidden_size
        self.gate_proj = nn.Linear(dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerAssistantDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MuseGlimmerAssistantAttention(config, layer_idx)
        self.mlp = MuseGlimmerAssistantMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, context, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), context, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MuseGlimmerAssistantDraftModel(nn.Module):
    """Muse-Glimmer DFlash drafter (EAGLE-style block-diffusion).

    A small decoder stack that borrows the target's ``embed_tokens`` / ``lm_head``
    (bound at ``reset`` time) and, conditioned on the target's aux hidden states,
    denoises a block of ``[anchor, mask, ..., mask]`` tokens in a single forward.
    Runs the single-stream (B=1) DFlash loop; batched DFlash for Muse is not wired
    (its rotating/full caches can't express ragged per-row rollback — see
    ``rollback_speculative_cache`` in ``models/muse_glimmer/language.py``).
    """

    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.layers = [
            MuseGlimmerAssistantDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.encoder = MuseGlimmerAssistantContextProjection(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)

        # Bound to the target model at reset() time.
        self.embed_tokens = None
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model, tokenizer=None) -> "MuseGlimmerAssistantDraftModel":
        inner = getattr(target_model, "model", target_model)
        language_model = getattr(target_model, "language_model", None)
        if language_model is not None:
            inner = getattr(language_model, "model", language_model)
        if not hasattr(inner, "embed_tokens"):
            raise AttributeError(
                f"Cannot find target embed_tokens in {type(target_model).__name__}."
            )
        self.embed_tokens = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        target_inner = getattr(target, "model", target)
        target_layers = getattr(target_inner, "layers", None)
        if target_layers is None:
            raise ValueError("Target model has no .layers attribute")
        max_id = max(self.config.target_layer_ids)
        if len(target_layers) <= max_id:
            raise ValueError(
                "Muse-Glimmer drafter needs a target with at least "
                f"{max_id + 1} layers (aux ids {self.config.target_layer_ids}), "
                f"got {len(target_layers)}"
            )

    def make_cache(self) -> List[RotatingKVCache]:
        return [
            RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
            for _ in self.layers
        ]

    def reset(self, target_model, tokenizer=None) -> List[RotatingKVCache]:
        self.bind(target_model, tokenizer=tokenizer)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        if self.embed_tokens is None:
            raise RuntimeError(
                "bind(target_model) must run before Muse DFlash generation."
            )
        # The assistant blocks need the raw embedding lookup WITHOUT the target's
        # NormedEmbedding RMS-norm (the drafter's own input norms normalize
        # internally); mirrors transformers' DflashCandidateGenerator.
        return self.embed_tokens.weight[inputs]

    def _hidden(
        self, inputs: mx.array, target_hidden: mx.array, cache: List[RotatingKVCache]
    ) -> mx.array:
        h = self._embed_input_tokens(inputs)
        context = self.encoder(target_hidden)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, context, self.rope, layer_cache)
        return self.norm(h)

    def _logits(self, hidden: mx.array) -> mx.array:
        if self.lm_head is None:
            raise RuntimeError(
                "bind(target_model) must run before Muse DFlash generation."
            )
        return self.lm_head(hidden)

    def __call__(self, inputs, target_hidden, cache):
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[RotatingKVCache],
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
        logits = self._logits(self._hidden(block, hidden, cache))
        return sampler(logits[:, 1:])

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict:
        return {key.removeprefix("model."): value for key, value in weights.items()}


Model = MuseGlimmerAssistantDraftModel

__all__ = [
    "MuseGlimmerAssistantConfig",
    "MuseGlimmerAssistantDraftModel",
    "Model",
]
