from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.qwen.text_encoder import Qwen3TextEncoder

from .config import ZImageTextEncoderConfig


class _ComputedRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> tuple[mx.array, mx.array]:
        freqs = position_ids.astype(mx.float32)[..., None] * self._inv_freq
        embedding = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(embedding).astype(x.dtype), mx.sin(embedding).astype(x.dtype)


class ZImageTextEncoder(Qwen3TextEncoder):
    def __init__(self, config: ZImageTextEncoderConfig | None = None) -> None:
        config = config or ZImageTextEncoderConfig()
        self.config = config
        super().__init__(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=config.head_dim,
        )
        self.rotary_emb = _ComputedRotaryEmbedding(
            dim=config.head_dim,
            base=config.rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        *,
        output_penultimate: bool = False,
    ) -> mx.array:
        hidden, hidden_states = super().__call__(
            input_ids,
            attention_mask,
            output_hidden_states=output_penultimate,
        )
        if output_penultimate:
            if hidden_states is None:
                raise RuntimeError("Hidden states not available for prompt embedding")
            return hidden_states[-2]
        return hidden


def sanitize_text_encoder_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    return {
        key.removeprefix("model."): value
        for key, value in weights.items()
        if "rotary_emb" not in key and not key.startswith("lm_head.")
    }


__all__ = ["ZImageTextEncoder", "sanitize_text_encoder_weights"]
