from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import RMSNorm

from ....models.gemma4.language import MLP as Gemma4MLP
from ....models.rope_utils import initialize_rope
from ..dspark.dspark import DSparkDraftModel
from ..qwen3_dflash.dflash import DFlashAttention
from .config import Gemma4DsparkConfig


def _build_gemma4_rope(config: Gemma4DsparkConfig):
    """Build RoPE with Gemma 4's per-layer-type parameters.

    The generic DFlash helper keys off ``head_dim`` and a flat ``rope_theta``.
    Gemma 4 publishes theta and ``partial_rotary_factor`` per layer type and
    rotates ``global_head_dim`` on full-attention layers, which is what every
    draft layer here is.
    """
    layer_types = config.layer_types or ["full_attention"] * config.num_hidden_layers
    layer_key = (
        "sliding_attention"
        if layer_types[0] == "sliding_attention"
        else "full_attention"
    )
    params = dict(config.rope_parameters.get(layer_key) or {})
    dims = (
        config.global_head_dim
        if layer_key == "full_attention" and config.global_head_dim
        else config.head_dim
    )
    return initialize_rope(
        dims=dims,
        traditional=config.rope_traditional,
        base=params.get("rope_theta", 10000.0),
        scaling_config=params,
        max_position_embeddings=config.max_position_embeddings,
    )


class Gemma4DSparkAttention(DFlashAttention):
    """DFlash context/proposal attention with Gemma 4 projection geometry.

    Gemma 4 full-attention layers use ``global_head_dim`` rather than
    ``head_dim``, a unit softmax scale, and share one projection between keys
    and values, so the draft checkpoint carries no ``v_proj``.
    """

    def __init__(self, config: Gemma4DsparkConfig, layer_idx: int):
        nn.Module.__init__(self)
        layer_types = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.use_k_eq_v = bool(config.attention_k_eq_v) and not self.is_sliding
        self.n_heads = config.num_attention_heads
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads
        self.scale = 1.0

        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _project_kv(self, x: mx.array):
        keys = self.k_proj(x)
        return (keys, keys) if self.use_k_eq_v else (keys, self.v_proj(x))


class Gemma4DSparkDecoderLayer(nn.Module):
    """Gemma 4 sandwich-norm block driven by the DFlash context split."""

    def __init__(self, config: Gemma4DsparkConfig, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Gemma4DSparkAttention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache) -> mx.array:
        residual = x
        h = self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        h = residual + self.post_attention_layernorm(h)

        residual = h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        h = residual + h
        return h * self.layer_scalar


class Gemma4DSparkDraftModel(DSparkDraftModel):
    """DSpark drafter for Gemma 4 targets.

    Unlike DFlash, the published checkpoint ships untied ``embed_tokens`` and
    ``lm_head``, so binding to a target must not replace them.
    """

    layer_class = Gemma4DSparkDecoderLayer

    def __init__(self, config: Gemma4DsparkConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_scale = config.hidden_size**0.5
        self.rope = _build_gemma4_rope(config)

    def bind(self, target_model) -> "Gemma4DSparkDraftModel":
        embed_tokens, lm_head, embed_scale = (
            self.embed_tokens,
            self.lm_head,
            self.embed_scale,
        )
        super().bind(target_model)
        self.embed_tokens, self.lm_head, self.embed_scale = (
            embed_tokens,
            lm_head,
            embed_scale,
        )
        return self

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        return super().sanitize(
            {key.removeprefix("model."): value for key, value in weights.items()}
        )


Model = Gemma4DSparkDraftModel
ModelConfig = Gemma4DsparkConfig


__all__ = [
    "Gemma4DSparkAttention",
    "Gemma4DSparkDecoderLayer",
    "Gemma4DSparkDraftModel",
    "Model",
    "ModelConfig",
]
