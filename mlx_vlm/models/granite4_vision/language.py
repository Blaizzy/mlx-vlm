from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .config import TextConfig


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        head_dim = config.hidden_size // n_heads
        self.scale = config.attention_multiplier

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=config.attention_bias)

        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class SharedMLP(nn.Module):
    """GraniteMoeHybrid shared MLP with fused gate+up projection."""

    def __init__(self, config: TextConfig):
        super().__init__()
        # Fused gate + up projection: output is 2 * intermediate_size
        self.input_linear = nn.Linear(
            config.hidden_size,
            config.shared_intermediate_size * 2,
            bias=config.mlp_bias,
        )
        self.output_linear = nn.Linear(
            config.shared_intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x) -> mx.array:
        x = self.input_linear(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.output_linear(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.shared_mlp = SharedMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.residual_multiplier = config.residual_multiplier

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * self.residual_multiplier
        r = self.shared_mlp(self.post_attention_layernorm(h))
        out = h + r * self.residual_multiplier
        return out


class Granite(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        # Pop deepstack features (only used during initial prefill)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        # Target layers stored as attribute by get_input_embeddings
        deepstack_target_layers = getattr(self, "_deepstack_target_layers", None)

        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
            # No deepstack for cached token steps
            deepstack_visual_embeds = None
        else:
            h = inputs_embeds

        h = h * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            # Inject deepstack features at target layers
            if (
                deepstack_visual_embeds is not None
                and deepstack_target_layers is not None
                and visual_pos_masks is not None
            ):
                for feat_idx, target_layer in enumerate(deepstack_target_layers):
                    if layer_idx == target_layer:
                        features = deepstack_visual_embeds[feat_idx]
                        # Add features at image positions
                        h = mx.where(
                            visual_pos_masks[..., None],
                            h + features,
                            h,
                        )

            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Granite(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_scaling = config.logits_scaling

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        out = self.model(
            inputs, mask=mask, cache=cache, inputs_embeds=inputs_embeds, **kwargs
        )
        logits = self.lm_head(out)
        logits = logits / self.logits_scaling
        return LanguageModelOutput(logits=logits)

    @staticmethod
    def sanitize(weights):
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
