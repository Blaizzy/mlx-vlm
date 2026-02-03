"""GLM-OCR Language Model.

Simplified language model for GLM-OCR 0.9B parameters.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .config import TextConfig


def _compute_default_rope_parameters(
    config: Optional[TextConfig] = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:
    """Compute RoPE parameters for GLM-OCR."""
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
    )
    return inv_freq, attention_factor


class GLMOCRRotaryEmbedding(nn.Module):
    """Rotary embeddings for GLM-OCR."""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.rope_kwargs = {}
        
        inv_freq, self.attention_scaling = _compute_default_rope_parameters(
            config, **self.rope_kwargs
        )
        self.inv_freq = inv_freq
        
        # M-RoPE parameters
        self.mrope_section = config.rope_scaling.get("mrope_section", [16, 24, 24])
        self._freqs_cache = None
        
    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary embeddings."""
        if self._freqs_cache is None or self._freqs_cache.shape[1] < position_ids.max() + 1:
            seq_len = int(position_ids.max()) + 1
            freqs = []
            for i in range(3):
                section = self.mrope_section[i]
                freq = mx.outer(
                    position_ids[:, i].astype(mx.float32),
                    self.inv_freq[i * len(self.inv_freq) // 3 : (i + 1) * len(self.inv_freq) // 3]
                )
                freqs.append(freq)
            self._freqs_cache = mx.stack(freqs, axis=1)
        
        # Get cached frequencies
        cos = mx.cos(self._freqs_cache)
        sin = mx.sin(self._freqs_cache)
        
        return cos, sin


class GLMOCRAttention(nn.Module):
    """Multi-head attention for GLM-OCR."""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = GLMOCRRotaryEmbedding(config)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[mx.array, Optional[Any]]:
        """Forward pass."""
        bsz, q_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = mx.arange(q_len, dtype=mx.int64)[None, :]
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # Expand key/value heads for GQA
        key_states = mx.repeat(key_states, self.num_heads // self.num_key_value_heads, axis=1)
        value_states = mx.repeat(value_states, self.num_heads // self.num_key_value_heads, axis=1)
        
        # Attention
        attn_output = scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, self.attention_dropout
        )
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


class GLMOCRMLP(nn.Module):
    """MLP for GLM-OCR."""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = nn.SiLU()
        
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass."""
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(self.activation_fn(gate) * up)


class GLMOCRDecoderLayer(nn.Module):
    """Decoder layer for GLM-OCR."""
    
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GLMOCRAttention(config)
        self.mlp = GLMOCRMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[mx.array, Optional[Any]]:
        """Forward pass."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class GLMOCRModel(nn.Module):
    """Base model for GLM-OCR."""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GLMOCRDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        """Forward pass."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            bsz, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            bsz, seq_len = inputs_embeds.shape[:2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if attention_mask is None:
            attention_mask = create_attention_mask(inputs_embeds, None)
        
        hidden_states = inputs_embeds
        
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class LanguageModel(nn.Module):
    """Language model for GLM-OCR."""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = GLMOCRModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> LanguageModelOutput:
        """Forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = mx.nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="mean",
            )
        
        return LanguageModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
        )
