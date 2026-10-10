from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from transformers import AutoTokenizer

from mlx_vlm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.rope_utils import initialize_rope


@dataclass(frozen=True, slots=True)
class ErnieImageTextConfig:
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    rope_theta: float = 1_000_000.0
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32,
            "beta_slow": 1,
            "llama_4_scaling_beta": 0.1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ErnieImageTextConfig":
        nested = value.get("text_config")
        if isinstance(nested, dict):
            value = nested
        allowed = {field.name for field in fields(cls)}
        data = {key: item for key, item in value.items() if key in allowed}
        rope_parameters = value.get("rope_parameters") or value.get("rope_scaling")
        if isinstance(rope_parameters, dict):
            data["rope_parameters"] = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                data["rope_theta"] = float(rope_parameters["rope_theta"])
        return cls(**data)


class ErnieImageTextAttention(nn.Module):
    def __init__(self, config: ErnieImageTextConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.rope = initialize_rope(
            config.head_dim,
            config.rope_theta,
            False,
            config.rope_parameters,
            config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, sequence, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch, sequence, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        attended = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.o_proj(attended)


class ErnieImageTextMLP(nn.Module):
    def __init__(self, config: ErnieImageTextConfig) -> None:
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
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class ErnieImageTextBlock(nn.Module):
    def __init__(self, config: ErnieImageTextConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ErnieImageTextAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = ErnieImageTextMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask, cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class ErnieImageTextEncoder(nn.Module):
    def __init__(self, config: ErnieImageTextConfig | None = None) -> None:
        super().__init__()
        self.config = config or ErnieImageTextConfig()
        config = self.config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            ErnieImageTextBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        normalize: bool = False,
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        layers = self.layers if normalize else self.layers[:-1]
        if cache is None:
            cache = [None] * len(layers)
        if len(cache) != len(layers):
            raise ValueError(
                f"Expected {len(layers)} text cache entries, got {len(cache)}"
            )
        mask = create_attention_mask(hidden_states, cache[0])
        for layer, layer_cache in zip(layers, cache):
            hidden_states = layer(hidden_states, mask, layer_cache)
        return self.norm(hidden_states) if normalize else hidden_states


class ErnieImageTokenizer:
    def __init__(self, model_path: str | Path, *, max_length: int = 2048) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).expanduser() / "tokenizer"),
            local_files_only=True,
            use_fast=True,
        )

    def encode(self, prompt: str) -> mx.array:
        tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"])

    def count_tokens(self, prompt: str) -> int:
        return int(self.encode(prompt).shape[1])


__all__ = [
    "ErnieImageTextConfig",
    "ErnieImageTextEncoder",
    "ErnieImageTokenizer",
]
