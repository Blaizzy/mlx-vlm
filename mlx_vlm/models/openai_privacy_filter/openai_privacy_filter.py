import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..switch_layers import SwitchLinear, _gather_sort, _scatter_unsort
from .config import ModelConfig


@dataclass
class TokenClassifierOutput:
    logits: mx.array


class RMSNorm(nn.Module):
    """RMSNorm matching the upstream model's explicit FP32 normalization."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x * self.weight.astype(mx.float32)).astype(dtype)


class PrivacyFilterRoPE(nn.Module):
    """Interleaved YaRN RoPE used by OpenAI Privacy Filter."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        params = config.rope_parameters
        if params.get("rope_type", "yarn") != "yarn":
            raise ValueError("OpenAI Privacy Filter currently requires YaRN RoPE")

        dims = config.head_dim
        base = float(params.get("rope_theta", 150000.0))
        factor = float(params.get("factor", 32.0))
        original_context = int(params.get("original_max_position_embeddings", 4096))
        beta_fast = float(params.get("beta_fast", 32.0))
        beta_slow = float(params.get("beta_slow", 1.0))

        def correction_dim(rotations: float) -> float:
            return (
                dims
                * math.log(original_context / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if params.get("truncate", True):
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0.0)
        high = min(high, dims - 1.0)
        if low == high:
            high += 0.001

        pos_freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / float(dims))
        ramp = mx.clip(
            (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low),
            0.0,
            1.0,
        )
        extrapolation = 1.0 - ramp
        inv_freq = (1.0 / (factor * pos_freqs)) * (1.0 - extrapolation)
        inv_freq = inv_freq + (1.0 / pos_freqs) * extrapolation

        # mx.fast.rope takes wavelength-like denominators rather than inverse
        # frequencies. Prefixing the name with '_' keeps it out of checkpoints.
        self._freqs = 1.0 / inv_freq
        self._dims = dims
        self._attention_factor = 1.0 + 0.1 * math.log(factor) if factor > 1 else 1.0

    def __call__(self, x: mx.array) -> mx.array:
        if self._attention_factor != 1.0:
            x = x * mx.array(self._attention_factor, dtype=x.dtype)
        return mx.fast.rope(
            x,
            self._dims,
            traditional=True,
            base=None,
            scale=1.0,
            offset=0,
            freqs=self._freqs,
        )


def _bidirectional_local_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    sinks: mx.array,
    radius: int,
    attention_mask: Optional[mx.array],
    chunk_size: int,
) -> mx.array:
    """Run symmetric local attention without materializing a full T x T mask."""

    seq_len = q.shape[2]
    chunks = []
    for query_start in range(0, seq_len, chunk_size):
        query_end = min(query_start + chunk_size, seq_len)
        key_start = max(0, query_start - radius)
        key_end = min(seq_len, query_end + radius)

        query_positions = mx.arange(query_start, query_end)[:, None]
        key_positions = mx.arange(key_start, key_end)[None, :]
        local_mask = mx.abs(query_positions - key_positions) <= radius
        local_mask = local_mask[None, None, :, :]
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, key_start:key_end].astype(mx.bool_)
            local_mask = local_mask & key_mask

        chunks.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, query_start:query_end],
                k[:, :, key_start:key_end],
                v[:, :, key_start:key_end],
                scale=1.0,
                mask=local_mask,
                # MLX SDPA requires sinks to match the attention output type.
                # Keep checkpoint storage FP32 and narrow only at the kernel.
                sinks=sinks.astype(q.dtype),
            )
        )
    return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.chunk_size = config.attention_chunk_size
        self.qk_scale = config.head_dim**-0.25

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.sinks = mx.zeros((config.num_attention_heads,), dtype=mx.float32)
        self.rope = PrivacyFilterRoPE(config)

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(
            batch, seq_len, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).reshape(
            batch, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch, seq_len, self.num_key_value_heads, self.head_dim
        )
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = (self.rope(q) * self.qk_scale).astype(q.dtype)
        k = (self.rope(k) * self.qk_scale).astype(k.dtype)
        output = _bidirectional_local_attention(
            q,
            k,
            v,
            sinks=self.sinks,
            radius=self.sliding_window,
            attention_mask=attention_mask,
            chunk_size=self.chunk_size,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output)


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    gate = mx.clip(gate, a_min=None, a_max=7.0)
    up = mx.clip(up, a_min=-7.0, a_max=7.0)
    return (up + 1.0) * gate * mx.sigmoid(1.702 * gate)


class Experts(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Keep the projections split to match existing mlx-community
        # checkpoints. The Hugging Face checkpoint's fused gate/up tensors are
        # split by Model.sanitize before loading.
        self.gate_proj = SwitchLinear(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            bias=True,
        )
        self.up_proj = SwitchLinear(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            bias=True,
        )
        self.down_proj = SwitchLinear(
            config.intermediate_size,
            config.hidden_size,
            config.num_local_experts,
            bias=True,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # The reference implementation performs selected-expert projections and
        # accumulation in FP32. Mixed BF16/FP32 gather_mm preserves that contract
        # without expanding all 1.4B expert parameters to FP32 in memory.
        x = mx.expand_dims(x.astype(mx.float32), (-2, -3))
        do_sort = indices.size >= 64
        sorted_indices = indices
        inverse_order = None
        if do_sort:
            x, sorted_indices, inverse_order = _gather_sort(x, indices)
        if self.training:
            sorted_indices = mx.stop_gradient(sorted_indices)

        gate = self.gate_proj(x, sorted_indices, sorted_indices=do_sort).astype(
            mx.float32
        )
        up = self.up_proj(x, sorted_indices, sorted_indices=do_sort).astype(mx.float32)
        output = self.down_proj(
            _swiglu(gate, up),
            sorted_indices,
            sorted_indices=do_sort,
        ).astype(mx.float32)

        if do_sort:
            output = _scatter_unsort(output, inverse_order, indices.shape)
        return output.squeeze(-2)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.chunk_size = config.moe_chunk_size
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = Experts(config)

    def _route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if hasattr(self.router, "scales"):
            logits = self.router(x.astype(mx.float32)).astype(mx.float32)
        else:
            logits = x.astype(mx.float32) @ self.router.weight.astype(mx.float32).T
            logits = logits + self.router.bias.astype(mx.float32)

        indices = mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        top_logits = mx.take_along_axis(logits, indices, axis=-1)
        order = mx.argsort(-top_logits, axis=-1)
        indices = mx.take_along_axis(indices, order, axis=-1)
        top_logits = mx.take_along_axis(top_logits, order, axis=-1)
        weights = mx.softmax(top_logits, axis=-1, precise=True) / self.top_k
        return mx.stop_gradient(indices), weights

    def _forward_chunk(self, x: mx.array, output_dtype) -> mx.array:
        indices, weights = self._route(x)
        expert_outputs = self.experts(x, indices)
        output = mx.sum(expert_outputs * weights[..., None], axis=-2)
        return (output * self.top_k).astype(output_dtype)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        chunks = [
            self._forward_chunk(
                flat[start : start + self.chunk_size], hidden_states.dtype
            )
            for start in range(0, flat.shape[0], self.chunk_size)
        ]
        output = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        return output.reshape(original_shape)


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Encoder(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=True)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        if attention_mask is None:
            attention_mask = mask
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        hidden_states = self.model(input_ids, attention_mask, inputs_embeds)
        return TokenClassifierOutput(logits=self.score(hidden_states))

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.endswith(".mlp.experts.gate_up_proj"):
                value = mx.contiguous(value.swapaxes(-1, -2))
                prefix = key.removesuffix("gate_up_proj")
                gate, up = mx.split(value, 2, axis=-2)
                sanitized[f"{prefix}gate_proj.weight"] = gate
                sanitized[f"{prefix}up_proj.weight"] = up
                continue
            elif key.endswith(".mlp.experts.gate_up_proj_bias"):
                prefix = key.removesuffix("gate_up_proj_bias")
                gate, up = mx.split(value, 2, axis=-1)
                sanitized[f"{prefix}gate_proj.bias"] = gate
                sanitized[f"{prefix}up_proj.bias"] = up
                continue
            elif ".mlp.experts.gate_up_proj." in key:
                # Accept MLX checkpoints produced with the earlier fused
                # runtime layout as well as the published split layout. This
                # applies to dense weights and every quantization companion.
                prefix, suffix = key.split("gate_up_proj.", 1)
                output_axis = -1 if suffix == "bias" else -2
                gate, up = mx.split(value, 2, axis=output_axis)
                sanitized[f"{prefix}gate_proj.{suffix}"] = gate
                sanitized[f"{prefix}up_proj.{suffix}"] = up
                continue
            elif key.endswith(".mlp.experts.down_proj"):
                key = f"{key}.weight"
                value = mx.contiguous(value.swapaxes(-1, -2))
            elif key.endswith(".mlp.experts.down_proj_bias"):
                key = key.replace("down_proj_bias", "down_proj.bias")
            sanitized[key] = value
        return sanitized

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith(".sinks")

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if path.endswith(".mlp.router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
