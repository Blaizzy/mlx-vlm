import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import _BaseCache

from ..base import LanguageModelOutput, create_attention_mask, visualize_attention_mask
from ..cache import ChunkedKVCache, KVCache, RotatingKVCache
from .config import TextConfig



class Gemma3nRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        with_scale: bool = True,
    ):
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Compute variance along last dimension
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        # Normalize
        normed = x / mx.sqrt(variance + self.eps)

        # Apply weight scaling
        if self.with_scale:
            weight = self.weight
        else:
            weight = mx.ones(x.shape[-1], dtype=x.dtype)

        scaled_weight = weight + self.scale_shift
        output = normed * scaled_weight

        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(
            self.config.hidden_size, self.config.laurel_rank, bias=False
        )
        self.linear_right = nn.Linear(
            self.config.laurel_rank, self.config.hidden_size, bias=False
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: Optional[mx.array] = None,
    unsqueeze_dim: int = 1,
):

    cos = mx.expand_dims(cos, unsqueeze_dim)
    sin = mx.expand_dims(sin, unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma3nRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # TODO: This is a hack to get the RoPE parameters just for testing.
        # Will be removed...
        from transformers.modeling_rope_utils import _compute_default_rope_parameters

        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)

    def __call__(self, x, position_ids):
        inv_freq_expanded = self._inv_freq[None, :, None].astype(mx.float32)
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)

        freqs = (
            inv_freq_expanded.astype(mx.float32)
            @ position_ids_expanded.astype(mx.float32)
        ).transpose(0, 2, 1)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


class Gemma3nAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern
        self.attn_logit_softcapping = config.attn_logit_softcapping

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = config.query_rescale_scalar / config.query_pre_attn_scalar

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.qkv_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx

        # Compute the layer index from which shared KV cache values will be retrieved.
        if not self.is_kv_shared_layer:
            self.kv_shared_layer_index = None
        elif self.is_sliding:
            # The last layer that computes local sliding attention is always 2 before sharing starts
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 2
        else:
            # The last layer before sharing starts is always the last that computes global attention layer
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 1

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        caches: Optional[List[Any]] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        queries = self.q_proj(x)
        queries = queries.reshape(hidden_shape)
        queries = self.qkv_norm(queries)
        queries = apply_rotary_pos_emb(queries, cos, sin, unsqueeze_dim=2)
        queries = queries.transpose(0, 2, 1, 3)

        if (
            self.is_kv_shared_layer
            and self.kv_shared_layer_index is not None
            and caches is not None
            and cache is not None
            and cache.offset > 0
        ):
            # For shared layers, retrieve KV from the designated cache layer
            shared_cache = caches[self.kv_shared_layer_index]
            keys, values = shared_cache.state

        else:
            keys = self.k_proj(x).reshape(hidden_shape)
            keys = self.qkv_norm(keys)
            keys = apply_rotary_pos_emb(keys, cos, sin, unsqueeze_dim=2)
            keys = keys.transpose(0, 2, 1, 3)

            values = self.v_proj(x).reshape(hidden_shape)
            values = self.qkv_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        keys = mx.repeat(keys, repeats=self.repeats, axis=1)
        values = mx.repeat(values, repeats=self.repeats, axis=1)

        attn_weights = mx.matmul(queries, keys.swapaxes(2, 3)) * self.scale

        if self.attn_logit_softcapping is not None and self.attn_logit_softcapping > 0:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = mx.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping
        if mask is not None:  # no matter the length, we just slice it
            causal_mask = mask[:, :, :, : keys.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(
            queries.dtype
        )

        output = mx.matmul(attn_weights, values)

        output = output.transpose(0, 2, 1, 3).reshape(input_shape + (-1,))

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0

    def __call__(self, x: mx.array):
        gate_proj = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = nn.gelu_approx(gate_proj)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: mx.array) -> mx.array:
        # For normal distribution, icdf(p) = -sqrt(2) * erfinv(2p - 1)
        p = mx.array(self.activation_sparsity, dtype=mx.float32)
        std_multiplier = mx.sqrt(2) * mx.erfinv(2 * p - 1)
        std_multiplier = std_multiplier.astype(inputs.dtype)
        inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
        inputs_std = mx.std(inputs, axis=-1, keepdims=True)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return mx.maximum(0, inputs - cutoff_x)


class Gemma3nAltUp(nn.Module):
    """Alternating Updates (AltUp)"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.correct_output_scale = mx.zeros((self.config.hidden_size,))
        self.correction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False
        )
        self.modality_router = nn.Linear(
            self.config.hidden_size, self.config.altup_num_inputs, bias=False
        )
        self.router_norm = Gemma3nRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )
        self._router_input_scale = mx.array(self.config.hidden_size**-1.0)

    def compute_router_modalities(self, x: mx.array) -> mx.array:
        router_inputs = self.router_norm(x) * self._router_input_scale.astype(
            self.router_norm.weight.dtype
        )
        routed = self.modality_router(router_inputs).astype(mx.float32)
        return mx.tanh(routed)

    def predict(self, x: mx.array) -> mx.array:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])

        # Force float32 computation like PyTorch
        self.prediction_coefs.weight = self.prediction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight = mx.clip(
                self.prediction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        # Fix: Use permute pattern that matches PyTorch exactly
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(
                *modalities.shape[:-1],
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
            )
            .transpose(0, 1, 3, 2)  # This should match PyTorch's permute(0, 1, 3, 2)
        )

        # Fix: Match PyTorch's tensor manipulation exactly
        # PyTorch: hidden_states.float().permute(1, 2, 3, 0)
        x_permuted = x.astype(mx.float32).transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_permuted, all_coefs)
        predictions = predictions.transpose(
            3, 0, 1, 2
        )  # Match PyTorch's permute(3, 0, 1, 2)
        predictions += x
        return predictions.astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array):
        modalities = self.compute_router_modalities(activated)

        # Force float32 computation like PyTorch
        self.correction_coefs.weight = self.correction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight = mx.clip(
                self.correction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        # Fix: Match PyTorch's broadcasting approach instead of loop
        all_coefs = self.correction_coefs(modalities) + 1.0

        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        # Replicate innovation for all inputs like PyTorch
        innovation_expanded = mx.broadcast_to(
            mx.expand_dims(innovation, axis=0),
            (self.config.altup_num_inputs,) + innovation.shape,
        )

        # Fix: Match PyTorch's tensor manipulation
        # PyTorch: all_coefs.permute(2, 1, 0).unsqueeze(1)
        all_coefs_reshaped = all_coefs.transpose(2, 1, 0)
        all_coefs_reshaped = mx.expand_dims(all_coefs_reshaped, axis=1)

        # Broadcast multiply like PyTorch
        corrected = innovation_expanded * all_coefs_reshaped
        corrected += predictions

        return corrected.astype(activated.dtype)

    def scale_corrected_output(self, corrected: mx.array):
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale

    def __call__(self, x: mx.array, activated: mx.array):
        predictions = self.predict(x)
        corrected = self.correct(predictions=predictions, activated=activated)
        output = corrected[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            output = self.scale_corrected_output(output)
        return corrected, output


class Gemma3nDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3nAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )

        self.post_attention_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.altup = Gemma3nAltUp(config)
        self.laurel = Gemma3nLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(
            self.hidden_size, self.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            self.hidden_size_per_layer_input, self.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        caches: Optional[List[Any]] = None,
        cache_position: Optional[mx.array] = None,
        position_embeddings_global: Optional[mx.array] = None,
        position_embeddings_local: Optional[mx.array] = None,
    ):
        if isinstance(x, list):
            x = mx.stack(x, axis=0)

        if self.is_sliding and mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)

            min_dtype = mx.finfo(mask.dtype).min
            sliding_window_mask = mx.tril(
                mx.ones(mask.shape, dtype=mx.bool_), k=-self.sliding_window
            )
            mask = mx.where(sliding_window_mask, min_dtype, mask)
            # In case we are beyond the sliding window, we need to correctly offset the mask slicing
            offset = cache_position[-1] - effective_seq_len + 1
            # Should only be used when beyond the sliding window (i.e. offset > 0)
            offset = mx.clip(offset, a_min=0, a_max=None)
            # equivalent to: `attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indexes = mx.arange(min(effective_seq_len, mask.shape[-1]))
            mask_indexes += offset
            mask = mask[:, :, :, mask_indexes.astype(mx.int32)]

        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
            caches,
            position_embeddings,
        )

        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / mx.sqrt(
            mx.array(2.0, dtype=active_prediction.dtype)
        )

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = nn.gelu_approx(first_prediction)

        first_prediction = mx.multiply(first_prediction, per_layer_input)

        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        for i in range(1, len(corrected_predictions)):
            corrected_predictions[i] = corrected_predictions[i] + first_prediction

        return corrected_predictions


class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    """This module overrides nn.Embeddings' forward by multiplying with embeddings scale."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: Optional[float] = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def __call__(self, x: mx.array):
        return super().__call__(x) * mx.array(self.embed_scale, mx.float32).astype(
            self.weight.dtype
        )


class Gemma3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0

        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size**0.5
        )
        self.layers = [
            Gemma3nDecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.altup_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.altup_unembed_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.norm = Gemma3nRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self._per_layer_projection_scale = mx.array(self.hidden_size**-0.5)
        self._per_layer_input_scale = mx.rsqrt(mx.array(2.0))

        self.rope_embedding = Gemma3nRotaryEmbedding(config)
        self.rope_embedding_local = Gemma3nRotaryEmbedding(config)

    def _update_causal_mask(
        self,
        attention_mask: mx.array,
        input_tensor: mx.array,
        cache_position: mx.array,
        past_key_values: list | None,
    ):
        """
        Build a 4-D (B,1,Q,K) mask that bans
            • every key to the right of the current query token (causality)
            • every key that is padding (attention_mask == 0)
        and leaves allowed cells at 0 · 0.
        """

        dtype  = input_tensor.dtype
        batch  = input_tensor.shape[0]
        q_len  = input_tensor.shape[1]           # sequence_length

        # -- determine how *wide* the mask must be ---------------------------
        if (
            past_key_values                                    # cache list exists
            and isinstance(past_key_values[0], _BaseCache)     # our KV-cache class
        ):
            layer0 = past_key_values[0]

            if layer0.keys is not None:                        # cache already filled
                k_len = int(layer0.keys.shape[2])
            else:                                              # first forward pass
                k_len = int(layer0.max_size)                   # use full capacity

            k_len = max(k_len, q_len)                          # safety for pre-fill
        else:
            k_len = (
                int(attention_mask.shape[-1])
                if attention_mask is not None
                else q_len
            )

        # -- build / return ---------------------------------------------------
        return self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length = q_len,
            target_length   = k_len,
            dtype           = dtype,
            cache_position  = cache_position,
            batch_size      = batch,
        )



    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: mx.array | None,   # (B ,K) or (B,1,Q,K) or None
        sequence_length: int,              # Q
        target_length:   int,              # K
        dtype,
        cache_position: mx.array,          # (Q,)  – indices of current tokens
        batch_size: int,
    ):
        """
        Exact port of HF Gemma3 · 5 logic (2025-05-29).

        Returned tensor shape → (B, 1, Q, K)   – values are 0.0 (keep) or
        the minimum finite value of `dtype` (ban).
        """
        # ------------------------------------------------------------------ #
        #  if caller already gave us a 4-D “inverted” mask (Flex, FA2, …) we
        #  trust it and short-circuit.
        # ------------------------------------------------------------------ #
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.astype(dtype)

        min_val = mx.finfo(dtype).min            # -∞  in chosen precision

        # ------------------------------------------------------------------ #
        # 1. causal part  – start full of -∞ then zero-out lower-triangle
        # ------------------------------------------------------------------ #
        causal = mx.full((sequence_length, target_length), min_val, dtype=dtype)

        if sequence_length != 1:
            causal = mx.triu(causal, k=1)        # keep q≥k at 0.0

        # static-cache shift: forbid keys that have *not* been written yet
        # (i.e. k_index  >  cache_position[q])
        causal *= (
            mx.arange(target_length) > cache_position.reshape(-1, 1)
        )

        # shape → (1,1,Q,K) then broadcast batch
        causal = mx.expand_dims(causal, 0)       # (1,Q,K)
        causal = mx.expand_dims(causal, 0)       # (1,1,Q,K)
        causal = mx.repeat(causal, repeats=batch_size, axis=0).astype(dtype)

        # ------------------------------------------------------------------ #
        # 2. padding part  – turn “mask = 0” keys into -∞ inside the slice
        # ------------------------------------------------------------------ #
        if attention_mask is not None:           # (B,K)
            Kmask_len  = int(attention_mask.shape[-1])
            key_pad    = attention_mask.astype(dtype)            # 1/0
            key_pad    = mx.expand_dims(key_pad, 1)               # (B,1,K)
            key_pad    = mx.expand_dims(key_pad, 2)               # (B,1,1,K)

            # slice only the prefix that overlaps real keys
            causal_prefix = causal[:, :, :, :Kmask_len]
            causal_prefix = mx.where(
                key_pad == 0,                        #==> padding key?
                min_val,                             # ban
                causal_prefix                        # keep previous (0 or –∞)
            )
            causal = mx.concatenate(
                [causal_prefix, causal[:, :, :, Kmask_len:]], axis=-1
            )

        return causal


    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ):
        per_layer_inputs = kwargs.get("per_layer_inputs", None)
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if per_layer_inputs is None and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)

        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        cache_position = None
        if cache_position is None:
            past_seen_tokens = 0
            if cache is not None and cache[0] is not None:
                past_seen_tokens = cache[0].offset

            cache_position = mx.arange(
                past_seen_tokens,
                past_seen_tokens + h.shape[1],
            )

        causal_mask = self._update_causal_mask(
            mask,
            h,
            cache_position,
            cache,
        )

        h0 = h

        # Initialize RoPE embeddings
        position_ids = cache_position[None, :]
        position_embeddings_global = self.rope_embedding(h0, position_ids)
        position_embeddings_local = self.rope_embedding_local(h0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        epsilon_tensor = mx.array(mx.finfo(h0.dtype).min, dtype=h0.dtype)

        h_list = [h0] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            altup_proj = self.altup_projections[i - 1](h_list[i])
            h_list[i] = altup_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h_list[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h_list[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.stack(h_list, axis=0)

        for i, (layer, c) in enumerate(
            zip(self.layers[: self.config.num_hidden_layers], cache)
        ):
            per_layer_input = per_layer_inputs[:, :, i, :]

            h = layer(
                h,
                causal_mask,
                c,
                per_layer_input,
                cache,
                cache_position,
                position_embeddings_global,
                position_embeddings_local,
            )

        # Per-layer inputs to single output
        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5

        for i in range(1, self.config.altup_num_inputs):
            altup_unemb_proj = self.altup_unembed_projections[i - 1](h[i])
            h[i] = altup_unemb_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.mean(h, axis=0)

        return self.norm(h)

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = mx.logical_and(
            input_ids >= 0, input_ids < self.vocab_size
        )
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        return result

    def project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: Optional[mx.array] = None
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self._per_layer_projection_scale.astype(
            inputs_embeds.dtype
        )

        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (
            per_layer_projection + per_layer_inputs
        ) * self._per_layer_input_scale.astype(inputs_embeds.dtype)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out = self.model(
            inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs
        )
        out = self.lm_head(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        sanitized_weights = {}

        for k, v in weights.items():
            if "language_model" in k:
                new_key = k.replace("language_model", "language_model.model")
                sanitized_weights[new_key] = v
            elif "self_attn.rotary_emb.inv_freq" in k:
                continue
            else:
                sanitized_weights[k] = v


        if "lm_head.weight" not in sanitized_weights:
            embed_tokens_key = "language_model.model.embed_tokens.weight"
            if embed_tokens_key in sanitized_weights:
                sanitized_weights["language_model.lm_head.weight"] = sanitized_weights[embed_tokens_key]

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        caches = []

        for i in range(self.config.num_hidden_layers):
            # Normal KVcache and RotatingKVCache work,
            # but results are better with StaticKVCache and SlidingWindowCache.
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(
                    StaticKVCache(
                        max_size=min(
                            self.config.sliding_window,
                            self.config.max_position_embeddings,
                        ),
                    )
                )
            else:
                caches.append(
                    SlidingWindowCache(
                        max_size=min(
                            self.config.sliding_window,
                            self.config.max_position_embeddings,
                        ),
                    )
                )

        return caches


class SlidingWindowCache(_BaseCache):
    """A sliding window cache for local attention layers."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        if self.keys is None:
            # Initialize cache
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Simple sliding window: keep only the last max_size tokens
        if self.offset + seq_len <= self.max_size:
            # Fits within current window
            start_idx = self.offset
            end_idx = self.offset + seq_len
            self.keys[:, :, start_idx:end_idx, :] = keys
            self.values[:, :, start_idx:end_idx, :] = values
            self.offset += seq_len
        else:
            # Need to slide the window
            # Shift existing content left
            shift_amount = seq_len
            if shift_amount < self.max_size:
                self.keys[:, :, :-shift_amount, :] = self.keys[:, :, shift_amount:, :]
                self.values[:, :, :-shift_amount, :] = self.values[
                    :, :, shift_amount:, :
                ]
                # Add new tokens at the end
                self.keys[:, :, -shift_amount:, :] = keys
                self.values[:, :, -shift_amount:, :] = values
            else:
                # New sequence is larger than cache, just keep the last max_size tokens
                self.keys = keys[:, :, -self.max_size :, :]
                self.values = values[:, :, -self.max_size :, :]
            self.offset = self.max_size

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.max_size

    def get_max_cache_shape(self):
        return self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return False  # Sliding window cache doesn't support trimming

    def trim(self, n):
        return 0  # No trimming for sliding window


class StaticKVCache(_BaseCache):
    """A static cache that grows to accommodate all tokens."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        # Initialize cache if needed
        if self.keys is None:
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Update cache
        end_pos = min(self.offset + seq_len, self.max_size)
        actual_seq_len = end_pos - self.offset

        if actual_seq_len > 0:
            self.keys = mx.concatenate(
                [
                    self.keys[:, :, : self.offset, :],
                    keys[:, :, :actual_seq_len, :],
                    self.keys[:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.values = mx.concatenate(
                [
                    self.values[:, :, : self.offset, :],
                    values[:, :, :actual_seq_len, :],
                    self.values[:, :, end_pos:, :],
                ],
                axis=2,
            )
            self.offset = end_pos

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                # Calculate offset based on non-zero entries
                self.offset = self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n