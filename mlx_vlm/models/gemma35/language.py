import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache, RotatingKVCache


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262208
    num_key_value_heads: int = 4
    laurel_rank: int = 64
    frac_shared_layers: float = 0.5
    altup_num_inputs: int = 4
    altup_coef_clip: Optional[float] = None
    altup_correct_scale: bool = True
    hidden_size_per_layer_input: int = 1024
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    query_pre_attn_scalar: float = 0.0625
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 5
    activation_sparsity_pattern: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Gemma3p5EinsumLayer(nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        einsum_str: str,
        *args,
        weight_init: Optional[Callable[..., mx.array]] = None,
        **kwargs,
    ):
        if "->" not in einsum_str:
            raise ValueError("Einsum must contain '->'")

        if len(einsum_str.split("->")[0].split(",")) != 2:
            raise ValueError("Need to have exactly two inputs in einsum instruction")

        super().__init__(*args, **kwargs)
        self.shape = shape
        self.einsum_str = einsum_str

        self.weight = mx.ones(shape)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return mx.einsum(self.einsum_str, x, self.weight)




class Gemma3p5RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = mx.ones(dim)
        else:
            self.weight = None



    def __call__(self, x: mx.array) -> mx.array:
        x = self._guard_against_excess_precision(x)

        scale = self.weight if self.weight is not None else mx.array(1.0)
        if self.scale_shift != 0.0:
            scale += self.scale_shift
        output = mx.fast.rms_norm(x, 1.0 + scale, self.eps)
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _guard_against_excess_precision(self, x: mx.array) -> mx.array:
        # TODO(ryanmullins): Implement Torch equivalent to jax.lax.reduce_precision
        return x


class Gemma3p5LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: TextConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        x = x + normed_laurel_x
        return x


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, num_layers_that_compute_kv: int):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = config.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern != 0

        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=(
                config.rope_local_base_freq
                if self.is_sliding
                else config.rope_global_base_freq
            ),
        )

        self.qkv_norm = Gemma3p5RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

        if num_layers_that_compute_kv is None:
            self.is_kv_shared_layer = False
        else:
            self.is_kv_shared_layer = layer_idx >= num_layers_that_compute_kv


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        queries = self.qkv_norm(queries)
        queries = self.rope(queries)

        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)


        if self.is_kv_shared_layer and cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        else:
            keys = self.qkv_norm(keys)
            keys = self.rope(keys)
            values = self.qkv_norm(values)

        # Sliding window
        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0, *args, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()
        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0

    def __call__(self, x: mx.array):
        gate_proj = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: mx.array) -> mx.array:
        # Calculate the cutoff value based on the target sparsity
        # For normal distribution, we use the inverse CDF (quantile function)
        # Convert to numpy, calculate the quantile, then back to mx.array
        # Use numpy's special functions instead of scipy
        if self.activation_sparsity <= 0.0:
            # For 0 sparsity, return infinity to match PyTorch behavior
            # This will make all values pass through
            inf_value = mx.array(float("inf"))
            return mx.broadcast_to(inf_value, inputs.shape)

        normal_dist = mx.random.normal((1,))

        # Generate a large sample from normal distribution
        sample_size = 100000
        normal_samples = mx.random.normal(shape=(sample_size,))

        # Sort the samples
        sorted_samples = mx.sort(normal_samples)

        # Find the index corresponding to our target sparsity
        idx = int(self.activation_sparsity * sample_size)

        # Get the value at that index as our std_multiplier
        std_multiplier = float(sorted_samples[idx]) if idx < sample_size else 0.0

        # Calculate mean and standard deviation along the last dimension
        inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
        inputs_std = mx.std(inputs, axis=-1, keepdims=True)

        # Calculate the cutoff threshold
        cutoff_x = inputs_mean + inputs_std * std_multiplier

        # Apply ReLU to zero out values below the cutoff
        return mx.maximum(0, inputs - cutoff_x)


class Gemma3p5AltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: TextConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.correct_output_scale = mx.zeros(
            (self.config.hidden_size)
        )
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )


    def compute_router_modalities(self, x: mx.array) -> mx.array:
        x_norm: mx.array = self.router_norm(x)
        router_inputs: mx.array = x_norm * self.config.hidden_size**-1.0
        # routed adapted from jax.numpy.einsum("btf,fd->btd", ...)
        routed: mx.array = self.modality_router(router_inputs)
        return mx.tanh(routed)

    def predict(self, x: List[mx.array]) -> List[mx.array]:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pij->...ij", ...)
        all_coefs: mx.array = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(
            *modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs
        )

        outputs: list[mx.array] = [mx.zeros_like(x[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            output = outputs[i]

            for j in range(self.config.altup_num_inputs):
                coef = mx.expand_dims(all_coefs[..., i, j], axis=-1)
                output += coef * x[j]

            x_i = x[i]
            outputs[i] = (x_i + output).type(x_i.dtype)

        return outputs

    def correct(self, predictions: List[mx.array], activated: mx.array):
        modalities = self.compute_router_modalities(activated)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        all_coefs: mx.array = self.correction_coefs(modalities)
        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        corrected = [mx.zeros_like(predictions[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            coef = mx.expand_dims(all_coefs[..., i] + 1, axis=-1)
            corrected[i] = (predictions[i] + coef * innovation).type(activated.dtype)

        return corrected

    def scale_corrected_output(self, corrected: mx.array):
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale

    def __call__(self, x: List[mx.array], activated: mx.array):
        predictions = self.predict(x)
        corrected = self.correct(predictions=predictions, activated=activated)
        return corrected


@partial(mx.compile, shapeless=True)
def clip_residual(x, y=None):
    bound = mx.finfo(mx.float16).max
    if y is None:
        if x.dtype == mx.float16:
            return mx.clip(x.astype(mx.float32), -bound, bound).astype(mx.float16)
        else:
            return x

    if x.dtype != mx.float16:
        return x + y

    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


class Gemma3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        num_layers_that_compute_kv: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx, num_layers_that_compute_kv)
        self.mlp = MLP(config)
        self.input_layernorm = Gemma3p5RMSNorm(
            dim=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = nn.GELU()

        self.altup = Gemma3p5AltUp(config)
        self.laurel = Gemma3p5LaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_laurel_norm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        per_layer_input = kwargs.get("per_layer_input")

        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_hidden_states = self.laurel(active_prediction_normed)
        laurel_normed = self.post_laurel_norm(laurel_hidden_states)
        laurel_output = active_prediction_normed + laurel_normed


        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * mx.rsqrt(mx.array(2.0))

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[0]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = mx.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        for i in range(1, len(corrected_predictions)):
            corrected_predictions[i] += first_prediction

        return corrected_predictions


class Gemma3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0

        attention_pattern_length = self.config.sliding_window_pattern
        frac_unshared_layers = 1 - self.config.frac_shared_layers
        num_unshared_layers: int = round(self.config.num_hidden_layers * frac_unshared_layers)

        if num_unshared_layers >= attention_pattern_length:
            numerator = num_unshared_layers + attention_pattern_length - 1
            num_unshared_layers = attention_pattern_length * numerator // attention_pattern_length
        else:
            print(
                "Not rounding unshared layers. round_up_to_nearest_attention_block is"
                " False or num_unshared_layers is less than attention_pattern_length."
            )
        self.num_layers_that_compute_kv = num_unshared_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma3p5DecoderLayer(config=config, layer_idx=layer_idx, num_layers_that_compute_kv=self.num_layers_that_compute_kv)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,

        )

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, bias=False
        )

        self.per_layer_projection_norm = Gemma3p5RMSNorm(
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.altup_projections = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]


        self.altup_unembed_projections = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]

        self.norm = Gemma3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs
    ):
        per_layer_inputs = kwargs.get("per_layer_inputs")
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if per_layer_inputs is None and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        h0 = h
        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1) ** 0.5
        epsilon_tensor = mx.finfo(mx.float16).min

        h: list[mx.array] = [h0] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj: mx.array = self.altup_projections[i - 1](h[i])
            h[i] = altup_proj.type(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )
            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c, per_layer_input)

         # Per-layer inputs to single output
        target_magnitude = mx.mean(h ** 2, axis=-1) ** 0.5
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj = self.altup_unembed_projections[i - 1](h[i])
            h[i] = altup_unemb_proj.type(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.mean(mx.stack(h), axis=0)

        return self.norm(h)


    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = mx.logical_and(input_ids >= 0, input_ids < self.vocab_size)
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens).reshape(
            *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input
        ) * mx.array(self.config.hidden_size**0.5, mx.bfloat16)
        return result.astype(input_ids.dtype)

    def project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: Optional[mx.array] = None
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds).reshape(
            *inputs_embeds.shape[:-1], self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * mx.rsqrt(mx.array(2.0))



class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        out = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):

        if "lm_head.weight" not in weights:
            weights["language_model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

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
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window,
                        keep=0,
                    )
                )
        return caches
