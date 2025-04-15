import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

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
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    query_pre_attn_scalar: float = 0.0625
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 6

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


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Gemma3p5LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: TextConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.linear_left = Gemma3p5EinsumLayer(
            shape=(self.config.hidden_size, self.config.laurel_rank),
            einsum_str="bld,dr->blr",
        )
        self.linear_right = Gemma3p5EinsumLayer(
            shape=(self.config.laurel_rank, self.config.hidden_size),
            einsum_str="blr,rd->bld",
        )
        self.post_laurel_norm = RMSNorm(
            dims=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        x = x + normed_laurel_x
        return x


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
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

        self.q_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
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

        self.q_norm = RMSNorm(
            dims=config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = RMSNorm(
            dims=config.head_dim,
            eps=config.rms_norm_eps,
        )

        self.v_norm = RMSNorm(
            dims=config.head_dim,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        values = self.v_norm(values)

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

    def forward(self, x: mx.array):
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

        self.correction_coefs = mx.zeros(
            (self.config.altup_num_inputs, self.config.altup_num_inputs)
        )
        self.prediction_coefs = mx.zeros(
            (
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
            )
        )
        self.modality_router = Gemma3p5EinsumLayer(
            shape=(self.config.hidden_size, self.config.altup_num_inputs),
            einsum_str="btf,fd->btd",
        )
        self.router_norm = RMSNorm(
            dims=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def compute_router_modalities(self, x: mx.array) -> mx.array:
        x_norm = self.router_norm(x)
        router_inputs = x_norm * self.config.hidden_size**-1.0
        routed: mx.array = self.modality_router(router_inputs)
        modalities = mx.tanh(routed)
        return modalities

    def predict(self, x: Sequence[mx.array]) -> Sequence[mx.array]:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])
        prediction_coefs = self.prediction_coefs

        if self.config.altup_coef_clip is not None:
            prediction_coefs = mx.clip(
                prediction_coefs,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        all_coefs = mx.einsum("...p,pij->...ij", modalities, prediction_coefs)

        outputs: list[mx.array] = [mx.zeros_like(x[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            output = 0.0

            for j in range(self.config.altup_num_inputs):
                coef = mx.expand_dims(all_coefs[..., i, j], axis=-1)
                output += coef * x[j]

            x_i = x[i]
            outputs[i] = (x_i + output).astype(x_i.dtype)

        return outputs

    def correct(
        self, predictions: Sequence[mx.array], activated: mx.array
    ) -> Sequence[mx.array]:
        modalities = self.compute_router_modalities(activated)
        correction_coefs = self.correction_coefs.float()

        if self.config.altup_coef_clip is not None:
            correction_coefs = mx.clip(
                correction_coefs,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        all_coefs = mx.einsum("...p,pi->...i", modalities, correction_coefs)

        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        corrected = [mx.zeros_like(predictions[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            coef = mx.expand_dims(all_coefs[..., i] + 1, axis=-1)
            corrected[i] = (predictions[i] + coef * innovation).astype(activated.dtype)

        return corrected

    def __call__(
        self, x: Sequence[mx.array], activated: mx.array, *args, **kwargs
    ) -> Sequence[mx.array]:
        predictions = self.predict(x, *args, **kwargs)
        corrected = self.correct(
            predictions=predictions, activated=activated, *args, **kwargs
        )
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


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        # Clip the input to avoid overflow in float16
        # Float16 has a max value of 65504. When values exceed this limit, they become inf.
        # Example: If x contains 70000.0 in float16, it becomes inf, causing gradient issues.
        # We upcast to float32 for operations that might exceed the limit, then clip and
        # convert back to float16 to maintain numerical stability.

        # Clip input to avoid overflow in float16
        x = clip_residual(x)

        # Self-attention block
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(r)

        # Add residual connection with overflow protection for float16
        h = clip_residual(x + h)

        # MLP block
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = self.post_feedforward_layernorm(r)

        # Add residual connection with overflow protection for float16
        out = clip_residual(h + out)

        return out


class Gemma3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )
            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c)

        return self.norm(h)


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
