# Copyright © 2025 Apple Inc.


import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import LanguageModelOutput, create_ssm_mask
from ..cache import ArraysCache
from ..ssm import ssm_update
from .config import ModelConfig


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.num_heads * args.head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.time_step_limit = args.time_step_limit
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)
        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)
        self.norm = MambaRMSNormGated(
            self.intermediate_size, eps=args.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def _conv(self, conv_input, cache, mask):
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (
                        conv_input.shape[0],
                        self.conv_kernel_size - 1,
                        self.conv_dim,
                    ),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                total_length = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, total_length - keep)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -keep:, :]
        else:
            padded_input = mx.pad(
                conv_input,
                [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)],
            )
        return nn.silu(self.conv1d(padded_input))

    def _ssm(self, hidden_states, b_value, c_value, dt, cache, mask):
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        b_value = b_value.reshape(
            batch_size,
            sequence_length,
            self.n_groups,
            self.ssm_state_size,
        )
        c_value = c_value.reshape(
            batch_size,
            sequence_length,
            self.n_groups,
            self.ssm_state_size,
        )
        if cache:
            state, lengths = cache[1], cache.lengths
        else:
            state, lengths = None, None
        output, state = ssm_update(
            hidden_states,
            self.A_log,
            b_value,
            c_value,
            self.D,
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
            lengths,
        )
        if cache:
            cache[1] = state
        return output.reshape(batch_size, sequence_length, self.intermediate_size)

    def __call__(self, hidden_states, mask, cache=None):
        projected = self.in_proj(hidden_states)
        gate, convolution_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        convolution_output = self._conv(convolution_input, cache, mask)
        hidden_states, b_value, c_value = mx.split(
            convolution_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        output = self._ssm(hidden_states, b_value, c_value, dt, cache, mask)
        if cache:
            cache.advance(output.shape[1])
        return self.out_proj(self.norm(output, gate))


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.mixer = Mamba2Block(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x, mask, cache=None):
        return self.mixer(self.norm(x), mask, cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            ResidualBlock(args, index) for index in range(args.num_hidden_layers)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        hidden_states = (
            self.embeddings(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_ssm_mask(hidden_states, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask, layer_cache)
        return self.norm_f(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        hidden_states = self.backbone(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        for key, value in weights.items():
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
        return weights

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self, batch_size: int = 1):
        return [ArraysCache(size=2) for _ in self.layers]
