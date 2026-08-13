# Copyright © 2024-2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import LanguageModelOutput
from ..cache import ArraysCache
from .config import ModelConfig


class MambaBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_bcdt_rms = args.use_bcdt_rms
        self.in_proj = nn.Linear(
            args.hidden_size, args.intermediate_size * 2, bias=args.use_bias
        )
        self.conv1d = nn.Conv1d(
            in_channels=args.intermediate_size,
            out_channels=args.intermediate_size,
            kernel_size=args.conv_kernel,
            groups=args.intermediate_size,
            bias=args.use_conv_bias,
            padding=0,
        )
        self.x_proj = nn.Linear(
            args.intermediate_size,
            self.time_step_rank + 2 * args.state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, args.intermediate_size, bias=True)
        state_indices = mx.arange(1.0, args.state_size + 1.0).reshape(
            1, args.state_size
        )
        self.A_log = mx.log(
            mx.repeat(state_indices, repeats=args.intermediate_size, axis=0)
        )
        self.D = mx.ones((args.intermediate_size,))
        self.out_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.use_bias
        )

    def ssm_step(self, x, state_matrix, state=None):
        delta_bc = self.x_proj(x)
        delta, b_value, c_value = mx.split(
            delta_bc,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        if self.use_bcdt_rms:
            epsilon = self.args.mixer_rms_eps
            delta = mx.fast.rms_norm(delta, weight=None, eps=epsilon)
            b_value = mx.fast.rms_norm(b_value, weight=None, eps=epsilon)
            c_value = mx.fast.rms_norm(c_value, weight=None, eps=epsilon)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(b_value, 1)
        if state is not None:
            new_state += state * mx.exp(mx.expand_dims(delta, -1) * state_matrix)
        output = (new_state @ mx.expand_dims(c_value, -1)).squeeze(2)
        return output + self.D * x, new_state

    def _process_sequence(self, x, convolution_cache, state_cache):
        _, sequence_length, _ = x.shape
        x, gate = self.in_proj(x).split(indices_or_sections=2, axis=-1)
        kernel_size = self.conv_kernel_size
        if convolution_cache is None:
            full_x = mx.pad(x, [(0, 0), (kernel_size - 1, 0), (0, 0)])
        else:
            full_x = mx.concatenate([convolution_cache, x], axis=1)
        convolution_output = self.conv1d(full_x)
        new_convolution_cache = full_x[:, -(kernel_size - 1) :, :]
        x = nn.silu(convolution_output)
        state_matrix = -mx.exp(self.A_log)
        current_state = state_cache
        outputs = []
        for token_index in range(sequence_length):
            output, current_state = self.ssm_step(
                x[:, token_index], state_matrix, current_state
            )
            outputs.append(output)
        output = mx.stack(outputs, axis=1)
        return self.out_proj(swiglu(gate, output)), (
            new_convolution_cache,
            current_state,
        )

    def __call__(self, x, cache):
        if cache is None:
            convolution_cache, state_cache = None, None
        else:
            convolution_cache, state_cache = cache[0], cache[1]
        output, (new_convolution_cache, new_state_cache) = self._process_sequence(
            x, convolution_cache, state_cache
        )
        if isinstance(cache, ArraysCache):
            cache[0] = new_convolution_cache
            cache[1] = new_state_cache
        return output


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs, cache, inputs_embeds=None):
        hidden_states = (
            self.embeddings(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, layer_cache)
        return self.norm_f(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)
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

    def make_cache(self):
        return [ArraysCache(size=2) for _ in self.layers]
