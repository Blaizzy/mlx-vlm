from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..rope_utils import initialize_rope
from ..ssm import ssm_update
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class GraniteMoeHybridRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class GraniteMoeHybridMamba2Mixer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_n_heads * args.mamba_d_head
        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.time_step_limit = args.time_step_limit
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.mamba_d_conv,
            padding=0,
            groups=self.conv_dim,
            bias=args.mamba_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = GraniteMoeHybridRMSNormGated(
            self.intermediate_size, eps=args.rms_norm_eps
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                total_length = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, total_length - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        return nn.silu(self.conv1d(padded_input))

    def _ssm(
        self,
        hidden_states: mx.array,
        b_value: mx.array,
        c_value: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        b_value = b_value.reshape(
            batch_size, sequence_length, self.n_groups, self.ssm_state_size
        )
        c_value = c_value.reshape(
            batch_size, sequence_length, self.n_groups, self.ssm_state_size
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
            self.D.astype(hidden_states.dtype),
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

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, b_value, c_value = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        output = self._ssm(hidden_states_ssm, b_value, c_value, dt, cache, mask)
        if cache:
            cache.advance(output.shape[1])
        output = self.norm(output, gate)
        return self.out_proj(output)


class GraniteMoeHybridAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = args.attention_multiplier
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        if args.position_embedding_type != "nope":
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
                None,
                args.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(
            batch_size, sequence_length, self.n_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)

        if self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class GraniteMoeHybridTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        logits = self.layer(hidden_states)
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits, precise=True, axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeHybridMoE(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )
        self.router = GraniteMoeHybridTopKGating(
            input_size=args.hidden_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        token_ids, gates = self.router(x)
        output = self.switch_mlp(x, token_ids)
        return (output * gates[..., None]).sum(axis=-2)


class GraniteMoeHybridSharedMLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.input_linear = nn.Linear(
            args.hidden_size, args.shared_intermediate_size * 2, bias=False
        )
        self.output_linear = nn.Linear(
            args.shared_intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(swiglu(gate, up))


class GraniteMoeHybridMLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class GraniteMoeHybridLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = args.residual_multiplier
        self.use_moe = args.use_moe

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if layer_type == "mamba":
            self.mamba = GraniteMoeHybridMamba2Mixer(args)
        elif layer_type == "attention":
            self.self_attn = GraniteMoeHybridAttention(args)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if self.use_moe:
            self.shared_mlp = GraniteMoeHybridSharedMLP(args)
            self.block_sparse_moe = GraniteMoeHybridMoE(args)
        else:
            self.mlp = GraniteMoeHybridMLP(args)

        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        hidden_states = self.input_layernorm(x)

        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)

        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            moe_output = self.block_sparse_moe(normed)
            shared_output = self.shared_mlp(normed)
            mlp_output = moe_output + shared_output
        else:
            mlp_output = self.mlp(normed)

        return residual + mlp_output * self.residual_multiplier


class GraniteMoeHybridModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GraniteMoeHybridLayer(args, layer_type) for layer_type in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        self.fa_idx = (
            args.layer_types.index("attention")
            if "attention" in args.layer_types
            else None
        )
        self.ssm_idx = (
            args.layer_types.index("mamba") if "mamba" in args.layer_types else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        hidden_states = hidden_states * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)

        attention_mask = None
        mamba_mask = None

        if self.fa_idx is not None:
            attention_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            mamba_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, layer_cache in zip(self.layers, cache):
            mask = attention_mask if layer.layer_type == "attention" else mamba_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = GraniteMoeHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        output = self.model(inputs, inputs_embeds, cache)

        if self.args.tie_word_embeddings:
            output = self.model.embed_tokens.as_linear(output)
        else:
            output = self.lm_head(output)

        return LanguageModelOutput(logits=output / self.logits_scaling)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "mamba":
                caches.append(ArraysCache(size=2))
            elif layer.layer_type == "attention":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        for key, value in weights.items():
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)

        if (
            self.args.use_moe
            and "model.layers.0.block_sparse_moe.input_linear.weight" in weights
        ):
            for layer_index in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{layer_index}.block_sparse_moe"

                input_weight = weights.pop(f"{prefix}.input_linear.weight")
                _, expert_hidden, _ = input_weight.shape
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = input_weight[
                    :, : expert_hidden // 2, :
                ]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = input_weight[
                    :, expert_hidden // 2 :, :
                ]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.output_linear.weight"
                )

        elif (
            not self.args.use_moe
            and "model.layers.0.shared_mlp.input_linear.weight" in weights
        ):
            for layer_index in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{layer_index}.shared_mlp"
                input_weight = weights.pop(f"{prefix}.input_linear.weight")
                gate_projection, up_projection = mx.split(input_weight, 2, axis=0)
                weights[f"model.layers.{layer_index}.mlp.gate_proj.weight"] = (
                    gate_projection
                )
                weights[f"model.layers.{layer_index}.mlp.up_proj.weight"] = (
                    up_projection
                )
                weights[f"model.layers.{layer_index}.mlp.down_proj.weight"] = (
                    weights.pop(f"{prefix}.output_linear.weight")
                )

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if self.args.use_moe and path.endswith("router.layer"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
