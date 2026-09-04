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
from ..ssm import ssm_update
from ..switch_layers import SwitchMLP
from .config import ModelConfig
from .speculative_verifier import NemotronHExactSpeculativeVerifier

_EXACT_SPECULATIVE_VERIFIER = NemotronHExactSpeculativeVerifier()


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, group_size: int):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)
        self.group_size = group_size

    def __call__(self, x: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

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
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        group_size = self.intermediate_size // self.n_groups
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=args.layer_norm_epsilon,
            group_size=group_size,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _split_projected_states(self, projected: mx.array):
        # Nemotron-H checkpoints may tensor-core-pad the projection with two
        # unused ``d_mlp`` branches. The reference derives their width from the
        # loaded weight rather than the config and discards them before gate.
        base_size = self.intermediate_size + self.conv_dim + self.num_heads
        extra_size = projected.shape[-1] - base_size
        if extra_size < 0 or extra_size % 2:
            raise ValueError(
                "invalid Nemotron-H Mamba projection width: "
                f"got {projected.shape[-1]}, expected {base_size} plus an even padding"
            )
        d_mlp = extra_size // 2
        gate_start = 2 * d_mlp
        conv_start = gate_start + self.intermediate_size
        dt_start = conv_start + self.conv_dim
        return (
            projected[..., gate_start:conv_start],
            projected[..., conv_start:dt_start],
            projected[..., dt_start:],
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
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
        else:
            state = None

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
        )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = self._split_projected_states(projected)
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelConfig, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size

        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(nn.relu2(self.up_proj(x)))


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class NemotronHMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size

        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        self.switch_mlp = SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_shared_expert_intermediate_size
            self.shared_experts = NemotronHMLP(
                config, intermediate_size=intermediate_size
            )

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )

    def __call__(self, x):
        residuals = x
        inds, scores = self.gate(x)

        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)

        return y


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelConfig, block_type: str):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif self.block_type == "E":
            self.mixer = NemotronHMoE(args)

    def __call__(
        self,
        x,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.norm(x)
        if self.block_type == "M" or self.block_type == "*":
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)

        return x + hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelConfig, with_embeddings: bool = True):
        super().__init__()
        self.with_embeddings = with_embeddings
        if with_embeddings:
            self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, block_type)
            for block_type in args.hybrid_override_pattern
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs=None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        hidden_sink: Optional[list[mx.array]] = None,
    ):
        if inputs is None and inputs_embeds is None:
            raise ValueError("Provide either inputs or inputs_embeds")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif self.with_embeddings:
            hidden_states = self.embeddings(inputs)
        else:
            raise ValueError("This Nemotron-H backbone has no token embedding table")

        if cache is None:
            cache = [None] * len(self.layers)
        has_attention = any(layer.block_type == "*" for layer in self.layers)
        has_mamba = any(layer.block_type == "M" for layer in self.layers)
        attn_cache = cache[self.fa_idx] if has_attention else None
        ssm_cache = cache[self.ssm_idx] if has_mamba else None
        attn_mask = create_attention_mask(hidden_states, attn_cache)
        ssm_mask = create_ssm_mask(hidden_states, ssm_cache)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        cache_counter = 0
        for index, layer in enumerate(self.layers):
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden_states)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"backbone.layers.{layer_idx}.mixer"
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                for suffix in ("weight", "scales", "biases"):
                    first_key = f"{prefix}.experts.0.{m}.{suffix}"
                    if first_key not in weights:
                        continue
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.{suffix}")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.{suffix}"] = mx.stack(to_join)

        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k and "A_log" not in k

        return predicate


class LanguageModel(nn.Module):
    requires_uniform_batch_acceptance = True

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        speculative_verify = bool(kwargs.pop("speculative_verify", False))
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)

        if speculative_verify:
            return _EXACT_SPECULATIVE_VERIFIER(
                self,
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
                capture_layer_ids=capture_layer_ids,
                return_hidden=return_hidden,
                return_shared_kv=return_shared_kv,
                skip_logits=skip_logits,
            )

        hidden_sink = [] if capture_layer_ids is not None else None
        out = self.backbone(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(out)
        return LanguageModelOutput(
            logits=None if skip_logits else self.lm_head(out),
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        if draft_model is None:
            return True
        prefill_kwargs = prefill_kwargs or {}
        if draft_kind in ("dflash", "dspark"):
            return prefill_kwargs.get("capture_layer_ids") is not None
        return False

    def speculative_draft_hidden(self, hidden: mx.array) -> mx.array:
        return hidden

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return _EXACT_SPECULATIVE_VERIFIER.linear(self.lm_head, hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        output = _EXACT_SPECULATIVE_VERIFIER.quantized_argmax(self.lm_head, hidden)
        if output is not None:
            return output
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    def speculative_verify_dflash_hidden(
        self, inputs: mx.array, cache, capture_layer_ids: list[int]
    ):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        return out.hidden_states[:-1], out.hidden_states[-1], out.gdn_states

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
        )
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        if isinstance(accepted, int):
            accepted_values = [accepted]
        elif isinstance(accepted, mx.array):
            accepted_values = [int(value) for value in accepted.reshape(-1).tolist()]
        else:
            accepted_values = [int(value) for value in accepted]
        if len(set(accepted_values)) != 1:
            raise ValueError(
                "Nemotron-H speculative rollback requires uniform acceptance."
            )
        if gdn_states is None:
            raise RuntimeError(
                "Nemotron-H speculative rollback requires verifier Mamba states."
            )

        max_accepted = accepted_values[0]
        if max_accepted < 0:
            raise ValueError("Accepted tokens must be non-negative.")
        retained = max_accepted + 1
        if retained > int(block_size):
            raise ValueError("Accepted tokens exceed the speculative block size.")
        trim = int(block_size) - retained
        state_index = 0
        for cache in caches:
            if cache is None:
                continue
            if isinstance(cache, ArraysCache):
                if state_index >= len(gdn_states):
                    raise RuntimeError(
                        "Nemotron-H verifier did not return every Mamba state."
                    )
                state_history, conv_input, kernel_size = gdn_states[state_index]
                state_index += 1
                if isinstance(state_history, dict):
                    from .speculative_verifier import replay_mamba_state

                    cache[1] = replay_mamba_state(state_history, retained)
                else:
                    cache[1] = state_history[:, max_accepted]
                cache[0] = conv_input[:, retained : retained + int(kernel_size) - 1]
                if cache._lengths is not None:
                    cache._lengths_advance -= trim
                if cache._left_padding is not None:
                    cache._left_padding_advance -= trim
                continue
            if not cache.is_trimmable():
                raise NotImplementedError(
                    "Nemotron-H speculative rollback requires trimmable attention caches."
                )
            if trim:
                cache.trim(trim)
        if state_index != len(gdn_states):
            raise RuntimeError("Nemotron-H verifier returned extra Mamba states.")
        return max_accepted

    def sanitize(self, weights):
        return Model.sanitize(self, weights)

    @property
    def cast_predicate(self):
        return Model.cast_predicate.fget(self)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        return Model.make_cache(self)
