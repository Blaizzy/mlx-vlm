from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5Attention as Qwen3_5MoeAttention
from ..qwen3_5.language import Qwen3_5GatedDeltaNet as Qwen3_5MoeGatedDeltaNet
from ..qwen3_5.language import Qwen3_5MLP as Qwen3_5MoeMLP
from ..qwen3_5.language import Qwen3_5Model
from ..switch_layers import SwitchGLU
from .config import ModelConfig, TextConfig


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3_5MoeMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        self.sharding_group = None

    def _shared_expert_scale(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.shared_expert_gate(x))

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = self._shared_expert_scale(x) * shared_y

        out = y + shared_y
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out


class Qwen3_5MoeDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5MoeAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MoeSparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
            )
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5MoeModel(Qwen3_5Model):

    def __init__(self, args: TextConfig):
        nn.Module.__init__(self)
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3_5MoeDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1


class LanguageModel(Qwen3_5LanguageModel):

    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5MoeModel(args)
        self._rope_deltas = None
        self._position_ids = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def shard(self, group: Optional[mx.distributed.Group] = None) -> None:
        """Tensor parallelism for Qwen3.5/3.6 MoE attention and expert layers."""
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        if size == 1:
            return
        if getattr(self, "_sharding_group", None) is not None:
            raise ValueError("The language model is already sharded")

        # Validate before mutating any weights or head counts. Linear attention
        # partitions complete heads; full attention can replicate KV heads.
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "num_attention_heads",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            value = getattr(self.args, name)
            if value % size:
                raise ValueError(f"{name} ({value}) must be divisible by {size}")
        kv_heads = self.args.num_key_value_heads
        if max(size, kv_heads) % min(size, kv_heads):
            raise ValueError(
                f"num_key_value_heads ({kv_heads}) must divide or be divisible by {size}"
            )
        for layer in self.layers:
            attention = layer.linear_attn if layer.is_linear else layer.self_attn
            output_projection = (
                attention.out_proj if layer.is_linear else attention.o_proj
            )
            for projection in (
                output_projection,
                layer.mlp.switch_mlp.down_proj,
                layer.mlp.shared_expert.down_proj,
            ):
                for name in ("weight", "scales", "biases"):
                    parameter = getattr(projection, name, None)
                    if parameter is not None and parameter.shape[-1] % size:
                        raise ValueError(
                            f"Cannot split {name} input dimension "
                            f"({parameter.shape[-1]}) across {size} ranks; "
                            "quantization groups must fit within each shard"
                        )

        def repeat_kv_heads(projection):
            if size <= kv_heads:
                return

            def repeat(weight):
                shape = weight.shape
                weight = weight.reshape(kv_heads, shape[0] // kv_heads, *shape[1:])
                return mx.repeat(weight, size // kv_heads, axis=0).reshape(
                    -1, *shape[1:]
                )

            projection.update(tree_map(repeat, projection.parameters()))

        for layer in self.layers:
            if layer.is_linear:
                attention = layer.linear_attn
                key_dim = attention.key_dim
                # Q, K and V are separate contiguous segments, not interleaved.
                shard_inplace(
                    attention.conv1d,
                    lambda path, weight: (0, [key_dim, 2 * key_dim]),
                    group=group,
                )
                attention.conv1d.groups //= size
                shard_inplace(
                    attention.in_proj_qkv,
                    "all-to-sharded",
                    segments=[key_dim, 2 * key_dim],
                    group=group,
                )
                for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
                    shard_inplace(
                        getattr(attention, name), "all-to-sharded", group=group
                    )
                for name in ("dt_bias", "A_log"):
                    setattr(
                        attention,
                        name,
                        mx.contiguous(mx.split(getattr(attention, name), size)[rank]),
                    )
                shard_inplace(attention.out_proj, "sharded-to-all", group=group)
                for name in (
                    "num_k_heads",
                    "num_v_heads",
                    "key_dim",
                    "value_dim",
                    "conv_dim",
                ):
                    setattr(attention, name, getattr(attention, name) // size)
                attention.sharding_group = group
            else:
                attention = layer.self_attn
                repeat_kv_heads(attention.k_proj)
                repeat_kv_heads(attention.v_proj)
                for name in ("q_proj", "k_proj", "v_proj"):
                    setattr(
                        attention,
                        name,
                        shard_linear(
                            getattr(attention, name), "all-to-sharded", group=group
                        ),
                    )
                attention.o_proj = shard_linear(
                    attention.o_proj, "sharded-to-all", group=group
                )
                attention.num_attention_heads //= size
                attention.num_key_value_heads = max(1, kv_heads // size)

            # Keep the router replicated so every rank selects the same experts.
            # Sum routed and shared expert partial outputs in a single collective.
            for expert in (layer.mlp.switch_mlp, layer.mlp.shared_expert):
                for name in ("gate_proj", "up_proj"):
                    shard_inplace(getattr(expert, name), "all-to-sharded", group=group)
                shard_inplace(expert.down_proj, "sharded-to-all", group=group)
            layer.mlp.sharding_group = group

        self._sharding_group = group
