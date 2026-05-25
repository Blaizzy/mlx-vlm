from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5Attention as Qwen3_5MoeAttention
from ..qwen3_5.language import Qwen3_5GatedDeltaNet as Qwen3_5MoeGatedDeltaNet
from ..qwen3_5.language import Qwen3_5MLP as Qwen3_5MoeMLP
from ..qwen3_5.language import Qwen3_5Model, _target_verify_linear
from .config import ModelConfig, TextConfig


def _target_verify_switch_glu(switch_mlp: SwitchGLU, x, indices, target_verify: bool):
    if not (target_verify and x.ndim == 3 and x.shape[1] > 1):
        return switch_mlp(x, indices)

    B, T, D = x.shape
    k = indices.shape[-1]
    flat_x = x.reshape(B * T, D)
    flat_indices = indices.reshape(B * T, k)
    flat_x = mx.expand_dims(flat_x, (-2, -3))

    up = switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
    gate = switch_mlp.gate_proj(flat_x, flat_indices, sorted_indices=False)
    out = switch_mlp.down_proj(
        switch_mlp.activation(up, gate),
        flat_indices,
        sorted_indices=False,
    )
    return out.squeeze(-2).reshape(B, T, k, -1)


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

    def __call__(
        self,
        x: mx.array,
        target_verify: bool = False,
    ) -> mx.array:
        gates = _target_verify_linear(self.gate, x, target_verify)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = _target_verify_switch_glu(self.switch_mlp, x, inds, target_verify)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x, target_verify)
        shared_y = (
            mx.sigmoid(_target_verify_linear(self.shared_expert_gate, x, target_verify))
            * shared_y
        )

        return y + shared_y


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
        gdn_sink: Optional[list] = None,
        target_verify: bool = False,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
                gdn_sink=gdn_sink,
                target_verify=target_verify,
            )
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask,
                cache,
                position_ids,
                target_verify=target_verify,
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h), target_verify)
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
