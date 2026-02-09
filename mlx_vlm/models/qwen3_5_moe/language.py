from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5Attention as Qwen3_5MoeAttention
from ..qwen3_5.language import Qwen3_5GatedDeltaNet as Qwen3_5MoeGatedDeltaNet
from ..qwen3_5.language import Qwen3_5MLP as Qwen3_5MoeMLP
from ..qwen3_5.language import Qwen3_5Model
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

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

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
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
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

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
