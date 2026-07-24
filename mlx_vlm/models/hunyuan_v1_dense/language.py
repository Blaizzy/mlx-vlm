from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from ..transformer_block import BlockSpec, TransformerBlock
from .config import ModelConfig


class DynamicNTKAlphaRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000,
        scaling_alpha: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def block_spec(args: ModelConfig) -> BlockSpec:
    head_dim = (
        args.head_dim
        if args.head_dim is not None
        else args.hidden_size // args.num_attention_heads
    )
    scaling_alpha = 1.0
    if args.rope_scaling and "alpha" in args.rope_scaling:
        scaling_alpha = args.rope_scaling["alpha"]
    rope = DynamicNTKAlphaRoPE(
        head_dim, base=args.rope_theta, scaling_alpha=scaling_alpha
    )
    return BlockSpec(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        intermediate_size=args.intermediate_size,
        rope=rope,
        rms_norm_eps=args.rms_norm_eps,
        attn_bias=args.attention_bias,
        qk_norm=args.use_qk_norm,
        qk_norm_post_rope=True,
        qk_norm_names=("query_layernorm", "key_layernorm"),
    )


class HunyuanV1DenseModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(block_spec(args)) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = HunyuanV1DenseModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        out = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return (
            self.args.head_dim or self.args.hidden_size // self.args.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
