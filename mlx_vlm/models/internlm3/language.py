import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from ..transformer_block import BlockSpec, TransformerBlock
from .config import ModelConfig


class DynamicNTKScalingRoPE(nn.Module):
    """Implements the rotary positional encoding with Dynamic NTK scaling."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.dims = dims
        self.traditional = traditional
        self.scale = scale

    def extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}, max_position_embeddings={self.max_position_embeddings}, scaling_factor={self.scale}"

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        if seq_len > self.max_position_embeddings:
            base = self.original_base * (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))
        else:
            base = self.original_base

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )


def block_spec(args: ModelConfig) -> BlockSpec:
    head_dim = args.hidden_size // args.num_attention_heads
    rope_scale = (
        1 / args.rope_scaling["factor"]
        if args.rope_scaling is not None and args.rope_scaling["rope_type"] == "linear"
        else 2.0
    )
    rope = DynamicNTKScalingRoPE(
        head_dim,
        max_position_embeddings=args.max_position_embeddings,
        traditional=args.rope_traditional,
        base=args.rope_theta,
        scale=rope_scale,
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
        attn_bias=args.qkv_bias,
        mlp_bias=args.bias,
    )


class InternLM2Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        assert args.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(block_spec(args)) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternLM2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "attention.rope.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
