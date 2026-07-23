import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..transformer_block import BlockSpec, Gemma1pRMSNorm, TransformerBlock
from .config import ModelConfig


def _block_spec(args: ModelConfig) -> BlockSpec:
    head_dim = args.head_dim
    rope = nn.RoPE(head_dim, traditional=args.rope_traditional, base=args.rope_theta)
    return BlockSpec(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=head_dim,
        scale=1.0 / (args.query_pre_attn_scalar**0.5),
        intermediate_size=args.intermediate_size,
        rope=rope,
        norm_type="gemma",
        rms_norm_eps=args.rms_norm_eps,
        layout="sandwich",
        attn_bias=False,
        qk_norm=False,
        attn_logit_softcapping=args.attn_logit_softcapping,
        mlp_act=nn.gelu_approx,
        mlp_bias=False,
    )


class GemmaModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(_block_spec(args)) for _ in range(args.num_hidden_layers)
        ]
        self.norm = Gemma1pRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.model_type = args.model_type
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.layers]
