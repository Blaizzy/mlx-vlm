from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from ..transformer_block import BlockSpec, TransformerBlock
from .config import ModelConfig


def block_spec(args: ModelConfig, layer_idx: int) -> BlockSpec:
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
    if args.layer_types[layer_idx] != "full_attention":
        rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)
    else:
        rope = initialize_rope(
            head_dim,
            traditional=False,
            base=args.rope_theta,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
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
        layout="post",
        attn_bias=args.attention_bias,
        qk_norm=True,
        qk_norm_full=True,
    )


class Olmo3Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.sliding_window = args.sliding_window

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(block_spec(args, i)) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.swa_idx = args.layer_types.index("sliding_attention")
        self.ga_idx = args.layer_types.index("full_attention")
        self.layer_types = args.layer_types

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.ga_idx])
        sliding_window_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.sliding_window
        )

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else sliding_window_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Olmo3Model(args)
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
        caches = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim or (
            self.args.hidden_size // self.args.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
