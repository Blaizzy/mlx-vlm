from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from ..transformer_block import BlockSpec, TransformerBlock
from .config import ModelConfig


def block_spec(args: ModelConfig, is_local: Optional[bool]) -> BlockSpec:
    head_dim = args.head_dim
    rope = initialize_rope(
        head_dim,
        base=args.rope_theta,
        traditional=False,
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
        qk_norm=True,
        use_rope=is_local is None or is_local,
        use_sliding=bool(is_local),
    )


class ExaoneModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.sliding_window_pattern
        self.layers = [
            TransformerBlock(
                block_spec(
                    args,
                    is_local=pattern[i % len(pattern)] == "L" if pattern else None,
                )
            )
            for i in range(args.num_hidden_layers)
        ]
        if pattern:
            self.swa_idx = pattern.index("L")
            self.full_idx = pattern.index("G")
        else:
            self.swa_idx = None
            self.full_idx = 0

        self.window_size = args.sliding_window
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)
        global_mask = create_attention_mask(h, cache[self.full_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.window_size
            )
        else:
            swa_mask = None

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else global_mask
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExaoneModel(args)
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

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if l.use_sliding
                else KVCache()
            )
            for l in self.layers
        ]

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        return weights
