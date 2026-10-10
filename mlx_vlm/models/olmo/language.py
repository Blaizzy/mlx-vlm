from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import LanguageModelOutput, create_attention_mask
from ..base import scaled_dot_product_attention as shared_scaled_dot_product_attention
from ..cache import KVCache
from .config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        dim = args.d_model

        self.ff_proj = nn.Linear(dim, args.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(args.mlp_hidden_size // 2, dim, bias=False)
        self.att_norm = nn.LayerNorm(dim, affine=False)
        self.ff_norm = nn.LayerNorm(dim, affine=False)

        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        self.att_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def attend(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, hidden_size = x.shape
        queries, keys, values = mx.split(self.att_proj(x), 3, axis=-1)
        queries = queries.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if isinstance(keys, mx.array):
            scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
            if mask is not None:
                if isinstance(mask, str):
                    query_length, key_length = scores.shape[-2:]
                    query_positions = mx.arange(key_length - query_length, key_length)
                    key_positions = mx.arange(key_length)
                    mask = query_positions[:, None] >= key_positions[None]
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores += mask
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                scores.dtype
            )
            output = probabilities @ values
        else:
            output = shared_scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.attn_out(output)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attend(self.att_norm(x), mask, cache)
        first, second = mx.split(self.ff_proj(self.ff_norm(h)), 2, axis=-1)
        return h + self.ff_out(swiglu(second, first))


class Transformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.weight_tying = args.weight_tying
        self.wte = nn.Embedding(args.embedding_size, args.d_model)
        self.blocks = [TransformerBlock(args) for _ in range(args.n_layers)]
        if not self.weight_tying:
            self.ff_out = nn.Linear(args.d_model, args.embedding_size, bias=False)
        self.norm = nn.LayerNorm(args.d_model, affine=False)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.wte(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.blocks)
        mask = create_attention_mask(h, cache[0])

        for block, layer_cache in zip(self.blocks, cache):
            h = block(h, mask, layer_cache)

        h = self.norm(h)
        if self.weight_tying:
            return self.wte.as_linear(h)
        return self.ff_out(h)


class OlmoModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.transformer = Transformer(args)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        return self.transformer(inputs, cache, inputs_embeds)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = OlmoModel(args)

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
        logits = self.model(inputs, cache, inputs_embeds)
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.transformer.blocks

    def make_cache(self):
        return [KVCache() for _ in self.layers]
