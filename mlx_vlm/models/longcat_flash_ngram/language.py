from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import ArraysCache, CacheList, KVCache
from ..longcat_flash.language import LanguageModel as LongcatFlashLM
from ..longcat_flash.language import LongcatFlashDecoderLayer
from .config import ModelConfig


class NgramEmbedding(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.m = args.ngram_vocab_size_ratio * args.vocab_size
        self.k = args.emb_split_num
        self.n = args.emb_neighbor_num

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = args.hidden_size // num_embedders

        self.embedders = []
        self.post_projs = []
        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
            self.post_projs.append(nn.Linear(emb_dim, args.hidden_size, bias=False))
        self._compute_vocab_mods()

    def _compute_vocab_mods(self):
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right(self, x: mx.array, n: int) -> mx.array:
        if n <= 0:
            return x
        batch_size, seq_len = x.shape
        if seq_len <= n:
            return mx.zeros_like(x)
        return mx.concatenate(
            [mx.zeros((batch_size, n), dtype=x.dtype), x[..., :-n]], axis=-1
        )

    def _get_ngram_ids(
        self,
        input_ids: mx.array,
        shifted_ids: Dict[int, mx.array],
        vocab_mods: List[int],
        ngram: int,
    ) -> mx.array:
        ngram_ids = input_ids
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[-1]

        input_ids = input_ids.astype(mx.int64)
        if cache is not None:
            context = cache[0]
            if context is None:
                context = input_ids
            else:
                context = mx.concatenate([context, input_ids], axis=-1)
            cache[0] = context[..., max(0, context.shape[-1] - self.n + 1) :]
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)
        vocab_mods = self._vocab_mods

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right(context, i - 1)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

        return x / (1 + self.k * (self.n - 1))


class LongcatFlashNgramModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array],
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] + [(None, None)] * self.num_layers

        h = (
            self.ngram_embeddings(input_ids, cache=cache[0])
            if inputs_embeds is None
            else inputs_embeds
        )

        mask = create_attention_mask(h, cache[1][0], return_array=True)

        for layer, c in zip(self.layers, cache[1:]):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashNgramModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        out = self.model(inputs, inputs_embeds, cache)
        return LanguageModelOutput(logits=self.lm_head(out))

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        return LongcatFlashLM.quant_predicate.fget(self)

    @property
    def cast_predicate(self):
        return LongcatFlashLM.cast_predicate.fget(self)

    def sanitize(self, weights):
        weights = LongcatFlashLM.sanitize(self, weights)
        if "model.embed_tokens.weight" in weights:
            weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(
                "model.embed_tokens.weight"
            )
        return weights

    def make_cache(self):
        return [ArraysCache(size=1)] + [
            CacheList(KVCache(), KVCache()) for _ in self.model.layers
        ]

    @property
    def head_dim(self):
        return self.args.qk_rope_head_dim + self.args.qk_nope_head_dim

    @property
    def n_kv_heads(self):
        return 1

    def shard(self, group: Optional[mx.distributed.Group] = None):
        LongcatFlashLM.shard(self, group)
