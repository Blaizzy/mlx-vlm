from functools import lru_cache, partial
from typing import Any, Optional

import mlx.core as mx

from ..deepseek_v32.language import Indexer


@lru_cache(maxsize=None)
def _partition_scores(top_k, masked):
    @partial(mx.compile, shapeless=True)
    def partition(q, k, weights, mask=None):
        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        scores = scores * weights.swapaxes(-1, -2)[..., None]
        scores = scores.sum(axis=1, keepdims=True)
        if masked:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-top_k, axis=-1)

    return partition


class HyV4Indexer(Indexer):
    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_norm(self.wk(x)).reshape(b, 1, s, self.head_dim)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if k.shape[2] <= self.index_topk:
            return None

        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        return _partition_scores(self.index_topk, mask is not None)(
            q, k, weights, mask
        )[..., -self.index_topk :]
