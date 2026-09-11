from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig


def _is_prime(n: int) -> bool:
    """Trial-division primality for table-size integers."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def find_next_prime(start: int, seen_primes: set) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def _raw_backend(tokenizer):
    """Unwrap to the raw `tokenizers` backend.

    HF wrappers (including transformers 5.x `TokenizersBackend`) report
    different lengths and decode paths than the raw backend the reference
    builds the map with; hashing must use the raw object to match.
    """
    for attr in ("backend_tokenizer", "_tokenizer"):
        backend = getattr(tokenizer, attr, None)
        if backend is not None and hasattr(backend, "decode"):
            return backend
    return tokenizer


def build_compressed_token_map(tokenizer):
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = getattr(tokenizer, "backend_tokenizer", None) or _raw_backend(tokenizer)
    try:
        n_tokens = len(tokenizer)
    except TypeError:
        n_tokens = backend.get_vocab_size()
    key_to_new = {}
    lookup = [0] * n_tokens
    for token_id in range(n_tokens):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, tokenizer_vocab_size: int):
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(mx.array(values * 2 + 1))
    return mx.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram), each split
    over `n_heads` heads. Every (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: ModelConfig):
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = (
            config.engram_max_ngram_size,
            config.engram_n_heads,
        )
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=config.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with the `max_ngram_size - 1`
    tokens before it. Look-back stops at the start of the sequence and at any dead token (an image
    span), so an n-gram never spans one. The cache carries all of this across the prefill/decode split.
    """

    DEAD = -1

    def __init__(self, config: ModelConfig, layout: EngramLayout, tokenizer):
        super().__init__()
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError((vocab_size, config.engram_compressed_vocab_size))
        self.pad_id = token_map[config.engram_pad_token_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self._primes = mx.array(layout.primes, dtype=mx.int64)
        self._offsets = mx.array(np.array(offsets), dtype=mx.int64)
        self._multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self._token_map = mx.array(token_map, dtype=mx.int64)
        self._cache = None

    def __call__(self, input_ids: mx.array, start_pos: int, token_mask=None):
        """token_mask: [B, L], False for tokens that take no part in an n-gram (image spans).
        Returns the hash ids, shaped [B, L, n_engram_layers, n_hash_cols]."""
        batch, seqlen = input_ids.shape
        compressed = self._token_map[input_ids]
        if token_mask is not None:
            compressed = mx.where(token_mask, compressed, self.DEAD)
        need_len = start_pos + seqlen
        cache = self._cache
        if cache is None:
            cache = mx.zeros((batch, need_len), dtype=mx.int64)
        else:
            if cache.shape[0] < batch:
                cache = mx.concatenate(
                    [
                        cache,
                        mx.zeros(
                            (batch - cache.shape[0], cache.shape[1]), dtype=mx.int64
                        ),
                    ],
                    axis=0,
                )
            if cache.shape[1] < need_len:
                cache = mx.concatenate(
                    [
                        cache,
                        mx.zeros(
                            (cache.shape[0], need_len - cache.shape[1]), dtype=mx.int64
                        ),
                    ],
                    axis=1,
                )
        parts = []
        if start_pos > 0:
            parts.append(cache[:batch, :start_pos])
        parts.append(compressed.astype(mx.int64))
        if cache.shape[1] > need_len:
            parts.append(cache[:batch, need_len:])
        head = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if cache.shape[0] > batch:
            cache = mx.concatenate([head, cache[batch:]], axis=0)
        else:
            cache = head
        self._cache = cache
        cur = cache[:batch]

        positions = mx.broadcast_to(
            mx.arange(start_pos, start_pos + seqlen), (batch, seqlen)
        )
        tokens, blocked = [], mx.zeros_like(positions, dtype=mx.bool_)
        for shift in range(self.layout.max_ngram_size):
            idx = mx.clip(positions - shift, 0, cur.shape[1] - 1)
            source = mx.take(cur, idx, axis=1)
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(mx.where(blocked, self.pad_id, source))
        tokens = mx.stack(tokens, axis=-1)

        products = tokens[:, :, None, :] * self._multipliers[None, None, :, :]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = mx.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self._primes[:, i - 1])
        return mx.concatenate(hashes, axis=-1) + self._offsets


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, config: ModelConfig, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.head_dim = layout.head_dim
        self.clamp_value = 1e-6
        self.eps = config.rms_norm_eps

        self.embed = nn.Embedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((config.hc_mult, config.hidden_size))
        self.k_weight = mx.ones((config.hc_mult, config.hidden_size))

    def __call__(self, x: mx.array, hash_ids: mx.array, token_mask=None) -> mx.array:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched."""
        dtype = x.dtype
        n_cols = hash_ids.shape[-1]
        kv = self.wkv(
            self.embed(hash_ids).reshape(*hash_ids.shape[:-1], n_cols * self.head_dim)
        )
        key, value = mx.split(kv, [self.hc_mult * self.dim], axis=-1)
        key = key.astype(mx.float32).reshape(*key.shape[:-1], self.hc_mult, self.dim)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = x.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(mx.square(h), axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(mx.square(key), axis=-1) + self.eps
        )
        dot = mx.sum(h * weight * key, axis=-1) * rstd * self.dim**-0.5
        signed = mx.where(
            dot >= 0,
            mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value)),
            -mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value)),
        )
        gate = mx.sigmoid(signed)
        if token_mask is not None:
            gate = mx.where(token_mask[..., None], gate, 0)
        return (h + gate[..., None] * value.astype(mx.float32)[:, :, None, :]).astype(
            dtype
        )
