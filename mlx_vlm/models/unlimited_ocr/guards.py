"""Generation guards for Unlimited-OCR-2.

The native stack never decodes unguarded: ``infer_multi`` applies a page-count guard by default and the
official SGLang client / ``infer.py`` add a no-repeat n-gram processor. Without them one wrong token can
send greedy decoding into a loop ("The document is a key to the development of the document." ...) that
runs to the token budget and loses every later page. The rules below are the native ones
(``_uocr_page_count_guard`` / ``_uocr_no_repeat_ngram``), as mlx-vlm logits processors over the
generated ids only.

* page-count guard: EOS is banned until every prompt ``<PAGE>`` has been answered, EOS is forced once
  there are more answered pages than prompt pages, and a page longer than ``max_page_tokens`` is cut by
  forcing the next ``<PAGE>`` (or EOS after the last one).
* no-repeat n-gram: a token that would complete an n-gram already seen in the last ``ngram_window``
  generated tokens is banned.

All state lives in MLX arrays and is updated lazily, so a step never waits on the host: reading the
sampled token on the host every step costs ~10% decode speed (it stops the next step from being built
while the GPU is still busy), the lazy version ~1-2%.
"""

from typing import Iterable, List, Optional

import mlx.core as mx
import numpy as np

# Native defaults: ``infer_multi(page_count_guard=True, max_page_tokens=20000)`` and the evaluation
# setup in ``infer.py`` (n-gram 35, window 1024 for several pages, 128 for one).
DEFAULT_MAX_PAGE_TOKENS = 20000
DEFAULT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW_MULTI = 1024
DEFAULT_NGRAM_WINDOW_SINGLE = 128


class UnlimitedOCRGuards:
    """Page-count guard + no-repeat n-gram for one request (``(tokens, logits) -> logits``).

    ``tokens`` is the prompt followed by the generated ids; the first call sees the prompt only and
    every later call exactly one more generated id. ``expected_pages`` is the number of ``<PAGE>``
    markers in the prompt (counted from the prompt at the first call when not given).
    """

    def __init__(
        self,
        page_token_id: Optional[int],
        eos_token_id: int,
        *,
        expected_pages: Optional[int] = None,
        page_count_guard: bool = True,
        max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
        ngram_size: int = 0,
        ngram_window: int = DEFAULT_NGRAM_WINDOW_MULTI,
        ngram_whitelist: Iterable[int] = (),
    ):
        self.page_token_id = page_token_id
        self.eos_token_id = int(eos_token_id)
        self.page_count_guard = bool(page_count_guard) and page_token_id is not None
        self.expected_pages = expected_pages
        self.max_page_tokens = int(max_page_tokens)
        self.ngram_size = int(ngram_size)
        self.ngram_window = int(ngram_window)
        self.ngram_whitelist = tuple(int(t) for t in ngram_whitelist)

        self._started = False
        # n-gram starts that fit in the window: the n-gram at ``s`` is buf[s : s + n]
        n, window = self.ngram_size, self.ngram_window
        self._ngram_on = n > 0 and window >= n
        if self._ngram_on:
            starts = window - n + 1
            self._prefix_idx = mx.arange(starts)[:, None] + mx.arange(n - 1)[None, :]
        # lazily updated state (mx arrays)
        self.markers = self.page_len = self.count = self.buf = None

    # -- state -----------------------------------------------------------

    def _init_state(self, tokens: mx.array, vocab: int):
        self.markers = mx.array(0, dtype=mx.int32)
        self.page_len = mx.array(0, dtype=mx.int32)
        self.count = mx.array(0, dtype=mx.int32)
        if self._ngram_on:
            self.buf = mx.full((self.ngram_window,), -1, dtype=mx.int32)
        self._vocab_ids = mx.arange(vocab)
        if self.page_count_guard and self.expected_pages is None:
            self.expected_pages = int((tokens == self.page_token_id).sum().item())
        self._keep = None
        if self.ngram_whitelist:
            keep = np.ones(vocab, dtype=bool)
            keep[[t for t in self.ngram_whitelist if 0 <= t < vocab]] = False
            self._keep = mx.array(keep)

    def _advance(self, last: mx.array):
        last = last.astype(mx.int32)
        if self.page_token_id is not None:
            is_page = last == self.page_token_id
            self.markers = self.markers + is_page.astype(mx.int32)
            self.page_len = mx.where(is_page, 0, self.page_len + 1)
        self.count = self.count + 1
        if self._ngram_on:
            self.buf = mx.concatenate([self.buf[1:], last[None]])

    # -- rules -----------------------------------------------------------

    def _ngram_banned(self, vocab: int) -> mx.array:
        """[vocab] bool: tokens completing an n-gram seen in the window (native semantics)."""
        n = self.ngram_size
        current = self.buf[self.ngram_window - n + 1 :]
        same_prefix = (self.buf[self._prefix_idx] == current).all(axis=-1)
        following = self.buf[n - 1 :]
        # padding (-1) never matches a real prefix, but a short history must not match at all
        hit = same_prefix & (following >= 0) & (self.count >= n)
        counts = mx.zeros((vocab,), dtype=mx.int32)
        counts = counts.at[mx.maximum(following, 0)].add(hit.astype(mx.int32))
        banned = counts > 0
        if self._keep is not None:
            banned = banned & self._keep
        return banned

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        vocab = logits.shape[-1]
        if not self._started:
            self._started = True
            self._init_state(tokens, vocab)
        else:
            self._advance(tokens[-1])

        unbanned = logits
        if self._ngram_on:
            logits = mx.where(self._ngram_banned(vocab), -mx.inf, logits)

        if self.page_count_guard and self.expected_pages > 0:
            expected = self.expected_pages
            ids = self._vocab_ids
            over = self.markers > expected
            capped = (
                self.page_len >= self.max_page_tokens
                if self.max_page_tokens > 0
                else mx.array(False)
            )
            # A forced token is never erased by an n-gram ban.
            forced = mx.where(
                ids
                == mx.where(
                    self.markers < expected, self.page_token_id, self.eos_token_id
                ),
                unbanned,
                -mx.inf,
            )
            pending = self.markers < expected
            banned_eos = mx.where(pending & (ids == self.eos_token_id), -mx.inf, logits)
            logits = mx.where(over | capped, forced, banned_eos)
        return logits


def make_guards(
    input_ids,
    page_token_id: Optional[int],
    eos_token_id,
    kwargs: Optional[dict] = None,
) -> List[UnlimitedOCRGuards]:
    """Per-request guards for ``input_ids``; consumes the guard options from ``kwargs``.

    Options (defaults in parentheses): ``page_count_guard`` (True), ``max_page_tokens`` (20000),
    ``no_repeat_ngram_size`` (35; 0 disables), ``ngram_window`` (1024 for several pages, else 128),
    ``ngram_whitelist_token_ids`` (none).
    """
    keys = (
        "page_count_guard",
        "max_page_tokens",
        "no_repeat_ngram_size",
        "ngram_window",
        "ngram_whitelist_token_ids",
    )
    opts = {k: kwargs.pop(k) for k in keys if k in kwargs} if kwargs is not None else {}
    if page_token_id is None:
        return []
    eos = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
    n_pages = int((np.asarray(input_ids).reshape(-1) == page_token_id).sum())
    default_window = (
        DEFAULT_NGRAM_WINDOW_MULTI if n_pages > 1 else DEFAULT_NGRAM_WINDOW_SINGLE
    )
    guards = UnlimitedOCRGuards(
        page_token_id,
        eos,
        expected_pages=n_pages,
        page_count_guard=opts.get("page_count_guard", True),
        max_page_tokens=opts.get("max_page_tokens", DEFAULT_MAX_PAGE_TOKENS),
        ngram_size=opts.get("no_repeat_ngram_size", DEFAULT_NGRAM_SIZE),
        ngram_window=opts.get("ngram_window", default_window),
        ngram_whitelist=opts.get("ngram_whitelist_token_ids") or (),
    )
    # Fully disabled: no processor, so decoding keeps its async pipelining.
    active = (guards.page_count_guard and n_pages > 0) or guards._ngram_on
    return [guards] if active else []


__all__ = ["UnlimitedOCRGuards", "make_guards"]
