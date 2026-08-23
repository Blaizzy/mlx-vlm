"""Shared invariant check for incremental decode against a prompt cache.

Every autoregressive model is used two ways: a prompt is processed in one call
during prefill, and then tokens arrive one (or a few) at a time during decode,
each appended to a cache. Those two paths must agree. When they do not, the
model prefills correctly and then drifts as soon as it starts generating -- the
kind of bug that never shows up in a single forward pass and is easy to
misattribute to sampling.

Concretely, splitting a prompt across successive cached calls must produce the
same logits as one full forward over the whole prompt. Getting this wrong
usually means a cache offset, an attention mask, or a rotary position is
computed from the current chunk rather than from the running sequence length.

The check itself is mechanical, so it was being written out by hand in every
model test that cared -- build the model, run a full forward, run two cached
forwards, concatenate, compare. This module is the assertion side of that
convention, so a new architecture can state the invariant in one line.

Seed before constructing the model if the comparison needs to be reproducible;
this helper does not touch the global RNG.

Usage::

    from mlx_vlm.tests.cache_invariants import assert_cached_forward_matches_full

    full_logits, _ = assert_cached_forward_matches_full(model)

Pass an explicit prompt, or split it somewhere other than the middle::

    assert_cached_forward_matches_full(model, tokens=[1, 2, 3, 4, 5], splits=(1, 3))
"""

import mlx.core as mx

__all__ = ["assert_cached_forward_matches_full"]

DEFAULT_TOKENS = (1, 2, 3, 4)


def _chunks(length, splits):
    bounds = [0, *splits, length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def assert_cached_forward_matches_full(
    model, *, tokens=DEFAULT_TOKENS, splits=None, cache=None, atol=1e-5, context=None
):
    """Assert cached, chunked prefill matches a single full forward.

    Args:
        model: a model exposing ``make_cache()`` and returning an object with a
            ``logits`` attribute.
        tokens: the prompt, as a sequence of token ids or an ``mx.array`` of
            shape ``(1, T)``.
        splits: token offsets to break the prompt at. Defaults to a single
            split at the midpoint, which is what the hand-written checks used.
        cache: cache to decode into. Defaults to ``model.make_cache()``. Pass
            one explicitly to assert against its contents afterwards.
        atol: absolute tolerance for the logit comparison.
        context: label used in failure messages. Defaults to the model's class
            name.

    Returns:
        ``(full_logits, cached_logits)``, so callers can go on to make
        architecture-specific assertions without running the model again.

    Raises:
        AssertionError: if the two paths disagree in shape or in value.
    """
    if context is None:
        context = type(model).__name__

    inputs = tokens if isinstance(tokens, mx.array) else mx.array([list(tokens)])
    length = inputs.shape[1]
    if splits is None:
        splits = (length // 2,)
    if not all(0 < offset < length for offset in splits):
        raise ValueError(
            f"{context}: splits {tuple(splits)} must fall strictly inside a "
            f"prompt of {length} tokens"
        )

    full_logits = model(inputs).logits

    if cache is None:
        cache = model.make_cache()
    cached_logits = mx.concatenate(
        [
            model(inputs[:, start:stop], cache=cache).logits
            for start, stop in _chunks(length, splits)
        ],
        axis=1,
    )
    mx.eval(full_logits, cached_logits)

    if full_logits.shape != cached_logits.shape:
        raise AssertionError(
            f"{context}: chunked prefill produced {tuple(cached_logits.shape)} "
            f"logits but one full forward produced {tuple(full_logits.shape)}."
        )

    if not mx.allclose(full_logits, cached_logits, atol=atol).item():
        difference = mx.abs(full_logits - cached_logits)
        worst = mx.max(difference).item()
        position = int(mx.argmax(mx.max(difference, axis=-1)).item())
        raise AssertionError(
            f"{context}: chunked prefill diverged from a full forward by "
            f"{worst:.3g} (atol={atol:g}), first worst at token {position} of "
            f"{length}, split at {tuple(splits)}. A cache offset, attention "
            f"mask, or rotary position is probably derived from the current "
            f"chunk instead of the running sequence length."
        )

    return full_logits, cached_logits
