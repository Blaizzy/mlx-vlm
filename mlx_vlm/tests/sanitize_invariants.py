"""Shared invariant checks for model ``sanitize()`` implementations.

Since #1498 ("Always sanitize loaded weights"), ``load_model`` calls
``sanitize()`` on every checkpoint it loads, including ones that are already in
MLX layout. Weight conversion is therefore no longer guarded by the caller, and
each ``sanitize()`` has to detect for itself whether its input still needs
converting.

Concretely, every ``sanitize()`` must be **idempotent**: running it on its own
output must be a no-op. When it is not, an already-converted checkpoint gets
transposed (or norm-shifted, or split) a second time, and the model either dies
at weight-load with a shape mismatch or -- worse -- loads and generates garbage.

Most implementations already satisfy this by comparing against the target layout
before converting; ``check_array_shape`` is the usual helper. This module is the
assertion side of that convention, so new architectures can state the invariant
in one line instead of hand-rolling it.

Usage::

    from .sanitize_invariants import assert_sanitize_idempotent

    assert_sanitize_idempotent(model, hf_weights)

For a submodule sanitizer, pass it explicitly::

    assert_sanitize_idempotent(
        model, hf_weights, sanitize=model.thinker.audio_tower.sanitize
    )
"""

import mlx.core as mx

__all__ = ["assert_sanitize_idempotent"]


def _describe(value):
    shape = getattr(value, "shape", None)
    return f"shape {tuple(shape)}" if shape is not None else repr(value)


def _assert_same(first, second, key, context):
    first_shape = getattr(first, "shape", None)
    second_shape = getattr(second, "shape", None)

    if first_shape != second_shape:
        raise AssertionError(
            f"{context}: sanitize() changed {key!r} on a second pass -- "
            f"{_describe(first)} became {_describe(second)}. sanitize() must "
            f"detect already-converted weights and leave them alone."
        )

    if first_shape is None:
        if first != second:
            raise AssertionError(
                f"{context}: sanitize() changed the value of {key!r} on a "
                f"second pass ({first!r} -> {second!r})."
            )
        return

    if not mx.array_equal(first, second):
        raise AssertionError(
            f"{context}: sanitize() left {key!r} at {_describe(first)} but "
            f"changed its values on a second pass. A layout conversion is "
            f"probably being applied twice."
        )


def assert_sanitize_idempotent(model, hf_weights, *, sanitize=None, context=None):
    """Assert that ``sanitize()`` is a no-op on its own output.

    Args:
        model: the model (or submodule) whose ``sanitize`` is under test.
        hf_weights: a weights dict in the *source* (usually PyTorch) layout.
            Only the keys the sanitizer cares about need to be present.
        sanitize: the callable to exercise. Defaults to ``model.sanitize``.
        context: label used in failure messages. Defaults to the model's class
            name.

    Returns:
        The sanitized weights, so callers can make architecture-specific
        assertions about them without sanitizing a second time.

    Raises:
        AssertionError: if a second ``sanitize()`` pass adds keys, drops keys,
            or changes any value.
    """
    if sanitize is None:
        sanitize = model.sanitize
    if context is None:
        context = type(model).__name__

    once = sanitize(dict(hf_weights))
    twice = sanitize(dict(once))

    added = sorted(set(twice) - set(once))
    dropped = sorted(set(once) - set(twice))
    if added or dropped:
        raise AssertionError(
            f"{context}: a second sanitize() pass changed the key set "
            f"(added={added}, dropped={dropped}). sanitize() must be safe to "
            f"run on an already-converted checkpoint."
        )

    for key in once:
        _assert_same(once[key], twice[key], key, context)

    return once
