from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlx_vlm.models.rope_utils as rope_utils
from mlx_vlm.models.rope_utils import (
    EagerRoPE,
    MRoPERotaryEmbedding,
    ProportionalRoPE,
    YarnRoPE,
    apply_mrope_frequency_layout,
    apply_multimodal_rotary_pos_emb,
    compute_mrope_frequencies,
    initialize_rope,
    mrope_position_selector,
)


def _max_diff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def _assert_pair_close(actual, expected, *, atol=1e-4):
    mx.eval(*actual, *expected)
    assert _max_diff(actual[0], expected[0]) < atol
    assert _max_diff(actual[1], expected[1]) < atol


def _disable_metal_fast_path(fn):
    has_metal = rope_utils._HAS_METAL
    rope_utils._HAS_METAL = False
    try:
        return fn()
    finally:
        rope_utils._HAS_METAL = has_metal


def _position_ids(batch=2, seq_len=4):
    base = mx.arange(batch * seq_len, dtype=mx.int32).reshape(batch, seq_len)
    return mx.stack([base, base + 3, base + 7])


def test_eager_rope_per_batch_offset_does_not_expand_seq():
    """Per-batch (array) offsets from batch caches must keep S, not grow it.

    Regression for the batched-path crash on multi-request streaming:
    ``_eager_rope_angles`` used ``arange(S) + offset`` which broadcast the
    batch-sized offset along the seq dim (S=1 -> S=B), so the attention mask
    (N=1) no longer matched the keys (S=B) and
    ``mx.fast.scaled_dot_product_attention`` raised
    ``Shapes (B,1,1,window) and (B,H,B,window+1) cannot be broadcast``.
    """
    rope = initialize_rope(
        dims=128,
        base=500000.0,
        traditional=False,
        scaling_config={"rope_type": "default"},
        implementation="eager",
    )
    assert isinstance(rope, EagerRoPE)

    rng = mx.random.key(0)
    x = mx.random.normal(key=rng, shape=(2, 32, 1, 128)).astype(mx.float32)
    offsets = mx.array([86, 34258], dtype=mx.int32)

    out = rope(x, offset=offsets)
    assert out.shape == x.shape, f"seq dim expanded: {out.shape} != {x.shape}"

    # Per-batch array offset must equal per-row scalar application.
    ref = mx.concatenate([rope(x[0:1], offset=86), rope(x[1:2], offset=34258)], axis=0)
    assert bool(mx.array_equal(out, ref).item())

    # Multi-token batch: (B, S) positions, still per-row equal to scalars.
    x2 = mx.random.normal(key=mx.random.key(1), shape=(2, 32, 3, 128)).astype(
        mx.float32
    )
    out2 = rope(x2, offset=offsets)
    ref2 = mx.concatenate(
        [rope(x2[0:1], offset=86), rope(x2[1:2], offset=34258)], axis=0
    )
    assert bool(mx.array_equal(out2, ref2).item())

    # Traditional layout follows the same rule.
    rope_t = initialize_rope(
        dims=128,
        base=500000.0,
        traditional=True,
        scaling_config={"rope_type": "default"},
        implementation="eager",
    )
    out3 = rope_t(x, offset=offsets)
    assert out3.shape == x.shape
    ref3 = mx.concatenate(
        [rope_t(x[0:1], offset=86), rope_t(x[1:2], offset=34258)], axis=0
    )
    assert bool(mx.array_equal(out3, ref3).item())

    # 0-d array offset (mx.array(86)) takes the same single code path as a
    # scalar int and must stay bit-identical to it.
    out4 = rope(x, offset=mx.array(86, dtype=mx.float32))
    ref4 = rope(x, offset=86)
    assert bool(mx.array_equal(out4, ref4).item())


def test_proportional_rope_evals_private_helper_arrays_on_init(monkeypatch):
    eval_args = []
    monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = ProportionalRoPE(
                dims=8, scaling_config={"partial_rotary_factor": 0.25}
            )

    host = Host()

    assert isinstance(host.rope, nn.Module)
    assert tree_flatten(host.parameters()) == []
    assert tree_flatten(host.trainable_parameters()) == []
    eager_arrays = host.rope.eager_eval_arrays()
    assert eager_arrays[0] is host.rope.freqs
    assert len(eval_args) == 1
    assert eval_args[0][0] is eager_arrays[0]


# TODO: Refactor this file into separate test classes for each RoPE variant.
@pytest.mark.parametrize(
    "style",
    [
        "chunked",
        "interleaved",
        "sectioned_half_split",
        "sectioned_even_odd",
        "split_select",
    ],
)
def test_mrope_apply_rotary_fast_path_matches_fallback(style):
    mx.random.seed(0)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    position_ids = _position_ids()
    kwargs = {"dim": 8, "base": 10000, "mrope_section": [2, 1, 1], "style": style}

    rotary = MRoPERotaryEmbedding(**kwargs)
    fast = rotary.apply_rotary(q, k, position_ids)

    fallback = MRoPERotaryEmbedding(**kwargs)
    fallback.fused_apply = False
    expected = fallback.apply_rotary(q, k, position_ids)

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


def test_sectioned_mrope_requires_three_sections():
    q = mx.zeros((1, 1, 2, 8))
    k = mx.zeros((1, 1, 2, 8))
    cos = mx.zeros((3, 1, 2, 8))
    sin = mx.zeros((3, 1, 2, 8))

    with pytest.raises(ValueError, match="exactly 3 sections"):
        _disable_metal_fast_path(
            lambda: apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, mrope_section=[2, 2], style="sectioned_half_split"
            )
        )


@pytest.mark.parametrize(
    ("style", "expected_selector"),
    [
        ("chunked", [0, 0, 1, 1, 2, 2]),
        ("interleaved", [0, 1, 2, 0, 1, 2]),
        ("split_select", [0, 0, 1, 1, 2, 2]),
    ],
)
def test_selected_frequency_fast_path_matches_layout_helper(style, expected_selector):
    mx.random.seed(3)
    position_ids = _position_ids(batch=2, seq_len=4)
    mrope_section = [2, 2, 2]
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector = mrope_position_selector(style, mrope_section, inv_freq.shape[0])
    assert position_selector.tolist() == expected_selector

    fast = compute_mrope_frequencies(
        position_ids,
        inv_freq,
        mrope_section,
        style=style,
        position_selector=position_selector,
    )
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    layout = apply_mrope_frequency_layout(freqs, mrope_section, style=style)

    mx.eval(fast, layout)
    assert _max_diff(fast, layout) < 1e-4


def _build_on_worker(factory):
    """Build on one thread, use on another, as the server does."""
    with ThreadPoolExecutor(max_workers=1) as loader:
        rope = loader.submit(factory).result()

    def use():
        return mx.eval(rope(mx.ones((1, 2, 4, 8))))

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(use).result()


def test_yarn_rope_runs_on_a_thread_it_was_not_built_on():
    _build_on_worker(lambda: YarnRoPE(dims=8, traditional=False, base=10000.0))
