import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlx_vlm.models.rope_utils as rope_utils
from mlx_vlm.models.gemma4.rope_utils import ProportionalRoPE
from mlx_vlm.models.rope_utils import (
    MRoPERotaryEmbedding,
    apply_mrope_frequency_layout,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_even_odd,
    compute_mrope_frequencies,
    compute_selected_mrope_cos_sin,
    mrope_position_selector,
    mrope_section_selectors,
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


def test_mrope_rotary_embedding_evals_private_helper_arrays_on_init(monkeypatch):
    eval_args = []
    monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = MRoPERotaryEmbedding(
                dim=8,
                mrope_section=[2, 1, 1],
                style="interleaved",
            )

    host = Host()

    assert isinstance(host.rotary_emb, nn.Module)
    assert tree_flatten(host.parameters()) == []
    assert tree_flatten(host.trainable_parameters()) == []
    eager_arrays = host.rotary_emb.eager_eval_arrays()
    assert eager_arrays[0] is host.rotary_emb.inv_freq
    assert eager_arrays[1] is host.rotary_emb.position_selector
    assert len(eval_args) == 1
    assert eval_args[0][0] is eager_arrays[0]
    assert eval_args[0][1] is eager_arrays[1]


def test_proportional_rope_evals_private_helper_arrays_on_init(monkeypatch):
    eval_args = []
    monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = ProportionalRoPE(
                dims=8,
                scaling_config={"partial_rotary_factor": 0.25},
            )

    host = Host()

    assert isinstance(host.rope, nn.Module)
    assert tree_flatten(host.parameters()) == []
    assert tree_flatten(host.trainable_parameters()) == []
    eager_arrays = host.rope.eager_eval_arrays()
    assert eager_arrays[0] is host.rope.freqs
    assert len(eval_args) == 1
    assert eval_args[0][0] is eager_arrays[0]


def test_proportional_rope_matches_zero_padded_rotate_half_layout():
    x = (mx.arange(2 * 3 * 8).reshape(1, 2, 3, 8) / 10).astype(mx.float32)
    rope = ProportionalRoPE(
        dims=8,
        base=10000,
        scaling_config={"partial_rotary_factor": 0.5, "factor": 2.0},
    )

    actual = rope(x, offset=4)

    inv_freq = 1.0 / mx.power(10000, mx.arange(0, 4, 2, dtype=mx.float32) / 8)
    inv_freq = mx.concatenate([inv_freq / 2.0, mx.zeros((2,), dtype=mx.float32)])
    positions = mx.arange(3, dtype=mx.float32) + 4
    freqs = positions[:, None] * inv_freq
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb).reshape(1, 1, 3, 8)
    sin = mx.sin(emb).reshape(1, 1, 3, 8)
    rotated = mx.concatenate([-x[..., 4:], x[..., :4]], axis=-1)
    expected = (x * cos) + (rotated * sin)

    mx.eval(actual, expected)
    assert _max_diff(actual, expected) < 1e-5
    assert _max_diff(actual[..., 2:4], x[..., 2:4]) < 1e-5
    assert _max_diff(actual[..., 6:8], x[..., 6:8]) < 1e-5


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
    kwargs = {
        "dim": 8,
        "base": 10000,
        "mrope_section": [2, 1, 1],
        "style": style,
    }

    rotary = MRoPERotaryEmbedding(**kwargs)
    fast = rotary.apply_rotary(q, k, position_ids)

    fallback = MRoPERotaryEmbedding(**kwargs)
    fallback.fused_apply = False
    expected = fallback.apply_rotary(q, k, position_ids)

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


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
def test_mrope_apply_rotary_fallback_routes_style_to_shared_apply(monkeypatch, style):
    q = (mx.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10) / 100).astype(mx.float32)
    k = (mx.arange(2 * 2 * 4 * 10).reshape(2, 2, 4, 10) / 80).astype(mx.float32)
    position_ids = _position_ids()
    calls = []
    sentinel = (q + 1, k + 1)

    def fake_apply(q_arg, k_arg, cos, sin, **kwargs):
        calls.append(
            {
                "q": q_arg,
                "k": k_arg,
                "cos_shape": cos.shape,
                "sin_shape": sin.shape,
                **kwargs,
            }
        )
        return sentinel

    monkeypatch.setattr(
        rope_utils,
        "apply_multimodal_rotary_pos_emb",
        fake_apply,
    )
    kwargs = {
        "dim": 8,
        "base": 10000,
        "mrope_section": [2, 1, 1],
        "style": style,
    }

    rotary = MRoPERotaryEmbedding(**kwargs)
    rotary.fused_apply = False
    actual = rotary.apply_rotary(
        q,
        k,
        position_ids,
        unsqueeze_dim=2,
        cast_output=False,
    )

    assert actual is sentinel
    assert len(calls) == 1
    call = calls[0]
    assert call["q"] is q
    assert call["k"] is k
    assert call["mrope_section"] == kwargs["mrope_section"]
    assert call["style"] == style
    assert call["unsqueeze_dim"] == 2
    assert call["cast_output"] is False
    assert call["cos_shape"] == call["sin_shape"]


@pytest.mark.parametrize("style", ["sectioned_half_split", "sectioned_even_odd"])
def test_sectioned_precomputed_rotary_fast_path_matches_fallback(style):
    mx.random.seed(1)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    cos = mx.random.normal((3, 2, 4, 8)).astype(mx.float32)
    sin = mx.random.normal((3, 2, 4, 8)).astype(mx.float32)
    kwargs = {"mrope_section": [2, 1, 1], "style": style}

    fast = apply_multimodal_rotary_pos_emb(q, k, cos, sin, **kwargs)
    expected = _disable_metal_fast_path(
        lambda: apply_multimodal_rotary_pos_emb(q, k, cos, sin, **kwargs)
    )

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
                q,
                k,
                cos,
                sin,
                mrope_section=[2, 2],
                style="sectioned_half_split",
            )
        )


@pytest.mark.parametrize("cos_layout", ["half", "full"])
def test_even_odd_precomputed_rotary_fast_path_matches_fallback(cos_layout):
    mx.random.seed(2)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    cos = mx.random.normal((2, 4, 8)).astype(mx.float32)
    sin = mx.random.normal((2, 4, 8)).astype(mx.float32)

    fast = apply_rotary_pos_emb_even_odd(q, k, cos, sin, cos_layout=cos_layout)
    expected = _disable_metal_fast_path(
        lambda: apply_rotary_pos_emb_even_odd(q, k, cos, sin, cos_layout=cos_layout)
    )

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


@pytest.mark.parametrize(
    ("style", "expected_selector"),
    [
        ("chunked", [0, 0, 1, 1, 2, 2]),
        ("interleaved", [0, 1, 2, 0, 1, 2]),
        ("split_select", [0, 0, 1, 1, 2, 2]),
    ],
)
def test_selected_frequency_fast_path_matches_layout_helper(
    style,
    expected_selector,
):
    mx.random.seed(3)
    position_ids = _position_ids(batch=2, seq_len=4)
    mrope_section = [2, 2, 2]
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector = mrope_position_selector(
        style,
        mrope_section,
        inv_freq.shape[0],
    )
    assert position_selector.tolist() == expected_selector

    fast = compute_mrope_frequencies(
        position_ids,
        inv_freq,
        mrope_section,
        style=style,
        position_selector=position_selector,
    )
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    layout = apply_mrope_frequency_layout(
        freqs,
        mrope_section,
        style=style,
    )

    mx.eval(fast, layout)
    assert _max_diff(fast, layout) < 1e-4


def test_mrope_section_selectors_interleave_selected_sections():
    position_selector, frequency_selector = mrope_section_selectors(
        [2, 2, 2],
        position_axes=(1, 2, 0),
        interleave_sections=(0, 1),
    )

    assert position_selector.tolist() == [1, 2, 1, 2, 0, 0]
    assert frequency_selector.tolist() == [0, 2, 1, 3, 4, 5]


def test_selected_mrope_cos_sin_applies_section_selectors():
    mx.random.seed(4)
    mrope_section = [2, 2, 2]
    position_axes = (1, 2, 0)
    interleave_sections = (0, 1)
    position_ids = _position_ids(batch=2, seq_len=4)
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector, frequency_selector = mrope_section_selectors(
        mrope_section,
        position_axes=position_axes,
        interleave_sections=interleave_sections,
    )

    fast = compute_selected_mrope_cos_sin(
        position_ids,
        inv_freq,
        position_selector,
        frequency_selector,
    )
    selected_positions = mx.take(position_ids, position_selector, axis=0).transpose(
        1, 2, 0
    )
    freqs = selected_positions.astype(mx.float32) * mx.take(
        inv_freq,
        frequency_selector,
    )
    emb = mx.repeat(freqs, repeats=2, axis=-1)
    expected = mx.cos(emb), mx.sin(emb)

    _assert_pair_close(fast, expected)
