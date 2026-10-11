"""Muse layer-specific RoPE settings must reach the selected implementation."""

import copy

import mlx.core as mx
import pytest

from mlx_vlm.models.muse_glimmer.config import TextConfig
from mlx_vlm.models.muse_glimmer.language import Attention
from mlx_vlm.models.rope_utils import EagerRoPE, YarnRoPE, initialize_rope


def config(**kwargs):
    return TextConfig(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        **kwargs,
    )


def test_default_preserves_eager_rope_and_layer_theta():
    args = config(layer_rope_theta=[123456.0, 500000.0, 500000.0, 0])
    attention = Attention(args, 0)
    assert isinstance(attention.rope, EagerRoPE)
    assert attention.rope.base == 123456.0
    assert attention.rope.scale == 1.0


def test_linear_scaling_honors_factor_without_mutating_config():
    args = config(rope_parameters={"rope_type": "linear", "factor": 8.0})
    before = copy.deepcopy(args.rope_parameters)
    attention = Attention(args, 0)
    assert isinstance(attention.rope, EagerRoPE)
    assert attention.rope.scale == 0.125
    assert args.rope_parameters == before


@pytest.mark.parametrize("offset", [0, 131071, 524287, 999999])
def test_yarn_matches_shared_implementation_at_large_offsets(offset):
    settings = {
        "rope_type": "yarn",
        "rope_theta": 500000.0,
        "factor": 8.0,
        "original_max_position_embeddings": 131072,
    }
    args = config(max_position_embeddings=1048576, rope_parameters=settings)
    before = copy.deepcopy(args.rope_parameters)
    attention = Attention(args, 0)
    assert isinstance(attention.rope, YarnRoPE)
    reference = initialize_rope(
        128,
        base=500000.0,
        traditional=False,
        scaling_config=settings,
        max_position_embeddings=1048576,
    )
    x = mx.arange(256, dtype=mx.float32).reshape(1, 1, 2, 128) / 256
    actual = attention.rope(x, offset=offset)
    expected = reference(x, offset=offset)
    mx.eval(actual, expected)
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.allclose(actual, expected).item()
    assert args.rope_parameters == before


def test_full_attention_keeps_rope_disabled():
    args = config(
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 131072,
        }
    )
    attention = Attention(args, 3)
    assert not attention.use_rope
    assert not attention.is_sliding
    assert isinstance(attention.rope, EagerRoPE)


def test_unknown_scaling_fails_instead_of_silently_ignoring_it():
    args = config(rope_parameters={"rope_type": "unsupported"})
    with pytest.raises(ValueError):
        Attention(args, 0)
