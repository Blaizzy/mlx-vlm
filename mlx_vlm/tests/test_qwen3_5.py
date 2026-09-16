"""Qwen3.5 weight sanitization and ragged attention fallbacks."""

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen3_5 import language as lang
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.qwen3_5 import Model

# Patch embedding layouts

PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"
SANITIZED_KEY = "vision_tower.patch_embed.proj.weight"


def _tiny_model(in_channels=3, temporal_patch_size=2, patch_size=4, hidden_size=8):
    text_config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision_config = VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    config = ModelConfig(
        text_config=text_config, vision_config=vision_config, model_type="qwen3_5"
    )
    return Model(config), vision_config


def test_patch_embed_is_transposed_from_ncdhw_to_ndhwc():
    """Qwen3.8 stores the Conv3d patch embed as NCDHW; MLX expects NDHWC."""
    model, vision_config = _tiny_model()
    expected = model.vision_tower.patch_embed.proj.weight.shape

    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({PATCH_EMBED_KEY: ncdhw})

    assert sanitized[SANITIZED_KEY].shape == expected


# Ragged decode launch fallbacks

# Qwen3.5-4B text config, whose head_dim=256 is what makes the pass-2 pipeline
# expensive enough to lose threads on an affected GPU.
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 256
PADS = [0, 8]  # two rows with different left padding -> a ragged batch
SCALE = HEAD_DIM**-0.5

# bf16 inputs, so the one-pass and two-pass accumulation orders differ in the
# last couple of mantissa bits.
TOLERANCE = 5e-3

REJECTION = ValueError(
    "Thread group size (1024) is greater than  the maximum allowed threads per "
    "threadgroup (896)."
)

_RAGGED_SDPA_ONLY = pytest.mark.skipif(
    not mx.metal.is_available(), reason="ragged decode is a Metal-only fast path"
)


@pytest.fixture
def clear_launchability_cache():
    """The cache is process-global; keep verdicts from leaking between tests."""
    lang._QWEN3_5_SDPA_LAUNCHABLE.clear()
    yield
    lang._QWEN3_5_SDPA_LAUNCHABLE.clear()


def two_pass_kv_len():
    """Smallest KV length where every row of the batch takes the two-pass plan.

    The planner's threshold depends on the GPU architecture, so probe it instead
    of hardcoding a length that only works on some parts.
    """
    for kv_len in (2048, 8192, 65536):
        plans = {
            lang._qwen3_5_sdpa_vector_plan(kv_len - pad, Q_HEADS, KV_HEADS)
            for pad in PADS
        }
        if len(plans) == 1 and next(iter(plans))[0] == "two_pass":
            return kv_len
    return None


def inputs(kv_len):
    mx.random.seed(0)
    batch = len(PADS)
    queries = mx.random.normal((batch, Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.normal((batch, KV_HEADS, kv_len, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((batch, KV_HEADS, kv_len, HEAD_DIM)).astype(mx.bfloat16)
    return queries, keys, values


def reference(queries, keys, values):
    """The portable per-pad-group path the caller uses when the fast path declines."""
    rows = [
        mx.fast.scaled_dot_product_attention(
            queries[i : i + 1],
            keys[i : i + 1, :, pad:],
            values[i : i + 1, :, pad:],
            scale=SCALE,
        )
        for i, pad in enumerate(PADS)
    ]
    return mx.concatenate(rows, axis=0)


def max_abs_diff(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def reject(monkeypatch, factory_name, launches):
    """Replace a kernel factory so its launches raise the way Metal does."""

    def factory(*args, **kwargs):
        def launch(**launch_kwargs):
            launches.append(factory_name)
            raise REJECTION

        return launch

    monkeypatch.setattr(lang, factory_name, factory)


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
def test_rejected_two_pass_degrades_to_one_pass(monkeypatch):
    kv_len = two_pass_kv_len()
    if kv_len is None:
        pytest.skip("no KV length on this GPU selects the two-pass plan")
    queries, keys, values = inputs(kv_len)
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_two_pass_2_kernel", [])

    out = lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)

    assert out is not None, "should fall back to the one-pass kernel, not decline"
    assert out.shape == (len(PADS), Q_HEADS, 1, HEAD_DIM)
    assert max_abs_diff(out, reference(queries, keys, values)) < TOLERANCE


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
def test_rejection_is_probed_once(monkeypatch):
    kv_len = two_pass_kv_len()
    if kv_len is None:
        pytest.skip("no KV length on this GPU selects the two-pass plan")
    queries, keys, values = inputs(kv_len)
    launches = []
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_two_pass_2_kernel", launches)

    for _ in range(3):
        assert (
            lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)
            is not None
        )

    assert launches == [
        "_qwen3_5_ragged_sdpa_two_pass_2_kernel"
    ], "a rejected pipeline must be probed once and remembered, not retried per call"


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
@pytest.mark.parametrize("kv_len", [512, 2048])
def test_launchable_pipelines_are_untouched(kv_len):
    queries, keys, values = inputs(kv_len)

    out = lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)

    if out is None:
        pytest.skip("this GPU declines the fast path for this shape")
    # Whichever kernel this GPU ends up on -- one-pass, two-pass, or one-pass
    # after a real rejection of the two-pass reduction -- the answer must match.
    assert max_abs_diff(out, reference(queries, keys, values)) < TOLERANCE
    assert lang._QWEN3_5_SDPA_LAUNCHABLE, "the dispatch should record what it probed"
