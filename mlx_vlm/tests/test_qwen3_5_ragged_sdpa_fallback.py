"""Ragged-decode fallbacks when a GPU rejects the kernels' 1024-thread launch.

``_qwen3_5_ragged_decode_attention`` dispatches the one-pass kernel and the
second pass of the two-pass plan with a fixed 1024 threads. Metal caps threads
per compiled *pipeline* rather than per device, so on some parts that launch is
illegal (observed at ``D_SIZE == 256`` on applegpu_g14d, where the pass-2
pipeline tops out at 896) and raises. Whether it happens is a property of the
GPU, so these tests inject the rejection rather than depending on the hardware:
they pin that a rejected launch degrades one step instead of propagating, that
the verdict is only probed once, and that nothing changes where the launch is
legal.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen3_5 import language as lang

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

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="ragged decode is a Metal-only fast path"
)


@pytest.fixture(autouse=True)
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


def test_rejected_one_pass_declines_to_the_portable_path(monkeypatch):
    queries, keys, values = inputs(512)
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_one_pass_kernel", [])
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_two_pass_2_kernel", [])

    out = lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)

    assert out is None, "with no launchable kernel it must let the caller fall back"


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
