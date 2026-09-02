"""Eager vs fused equivalence for the GLM-5-Next KDA single-token decode step.

``MLX_VLM_GLM5_FUSED_KDA=1`` replaces the ~30 tiny dispatches that make up the
post-projection half of ``Glm5NextLinearAttention`` (conv1d window update, silu,
two L2 norms, the safe forget gate, beta, the gated delta rule and the gated
RMSNorm) with a single ``mx.fast.metal_kernel`` launch per layer.

The kernel is a rounding-faithful transcription, not an approximation: it is
expected to be *bit-identical* to the eager path, including the fp32 recurrent
state carried across steps.  These tests pin that, and pin that the fast path
correctly declines anything it is not written for (prefill, S>1, batched or
masked decode).

``cast_predicate`` does not pin ``A_log`` / ``dt_bias``, so a conversion may leave
them in float32 or in the model dtype -- and ``a + dt_bias`` promotes differently
in the two cases.  Parity is therefore checked for both.

``MLX_VLM_GLM5_FUSED_KDA_QPROJ=1`` additionally folds the two small quantized
projections (``f_b_proj`` / ``g_b_proj``) into the same launch.

Run the 34-layer micro-bench with ``python -m mlx_vlm.tests.test_glm5_next_fused_kda``.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.fused_kda import fused_kda_probe

# GLM-5.3-Flash text_config, restricted to what the KDA layer reads.  The kernel
# is parameterised by linear_num_heads / linear_head_dim / short_conv_kernel_size
# / gate_lower_bound / rms_norm_eps, so those are the live values verbatim.
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=4096,
    intermediate_size=12288,
    moe_intermediate_size=2048,
    num_hidden_layers=1,
    num_attention_heads=64,
    num_key_value_heads=64,
    n_shared_experts=1,
    n_routed_experts=288,
    routed_scaling_factor=2.5,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=0,
    v_head_dim=256,
    qk_nope_head_dim=256,
    num_experts_per_tok=8,
    first_k_dense_replace=3,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=32,
    layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 64,
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
    },
)


def _config():
    return TextConfig.from_dict(dict(_CFG))


def _layer(config, seed=0, gate_dtype=mx.float32):
    mx.random.seed(seed)
    layer = glm5.Glm5NextLinearAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    H, D = config.linear_num_heads, config.linear_head_dim
    layer.conv1d.weight = (mx.random.normal(layer.conv1d.weight.shape) * 0.5).astype(
        mx.bfloat16
    )
    # cast_predicate leaves these to the conversion: float32 or the model dtype.
    layer.forget_gate.A_log = (mx.random.normal((H,)) * 0.5).astype(gate_dtype)
    layer.forget_gate.dt_bias = (mx.random.normal((H * D,)) * 0.5).astype(gate_dtype)
    layer.o_norm.weight = (mx.ones((D,)) + 0.02 * mx.random.normal((D,))).astype(
        mx.bfloat16
    )
    # The live build quantises the KDA projections to 8-bit, group 64.
    nn.quantize(layer, group_size=64, bits=8)
    mx.eval(layer.parameters())
    return layer


def _cache(config, batch=1, seed=1):
    mx.random.seed(seed)
    H, D, K = (
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
    )
    cache = ArraysCache(size=2)
    # "Warmed" states: a decode step never sees the zero-initialised cache.
    cache[0] = (mx.random.normal((batch, K - 1, 3 * H * D)) * 0.3).astype(mx.bfloat16)
    cache[1] = (mx.random.normal((batch, H, D, D)) * 0.05).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def _clone(cache):
    out = ArraysCache(size=2)
    out[0] = mx.array(cache[0])
    out[1] = mx.array(cache[1])
    mx.eval(out[0], out[1])
    return out


def _max_abs_rel(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a32 - b32)
    return float(d.max()), float((d / mx.maximum(mx.abs(a32), 1e-30)).max())


@pytest.fixture(autouse=True)
def _reset_toggle():
    saved = (glm5._FUSED_KDA_ENV, glm5._FUSED_KDA_QPROJ_ENV)
    yield
    glm5._FUSED_KDA_ENV, glm5._FUSED_KDA_QPROJ_ENV = saved


def _set_toggle(layer, on, qproj=False):
    glm5._FUSED_KDA_ENV = on
    glm5._FUSED_KDA_QPROJ_ENV = qproj
    layer._fused_kda = None
    layer._fused_kda_qproj = None
    layer._fused_kda_ty = None
    layer._fused_kda_qproj_ty = None


def _require_device(config, kind, layer=None):
    """Skip where the GPU cannot run this pipeline at any supported size.

    Uses the same probe the runtime uses, so a skip here means the model would
    have taken the eager fallback rather than produced a wrong answer.  CI's
    virtualized runners cap the qproj pipeline at 640 threads/threadgroup.
    """
    kwargs = dict(
        kind=kind,
        num_heads=config.linear_num_heads,
        head_dim=config.linear_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        dtype=mx.bfloat16,
        state_dtype=mx.float32,
    )
    if kind == "qproj":
        kwargs["bits"] = int(layer.forget_gate.f_b_proj.bits)
        kwargs["group_size"] = int(layer.forget_gate.f_b_proj.group_size)
    if fused_kda_probe(**kwargs) is None:
        pytest.skip(f"device threadgroup cap below the {kind} kernel's requirement")


@pytest.mark.parametrize("gate_dtype", [mx.float32, mx.bfloat16])
def test_fused_kda_matches_eager_over_32_decode_steps(gate_dtype):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    _require_device(config, "base")
    layer = _layer(config, gate_dtype=gate_dtype)
    eager_cache = _cache(config)
    fused_cache = _clone(eager_cache)

    mx.random.seed(4321)
    steps = [
        mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(32)
    ]
    mx.eval(steps)

    worst = (0.0, 0.0)
    for x in steps:
        _set_toggle(layer, False)
        assert not layer._fused_kda_ready()
        eager_out = layer(x, None, eager_cache)
        _set_toggle(layer, True)
        # Without this the comparison could silently be eager-vs-eager.
        assert layer._fused_kda_ready()
        fused_out = layer(x, None, fused_cache)
        mx.eval(eager_out, fused_out, eager_cache.cache, fused_cache.cache)
        for ref, got in (
            (eager_out, fused_out),
            (eager_cache[0], fused_cache[0]),
            (eager_cache[1], fused_cache[1]),
        ):
            worst = max(worst, _max_abs_rel(ref, got))

    # The transcription reproduces every rounding point of the eager chain, so
    # this is exact rather than merely close.  Keep a bf16-scale tolerance in the
    # assertion so a future MLX rounding change degrades to "still fine" rather
    # than a hard failure, but report the real number.
    assert worst[0] <= 2.0**-6, f"max abs diff {worst[0]:.3e} (rel {worst[1]:.3e})"
    assert worst == (0.0, 0.0), f"expected bit-identical, got {worst}"


@pytest.mark.parametrize("gate_dtype", [mx.float32, mx.bfloat16])
def test_fused_kda_qproj_matches_eager_over_32_decode_steps(gate_dtype):
    """MLX_VLM_GLM5_FUSED_KDA_QPROJ folds f_b_proj / g_b_proj into the kernel.

    The in-kernel GEMV transcribes MLX's affine ``qmv_quad`` partition, so it is
    bit-identical too -- a plain per-element dequant dot instead disagrees on
    ~0.01% of elements, which was enough to flip greedy tokens on 2 of 5 seeds.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config, gate_dtype=gate_dtype)
    _require_device(config, "base")
    _require_device(config, "qproj", layer)
    _set_toggle(layer, True, qproj=True)
    if not layer._fused_kda_qproj_ready(mx.bfloat16, mx.float32):
        pytest.skip("projection fold unsupported for this quantization")

    eager_cache = _cache(config)
    qproj_cache = _clone(eager_cache)
    mx.random.seed(1357)
    steps = [
        mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        for _ in range(32)
    ]
    mx.eval(steps)

    worst = (0.0, 0.0)
    for x in steps:
        _set_toggle(layer, False)
        eager_out = layer(x, None, eager_cache)
        _set_toggle(layer, True, qproj=True)
        assert layer._fused_kda_qproj_ready(mx.bfloat16, mx.float32)
        qproj_out = layer(x, None, qproj_cache)
        mx.eval(eager_out, qproj_out, eager_cache.cache, qproj_cache.cache)
        for ref, got in (
            (eager_out, qproj_out),
            (eager_cache[0], qproj_cache[0]),
            (eager_cache[1], qproj_cache[1]),
        ):
            worst = max(worst, _max_abs_rel(ref, got))
    assert worst == (0.0, 0.0), f"expected bit-identical, got {worst}"


def test_fused_kda_qproj_declines_unsupported_quantization():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    from mlx_vlm.models.glm5_next.fused_kda import fused_kda_qproj_supported

    config = _config()
    layer = _layer(config)
    fb, gb = layer.forget_gate.f_b_proj, layer.g_b_proj
    D = config.linear_head_dim
    assert fused_kda_qproj_supported(fb, gb, head_dim=D)
    # qmv_quad is only dispatched for head_dim in {64, 128} and bits == 8.
    assert not fused_kda_qproj_supported(fb, gb, head_dim=256)
    assert not fused_kda_qproj_supported(nn.Linear(D, D, bias=False), gb, head_dim=D)
    # An unfolded (dequantized) projection has no scales.
    assert not fused_kda_qproj_supported(fb, nn.Linear(D, D, bias=False), head_dim=D)


def test_fused_kda_declines_ineligible_shapes():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    _require_device(config, "base")
    layer = _layer(config)
    _set_toggle(layer, True)
    assert layer._fused_kda_ready()

    cache = _cache(config)
    ref = mx.zeros((1, 1, config.hidden_size), mx.bfloat16)
    ok = dict(B=1, S=1, mask=None, cache=cache, ref=ref)
    assert layer._fused_kda_eligible(**ok)
    assert not layer._fused_kda_eligible(**{**ok, "B": 2})
    assert not layer._fused_kda_eligible(**{**ok, "S": 8})
    assert not layer._fused_kda_eligible(**{**ok, "mask": mx.array([[True]])})
    assert not layer._fused_kda_eligible(**{**ok, "cache": None})
    assert not layer._fused_kda_eligible(**{**ok, "cache": ArraysCache(size=2)})
    assert not layer._fused_kda_eligible(**{**ok, "cache": _cache(config, batch=2)})
    assert not layer._fused_kda_eligible(
        **{**ok, "ref": mx.zeros((1, 1, config.hidden_size), mx.float32)}
    )


def test_fused_kda_prefill_then_decode_agrees():
    """Prefill (eager either way) followed by a fused decode step."""
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    _require_device(config, "base")
    _require_device(config, "qproj", layer)
    mx.random.seed(99)
    prompt = mx.random.normal((1, 16, config.hidden_size)).astype(mx.bfloat16)
    token = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    mx.eval(prompt, token)

    outs = []
    caches = []
    for on in (False, True):
        cache = ArraysCache(size=2)
        _set_toggle(layer, on)
        layer(prompt, None, cache)
        outs.append(layer(token, None, cache))
        caches.append(cache)
    mx.eval(outs, [c.cache for c in caches])
    assert _max_abs_rel(outs[0], outs[1]) == (0.0, 0.0)
    assert _max_abs_rel(caches[0][1], caches[1][1]) == (0.0, 0.0)


def test_toggle_off_uses_eager_path():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    layer = _layer(config)
    _set_toggle(layer, False)
    assert not layer._fused_kda_ready()
    cache = _cache(config)
    x = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    before = mx.array(cache[1])
    out = layer(x, None, cache)
    mx.eval(out, cache.cache)
    assert not bool(mx.all(before == cache[1]))  # eager path still advanced state


def _bench(n_layers=34, iters=20, warmup=5):  # pragma: no cover - manual bench
    import time

    config = _config()
    layers = [_layer(config, seed=i) for i in range(n_layers)]
    h = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    hq = mx.random.normal(
        (1, 1, config.linear_num_heads * config.linear_head_dim)
    ).astype(mx.bfloat16)
    hd = mx.random.normal((1, 1, config.linear_head_dim)).astype(mx.bfloat16)
    mx.eval(h, hq, hd)
    for layer in layers:
        mx.eval(layer._fused_in_proj(h))

    def timeit(fn):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000.0

    def sweep(fn):
        def run():
            mx.eval([fn(i) for i in range(n_layers)])

        return run

    for tag, on, qp in (
        ("eager", False, False),
        ("fused", True, False),
        ("qproj", True, True),
    ):
        for layer in layers:
            _set_toggle(layer, on, qproj=qp)
        caches = [_cache(config, seed=100 + i) for i in range(n_layers)]
        full = timeit(sweep(lambda i: layers[i](h, None, caches[i])))
        gemv = (
            timeit(sweep(lambda i: layers[i]._fused_in_proj(h)[0]))
            + timeit(sweep(lambda i: layers[i].o_proj(hq)))
            + timeit(sweep(lambda i: layers[i].g_b_proj(hd)))
        )
        print(
            f"{tag:5s} full={full:7.3f} ms  gemv_floor={gemv:7.3f} ms  "
            f"chain={full - gemv:7.3f} ms"
        )


if __name__ == "__main__":  # pragma: no cover
    _bench()
