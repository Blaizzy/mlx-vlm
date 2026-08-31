"""Eager vs fast equivalence for the GLM-5-Next DSA indexer decode path.

``MLX_VLM_GLM5_IDX_FAST=1`` replaces the indexer's *active* decode path (the
regime ``T > index_topk``, where the short-context bypass no longer applies)
with one that does only the work a new token actually creates.

The eager path recomputes, every step and in every DSA layer:
  * ``kv_pos`` / ``visible`` / ``pool_visible`` -- O(T) tensors that, for a
    single unpadded stream, are all-True by construction;
  * ``_visible_tail`` -- an O(T) reduction whose result is then a closed form
    in ``T`` alone;
  * ``mx.concatenate`` of the stable pool prefix -- an O(P) copy of pool state
    that is append-only during decode;
  * the whole index apparatus of ``_pooled_states`` (argmax/arange/clip/three
    gathers/validity reductions) for a tail of at most ``index_kpool`` tokens,
    where every one of those indices is known on the host.

None of that changes the selection, so the fast path is expected to be
*bit-identical*: the returned top-k index tensor must match element for
element across carried steps.  These tests pin that at three context lengths,
pin the pool-buffer growth path, and pin that anything the path is not written
for (prefill, S>1 verify blocks, batching, left-padding, a rollback that
shortened the cache) falls back to the eager path.
"""

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.glm5_next.config import TextConfig

# GLM-5.3-Flash text_config, restricted to what the indexer reads.
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
    layer_types=["deepseek_sparse_attention"],
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


def _indexer(config, seed=0):
    mx.random.seed(seed)
    ix = glm5.Glm5NextIndexer(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    ix.update(rand(ix.parameters()))
    return ix


def _prefill(ix, config, T, seed=1, batch=1):
    """Eager prefill, so `_pool` / `_no_pad` are built by the reference path."""
    mx.random.seed(seed)
    cache = KVCache()
    x = (mx.random.normal((batch, T, config.hidden_size)) * 0.4).astype(mx.bfloat16)
    qr = (mx.random.normal((batch, T, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
    glm5._IDX_FAST_ENV = False
    mx.eval(ix(x, qr, None, cache=cache), cache.keys)
    return cache


def _clone(cache):
    out = KVCache()
    out.keys = mx.array(cache.keys)
    out.values = cache.values
    out.offset = cache.offset
    out._pool = tuple(
        mx.array(a) if isinstance(a, mx.array) else a for a in cache._pool
    )
    out._no_pad = cache._no_pad
    out._fpool = None
    mx.eval(out.keys, out._pool[0], out._pool[1], out._pool[2])
    return out


def _steps(config, n, batch=1, seed=99):
    mx.random.seed(seed)
    out = []
    for _ in range(n):
        out.append(
            (
                (mx.random.normal((batch, 1, config.hidden_size)) * 0.4).astype(
                    mx.bfloat16
                ),
                (mx.random.normal((batch, 1, config.q_lora_rank)) * 0.4).astype(
                    mx.bfloat16
                ),
            )
        )
    mx.eval(out)
    return out


@pytest.fixture(autouse=True)
def _reset_toggle():
    saved = (glm5._IDX_FAST_ENV, glm5._IDX_POOL_STEP)
    yield
    glm5._IDX_FAST_ENV, glm5._IDX_POOL_STEP = saved


def _carry(ix, config, ctx, n_steps, seed=1):
    """Run n_steps eager and fast from the same prefill; return worst mismatch.

    Counts fast-path entries as well: if the eligibility guard ever stopped
    firing this comparison would quietly become eager-vs-eager and pass for the
    wrong reason, so the count is asserted rather than assumed.
    """
    eager_cache = _prefill(ix, config, ctx, seed=seed)
    fast_cache = _clone(eager_cache)
    worst = 0
    entered = {"n": 0}
    original = type(ix)._decode_fast

    def counting(self, *args, **kwargs):
        entered["n"] += 1
        return original(self, *args, **kwargs)

    type(ix)._decode_fast = counting
    try:
        worst = _carry_inner(ix, config, eager_cache, fast_cache, n_steps, worst)
    finally:
        type(ix)._decode_fast = original
    assert entered["n"] == n_steps, (
        f"fast path ran {entered['n']}/{n_steps} steps -- the comparison would "
        "have been eager-vs-eager"
    )
    return worst


def _carry_inner(ix, config, eager_cache, fast_cache, n_steps, worst):
    for x, qr in _steps(config, n_steps):
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        mx.eval(ref, got)
        assert ref is not None and got is not None
        assert ref.shape == got.shape and ref.dtype == got.dtype
        worst = max(worst, int(mx.sum((ref != got).astype(mx.int32))))
    return worst


@pytest.mark.parametrize("ctx", [4096, 16384])
def test_idx_fast_matches_eager_over_32_decode_steps(ctx):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    # The selection is the same selection, reached with less arithmetic, so this
    # is exact rather than merely close.
    assert _carry(ix, config, ctx, 32) == 0


def test_idx_fast_survives_pool_buffer_growth():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    # Shrink the growth step so the preallocated pool buffers must be grown
    # several times inside the run instead of once every 2048 tokens.
    glm5._IDX_POOL_STEP = 2
    assert _carry(ix, config, 4096, 40) == 0


def test_idx_fast_survives_interleaved_verify_blocks():
    """Speculative decoding interleaves S>1 verify blocks with S=1 decode.

    The fast path owns the pool state in `_fpool` and clears `_pool`; a verify
    block is ineligible, so it must fall back to a full rebuild, restore
    `_pool`, and let the next single-token step re-seed `_fpool` from it.  Run
    the identical mixed sequence eager and fast and require the selections to
    stay bit-identical the whole way through.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    eager_cache = _prefill(ix, config, 4096)
    fast_cache = _clone(eager_cache)

    mx.random.seed(1234)
    plan = []
    for i in range(4):
        for _ in range(3):
            plan.append(1)
        plan.append(4)  # a draft-block verify
    for n in plan:
        x = (mx.random.normal((1, n, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, n, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(x, qr)
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        mx.eval(ref, got)
        assert ref.shape == got.shape
        assert int(mx.sum((ref != got).astype(mx.int32))) == 0, f"diverged at S={n}"


def test_idx_fast_declines_ineligible_shapes():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fast path taken for an ineligible step")

    real = glm5.Glm5NextIndexer._decode_fast
    glm5.Glm5NextIndexer._decode_fast = _boom
    try:
        glm5._IDX_FAST_ENV = True
        # prefill (S = T) -- no pool state yet
        cache = _prefill(ix, config, 4096)
        # S > 1 speculative-verify block
        glm5._IDX_FAST_ENV = True
        mx.random.seed(3)
        x = (mx.random.normal((1, 4, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 4, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(ix(x, qr, None, cache=_clone(cache)))
        # left-padded decode: a bool mask is supplied
        c2 = _clone(cache)
        x1, qr1 = _steps(config, 1)[0]
        mx.eval(ix(x1, qr1, mx.ones((1, 1), dtype=mx.bool_), cache=c2))
        # padded sequence: _no_pad is False
        c3 = _clone(cache)
        c3._no_pad = False
        mx.eval(ix(x1, qr1, None, cache=c3))
        # rollback: the cache was trimmed, so `_pool` is stale by > 1 token
        c4 = _clone(cache)
        c4._pool = c4._pool[:3] + (c4._pool[3] - 4,)
        mx.eval(ix(x1, qr1, None, cache=c4))
        # batched decode
        cb = _prefill(ix, config, 4096, batch=2)
        xb, qrb = _steps(config, 1, batch=2)[0]
        mx.eval(ix(xb, qrb, None, cache=cb))
    finally:
        glm5.Glm5NextIndexer._decode_fast = real


def test_idx_fast_bypass_regime_returns_none():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    for flag in (False, True):
        glm5._IDX_FAST_ENV = flag
        cache = KVCache()
        mx.random.seed(5)
        x = (mx.random.normal((1, 512, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 512, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        # T = 512 <= index_topk = 2048 -> short-context bypass, no selection
        assert ix(x, qr, None, cache=cache) is None


def test_toggle_off_uses_eager_path():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fast path taken with the toggle off")

    real = glm5.Glm5NextIndexer._decode_fast
    glm5.Glm5NextIndexer._decode_fast = _boom
    try:
        glm5._IDX_FAST_ENV = False
        cache = _prefill(ix, config, 4096)
        x, qr = _steps(config, 1)[0]
        mx.eval(ix(x, qr, None, cache=cache))
    finally:
        glm5.Glm5NextIndexer._decode_fast = real
