import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import (
    SamplingConfig,
    _new_modes_keep,
    _PositionedTargetSampler,
    batched_row_sample,
)
from mlx_vlm.sample_utils import (
    apply_min_p,
    apply_p_less,
    apply_top_k,
    apply_top_n_sigma,
    apply_top_p,
    apply_typical_p,
)

NEG_INF = -float("inf")


def _log_normalize(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _ref_keep(masked_logits):
    """Token indices main's apply_* leaves unmasked (a single [V] row)."""
    return {i for i, v in enumerate(masked_logits.tolist()) if v != NEG_INF}


def _ours_keep(work_row, *, top_n_sigma=0.0, p_less=False, typical_p=1.0, temp=1.0):
    """Token indices our _new_modes_keep keeps for a single [V] row.

    The row's dtype is passed through untouched: _new_modes_keep is responsible
    for widening its own statistics, and casting here would hide float16 bugs.
    """
    keep = _new_modes_keep(
        work_row[None],
        mx.array([float(temp)]),
        mx.array([float(top_n_sigma)]),
        mx.array([bool(p_less)]),
        mx.array([float(typical_p)]),
    )
    return {i for i, b in enumerate(keep[0].tolist()) if b}


def test_positioned_sampler_select_restricts_to_kept_configs():
    # select(keep) returns a sampler with exactly the kept per-row configs, in
    # order -- so a speculative batch that drops finished rows keeps the
    # sampler's configs aligned to the shrunk logprobs.
    c0 = SamplingConfig(temperature=0.7, seed=1)
    c1 = SamplingConfig(temperature=1.5, top_k=40, seed=2)
    c2 = SamplingConfig(temperature=0.0, seed=3)
    sub = _PositionedTargetSampler([c0, c1, c2]).select([0, 2])
    assert sub.configs == [c0, c2]
    # sample_target then accepts a 2-row batch (was 3) without the width guard firing
    lp = mx.zeros((2, 5)) - mx.logsumexp(mx.zeros((2, 5)), axis=-1, keepdims=True)
    tok = sub.sample_target(lp, row_ids=[0, 2], positions=[4, 4])
    mx.eval(tok)
    assert tok.shape == (2,)


def _keys(n):
    return mx.stack([mx.random.key(i) for i in range(n)])


def test_config_is_frozen_and_hashable():
    c = SamplingConfig(temperature=0.7, top_p=0.9, top_k=40, min_p=0.0, seed=1)
    with pytest.raises(Exception):
        c.temperature = 0.1  # frozen
    assert (
        len(
            {c, SamplingConfig(temperature=0.7, top_p=0.9, top_k=40, min_p=0.0, seed=1)}
        )
        == 1
    )


def _sampled_set(
    row_logprobs,
    *,
    top_k,
    top_p=1.0,
    min_p=0.0,
    temperature=1.0,
    top_n_sigma=0.0,
    p_less=False,
    typical_p=1.0,
    n=256,
):
    """Distinct tokens sampled for a single [1,V] row across n distinct keys."""
    lp = row_logprobs - mx.logsumexp(row_logprobs, axis=-1, keepdims=True)
    out = set()
    for i in range(n):
        tok = batched_row_sample(
            lp,
            temperature=mx.array([float(temperature)]),
            top_p=mx.array([float(top_p)]),
            top_k=mx.array([int(top_k)], dtype=mx.int32),
            min_p=mx.array([float(min_p)]),
            keys=mx.stack([mx.random.key(i)]),
            top_n_sigma=mx.array([float(top_n_sigma)]),
            p_less=mx.array([bool(p_less)]),
            typical_p=mx.array([float(typical_p)]),
        )
        mx.eval(tok)
        out.add(int(tok[0].item()))
    return out


def test_topk_one_is_deterministic_argmax():
    # top_k=1 => exactly one eligible token => the argmax, for every key.
    lp = mx.array([[1.0, 7.0, 3.0, 0.0, 5.0]])
    assert _sampled_set(lp, top_k=1, temperature=2.0, n=64) == {1}


def test_topk_restricts_to_topk_indices():
    # distinct logits: only the top-3 indices may ever be sampled.
    lp = mx.array([[10.0, 8.0, 6.0, 4.0, 2.0, 0.0]])
    assert _sampled_set(lp, top_k=3, temperature=1.0) <= {0, 1, 2}


def test_topk_exactly_k_rank_cutoff_not_value_threshold_on_ties():
    # [5,3,3,3,1], top_k=2: rank cutoff keeps EXACTLY 2 (rank0 + one tied token).
    # A value-threshold impl (>= k-th value) would wrongly admit all three 3s
    # (indices 1,2,3) plus index 0 => up to 4 distinct. Assert at most 2 distinct
    # and the value-1 token (index 4) is never eligible.
    lp = mx.array([[5.0, 3.0, 3.0, 3.0, 1.0]])
    got = _sampled_set(lp, top_k=2, temperature=1.0)
    assert 4 not in got
    assert len(got) <= 2


def test_topk_matches_scalar_primitive_keep_set_for_uniform_k():
    # The eligible set must equal apply_top_k's kept set (which has exactly k members).
    lp = mx.random.normal((1, 2000))
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    for k in (1, 5, 40):
        ref = apply_top_k(lp, k)
        ref_set = {i for i in range(lp.shape[-1]) if ref[0, i].item() != NEG_INF}
        assert len(ref_set) == k
        assert _sampled_set(lp, top_k=k, temperature=1.0, n=128) <= ref_set


def test_greedy_rows_return_argmax():
    lp = mx.array([[0.0, 5.0, 1.0], [9.0, 0.0, 0.0]])
    got = batched_row_sample(
        lp,
        temperature=mx.zeros(2),
        top_p=mx.ones(2),
        top_k=mx.zeros(2, dtype=mx.int32),
        min_p=mx.zeros(2),
        keys=_keys(2),
    )
    assert got.tolist() == [1, 0]  # argmax per row, no NaN from temp==0


def test_mixed_greedy_and_sampled_batch():
    lp = mx.array([[0.0, 9.0, 0.0], [1.0, 1.0, 1.0]])
    got = batched_row_sample(
        lp,
        temperature=mx.array([0.0, 1.0]),
        top_p=mx.ones(2),
        top_k=mx.zeros(2, dtype=mx.int32),
        min_p=mx.zeros(2),
        keys=_keys(2),
    )
    assert int(got[0].item()) == 1  # greedy row -> argmax
    assert 0 <= int(got[1].item()) < 3  # sampled row -> valid draw


def test_new_modes_match_reference_keep_set():
    # Each new mode, single active row, keeps EXACTLY the tokens main's
    # sample_utils reference keeps -- pins the per-row masks to the accepted
    # algorithm, not just "it runs".
    mx.random.seed(0)
    for trial in range(8):
        L = _log_normalize(mx.random.normal((64,)) * (0.5 + trial))
        for n in (0.5, 1.0, 2.0):
            assert _ours_keep(L, top_n_sigma=n) == _ref_keep(apply_top_n_sigma(L, n))
        assert _ours_keep(L, p_less=True, temp=1.0) == _ref_keep(apply_p_less(L, 1.0))
        for tp in (0.2, 0.6, 0.95):
            assert _ours_keep(L, typical_p=tp) == _ref_keep(apply_typical_p(L, tp))


def test_new_modes_disabled_keep_all():
    L = _log_normalize(mx.random.normal((32,)))
    assert _ours_keep(L) == set(range(32))  # all defaults -> identity mask


def test_new_mode_sampler_restricts_to_reference():
    # A sampler row with a mode active never draws a token the reference filters.
    mx.random.seed(1)
    L = _log_normalize(mx.random.normal((1, 40)))
    row = L[0]
    assert _sampled_set(L, top_k=0, top_n_sigma=1.0, n=200) <= _ref_keep(
        apply_top_n_sigma(row, 1.0)
    )
    assert _sampled_set(L, top_k=0, typical_p=0.5, n=200) <= _ref_keep(
        apply_typical_p(row, 0.5)
    )
    assert _sampled_set(L, top_k=0, p_less=True, n=200) <= _ref_keep(
        apply_p_less(row, 1.0)
    )


def test_new_modes_per_row_activation():
    # Row 0 uses typical_p; row 1 disables everything. Each row is filtered on
    # its own params -- row 1 keeps all, row 0 matches the reference.
    mx.random.seed(2)
    work = _log_normalize(mx.random.normal((2, 24))).astype(mx.float32)
    keep = _new_modes_keep(
        work,
        mx.array([1.0, 1.0]),
        mx.array([0.0, 0.0]),
        mx.array([False, False]),
        mx.array([0.5, 1.0]),  # row0 typical_p active, row1 disabled
    )
    row0 = {i for i, b in enumerate(keep[0].tolist()) if b}
    row1 = {i for i, b in enumerate(keep[1].tolist()) if b}
    assert row0 == _ref_keep(apply_typical_p(work[0], 0.5))
    assert row1 == set(range(24))


def test_new_modes_grouping_invariance():
    # A row's keep set is identical alone vs co-batched with a row using a
    # different mode (per-row isolation, same property we hold for top_p/min_p).
    mx.random.seed(3)
    a = _log_normalize(mx.random.normal((48,))).astype(mx.float32)
    b = _log_normalize(mx.random.normal((48,))).astype(mx.float32)
    alone = _ours_keep(a, top_n_sigma=1.5)
    together = _new_modes_keep(
        mx.stack([a, b]),
        mx.array([1.0, 1.0]),
        mx.array([1.5, 0.0]),
        mx.array([False, True]),  # b uses a different mode (p_less)
        mx.array([1.0, 1.0]),
    )
    together_a = {i for i, x in enumerate(together[0].tolist()) if x}
    assert together_a == alone


def test_top_n_sigma_matches_reference_in_float16():
    # mx.std over a float16 log-prob vector overflows to inf for any real vocab
    # (the sum is ~V*log(V)), which would make the threshold -inf and silently
    # disable the mode. The statistics must be computed in float32.
    for V in (8192, 32000):
        mx.random.seed(4)
        lp = _log_normalize(mx.random.normal((V,)))
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            row = lp.astype(dtype)
            assert _ours_keep(row, top_n_sigma=1.0) == _ref_keep(
                apply_top_n_sigma(row, 1.0)
            ), f"V={V} dtype={dtype}"


def test_modes_tolerate_neg_inf_logprobs_without_collapsing_to_greedy():
    # Grammar/structured-output masking (and logit_bias=-inf) puts -inf in the
    # logprobs, which makes mx.std NaN. A `x >= threshold` keep-test drops every
    # token under NaN and pins the row to argmax; upstream's `x < threshold`
    # drop-test keeps them, so sampling must continue over the allowed tokens.
    V = 512
    allowed = mx.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    lg = mx.concatenate([allowed, mx.full((V - allowed.size,), -float("inf"))])
    lp = _log_normalize(lg)[None]
    finite = {i for i, v in enumerate(lp[0].tolist()) if v != NEG_INF}

    drawn = set()
    for i in range(128):
        tok = batched_row_sample(
            lp,
            temperature=mx.array([1.0]),
            top_p=mx.array([1.0]),
            top_k=mx.array([0], dtype=mx.int32),
            min_p=mx.array([0.0]),
            keys=mx.stack([mx.random.key(i)]),
            top_n_sigma=mx.array([2.0]),
            p_less=mx.array([False]),
            typical_p=mx.array([1.0]),
        )
        mx.eval(tok)
        drawn.add(int(tok[0].item()))
    assert len(drawn) > 1, "row collapsed to greedy under NaN statistics"
    assert drawn <= finite, "sampled a token the grammar masked out"


def test_typical_p_with_top_k_still_samples():
    # The modes are chained BEFORE the rank filters. If they were ANDed against a
    # top_k window computed on the full distribution, typical_p could reject the
    # entire window, empty the row and pin it to argmax -- a silent collapse to
    # greedy for a very common combination (Qwen ships top_k=20 by default).
    mx.random.seed(5)
    lp = _log_normalize(mx.random.normal((1, 4096)))
    drawn = set()
    for i in range(128):
        tok = batched_row_sample(
            lp,
            temperature=mx.array([1.0]),
            top_p=mx.array([1.0]),
            top_k=mx.array([20], dtype=mx.int32),
            min_p=mx.array([0.0]),
            keys=mx.stack([mx.random.key(i)]),
            top_n_sigma=mx.array([0.0]),
            p_less=mx.array([False]),
            typical_p=mx.array([0.5]),
        )
        mx.eval(tok)
        drawn.add(int(tok[0].item()))
    assert len(drawn) > 1, "typical_p + top_k collapsed to greedy"
    # chained semantics: the candidate set is top_k OF THE TYPICAL_P SURVIVORS
    ref = apply_top_k(apply_typical_p(lp, 0.5), 20)
    assert drawn <= {i for i in range(lp.shape[-1]) if ref[0, i].item() != NEG_INF}


def test_base_masks_unchanged_by_the_new_modes_plumbing():
    # With no mode active the sampler must still match the reference filters
    # exactly -- the chaining/normalization changes must not move the base path.
    mx.random.seed(6)
    lp = _log_normalize(mx.random.normal((1, 4096)))
    for top_p in (0.9, 0.5):
        ref = apply_top_p(lp, top_p)
        allowed = {i for i in range(lp.shape[-1]) if ref[0, i].item() != NEG_INF}
        assert _sampled_set(lp, top_k=0, top_p=top_p, n=128) <= allowed
    for min_p in (0.05, 0.2):
        ref = apply_min_p(lp, min_p)
        allowed = {i for i in range(lp.shape[-1]) if ref[0, i].item() != NEG_INF}
        assert _sampled_set(lp, top_k=0, min_p=min_p, n=128) <= allowed


def test_out_of_range_typical_p_disables_instead_of_filtering_everything():
    # make_sampler gates on 0 < typical_p < 1; a 0 or negative value must be a
    # no-op, not a mask that rejects every token (which would force greedy).
    lp = _log_normalize(mx.random.normal((64,)))
    for tp in (0.0, -0.5, 1.0, 2.0):
        assert _ours_keep(lp, typical_p=tp) == set(range(64)), f"typical_p={tp}"


def test_sampler_skips_mode_arrays_when_no_row_enables_them():
    # The extra masks cost an argsort + cumsum per decode step; when every row
    # leaves the modes off they are all-True, so they must not be built at all.
    off = _PositionedTargetSampler([SamplingConfig(temperature=0.7, top_p=0.9)])
    assert off._arrays()[4:] == (None, None, None)
    for cfg in (
        SamplingConfig(temperature=0.7, top_n_sigma=1.0),
        SamplingConfig(temperature=0.7, p_less=True),
        SamplingConfig(temperature=0.7, typical_p=0.5),
    ):
        assert all(a is not None for a in _PositionedTargetSampler([cfg])._arrays()[4:])


def test_non_bool_p_less_is_coerced_not_raised():
    # p_less arrives from a loosely-validated request field; a truthy non-bool
    # must not raise inside the GPU thread and tear down the whole batch.
    sampler = _PositionedTargetSampler([SamplingConfig(temperature=1.0, p_less="yes")])
    lp = _log_normalize(mx.random.normal((1, 32)))
    tok = sampler(lp)
    mx.eval(tok)
    assert 0 <= int(tok[0].item()) < 32


def test_new_modes_never_empty_no_nan():
    # Aggressive combined base+new masks must still yield a valid token per row
    # (the empty-row safety net), never a NaN collapse.
    got = batched_row_sample(
        _log_normalize(mx.random.normal((3, 50))),
        temperature=mx.ones(3),
        top_p=mx.array([0.05, 1.0, 0.1]),
        top_k=mx.array([1, 0, 2], dtype=mx.int32),
        min_p=mx.array([0.5, 0.0, 0.2]),
        keys=_keys(3),
        top_n_sigma=mx.array([0.3, 0.0, 0.5]),
        p_less=mx.array([True, False, True]),
        typical_p=mx.array([0.3, 1.0, 0.5]),
    )
    mx.eval(got)
    assert all(0 <= int(t) < 50 for t in got.tolist())


# --- 3-D verify logits ([B, T, V]) -----------------------------------------
#
# dflash/eagle3 verify loops call ``sampler(verify_out.logits)`` with the whole
# verify block at once and concatenate/index the result along axis 1, so the
# sampler contract is [B, T, V] -> [B, T]. The server hands those loops a
# _PositionedTargetSampler as their only sampler, so a 2-D-only sampler fails
# every request on a dflash/eagle3 drafter.


def _pinned_and_hot(seed_pinned=11, seed_hot=12):
    """Row 0 pinned to its argmax (top_k=1), row 1 free to roam."""
    return _PositionedTargetSampler(
        [
            SamplingConfig(temperature=1.0, top_k=1, seed=seed_pinned),
            SamplingConfig(temperature=2.0, seed=seed_hot),
        ]
    )


def test_sampler_accepts_3d_verify_logits():
    B, T, V = 2, 3, 16
    lp = _log_normalize(mx.random.normal((B, T, V), key=mx.random.key(0)))
    tokens = _pinned_and_hot()(lp)
    mx.eval(tokens)
    assert tokens.shape == (B, T)
    assert tokens.dtype in (mx.int32, mx.uint32, mx.int64)


def test_3d_rows_use_their_own_config():
    # Expanding the per-row configs in the wrong order (tile instead of repeat)
    # gives row 0's flattened positions row 1's config, so the pinned row stops
    # tracking its own argmax.
    B, T, V = 2, 8, 24
    lp = _log_normalize(mx.random.normal((B, T, V), key=mx.random.key(3)))
    tokens = _pinned_and_hot()(lp)
    argmax = mx.argmax(lp, axis=-1)
    mx.eval(tokens, argmax)
    # row 0 is top_k=1 -> its argmax at every position
    assert tokens[0].tolist() == argmax[0].tolist()
    # row 1 is hot: it must not be pinned to the argmax at every position
    assert tokens[1].tolist() != argmax[1].tolist()


def test_3d_positions_advance_within_the_block():
    # A single frozen key for the whole block would draw the same quantile at
    # every position; with a near-uniform row that shows up as one repeated
    # token across T.
    B, T, V = 1, 12, 40
    lp = _log_normalize(mx.zeros((B, T, V)))
    tokens = _PositionedTargetSampler([SamplingConfig(temperature=1.0, seed=5)])(lp)
    mx.eval(tokens)
    assert len(set(tokens[0].tolist())) > 1


def test_3d_matches_2d_per_slice():
    # Each [b, t] slice must be sampled from that slice alone -- no leakage
    # from neighbouring positions. Pinning every row to top_k=1 makes the
    # expected answer exact, so a transposed flatten/reshape fails loudly.
    B, T, V = 3, 4, 20
    lp = _log_normalize(mx.random.normal((B, T, V), key=mx.random.key(7)))
    sampler = _PositionedTargetSampler(
        [SamplingConfig(temperature=1.0, top_k=1, seed=s) for s in (1, 2, 3)]
    )
    tokens = sampler(lp)
    expected = mx.argmax(lp, axis=-1)
    mx.eval(tokens, expected)
    assert tokens.tolist() == expected.tolist()


def test_sampler_rejects_unsupported_rank():
    sampler = _PositionedTargetSampler([SamplingConfig(temperature=1.0, seed=1)])
    with pytest.raises(ValueError, match="rank"):
        sampler(mx.zeros((5,)))
    with pytest.raises(ValueError, match="rank"):
        sampler(mx.zeros((1, 1, 2, 5)))


def test_3d_width_mismatch_still_raises():
    sampler = _PositionedTargetSampler(
        [SamplingConfig(temperature=1.0, seed=1), SamplingConfig(seed=2)]
    )
    with pytest.raises(ValueError, match="configs length"):
        sampler(mx.zeros((3, 2, 5)))


# --- narrower batches on a uniform sampler ---------------------------------
#
# The dflash/eagle3 round loops build ONE sampler for the whole batch and then
# call it with fewer rows: draft_block drafts row by row ([1, T, V] per row,
# dflash.py:255-267), and continuous batching drops finished sequences from the
# verify batch without rebuilding the sampler (dflash.py:328-338). The server
# guarantees those configs are uniform (_serialize_speculative_batch_by_config
# returns [target_config] * len(kept)), so one config broadcasts to any width.


def test_uniform_configs_broadcast_to_narrower_batch():
    V = 12
    sampler = _PositionedTargetSampler([SamplingConfig(temperature=1.0, seed=4)] * 3)
    lp2 = _log_normalize(mx.random.normal((1, V), key=mx.random.key(1)))
    lp3 = _log_normalize(mx.random.normal((2, 5, V), key=mx.random.key(2)))
    tok2, tok3 = sampler(lp2), sampler(lp3)
    mx.eval(tok2, tok3)
    assert tok2.shape == (1,)
    assert tok3.shape == (2, 5)


def test_broadcast_uses_the_shared_config_not_a_default():
    # A broadcast row must carry the batch's config, not SamplingConfig() --
    # whose temperature=0.0 would silently turn the row greedy.
    V, T = 32, 16
    lp = _log_normalize(mx.zeros((1, T, V)))
    tokens = _PositionedTargetSampler([SamplingConfig(temperature=1.0, seed=4)] * 3)(lp)
    mx.eval(tokens)
    assert len(set(tokens[0].tolist())) > 1


def test_heterogeneous_configs_still_require_matching_width():
    sampler = _PositionedTargetSampler(
        [
            SamplingConfig(temperature=1.0, seed=1),
            SamplingConfig(temperature=1.5, seed=2),
            SamplingConfig(temperature=0.5, seed=3),
        ]
    )
    with pytest.raises(ValueError, match="configs length"):
        sampler(mx.zeros((1, 6)))
    with pytest.raises(ValueError, match="configs length"):
        sampler(mx.zeros((2, 4, 6)))
