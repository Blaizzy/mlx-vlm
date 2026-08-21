"""The parity contract between the row sampler and ``sample_utils.make_sampler``.

``batched_row_sample`` re-expresses make_sampler's chain as a single sorted-space
pass so a batch can carry per-row parameters. That re-expression is only
legitimate while it keeps *exactly* the candidate set the chain keeps: a server
request must not sample differently because it was routed to the continuous
batch instead of to make_sampler.

The contract, asserted below:

    for every parameter combination, input dtype and vocab size, the tokens
    batched_row_sample leaves alive must equal the tokens make_sampler's
    composed chain leaves alive -- unless the difference is named in
    KNOWN_DIVERGENCES and justified there.

This file exists because the previous parity test could not see a real
divergence: it pinned temperature to 1.0 (the one temperature at which the two
formulations agree) and asserted a subset rather than equality. Assert equality,
and sweep temperature.
"""

import math
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate.ar import (
    SamplingConfig,
    _new_modes_keep,
    _PositionedTargetSampler,
)
from mlx_vlm.sample_utils import (
    apply_min_p,
    apply_p_less,
    apply_top_k,
    apply_top_n_sigma,
    apply_top_p,
    apply_typical_p,
    make_sampler,
)

NEG_INF = -float("inf")
_REAL_CATEGORICAL = mx.random.categorical
_REAL_ARGSORT = mx.argsort


# --- KNOWN_DIVERGENCES -----------------------------------------------------
#
# Every entry is a deliberate, reviewed difference from make_sampler. Adding an
# entry is a design decision, not a way to quiet a failure.
#
# Established by three independent sweeps that wrote their own oracles rather
# than reusing this file's helpers: 56,736 rank-filter cases, 63,012 mode-chain
# cases, 5,446 pathological-input cases. Everything they found is below.
#
# 1. EMPTY ROWS. A rank filter chained after a mode can reject every token
#    upstream, and make_sampler then draws from an all -inf row -- an arbitrary
#    token. This keeps the top survivor instead.
#
# 2. INPUT NORMALIZATION, and what it does to typical_p. batched_row_sample
#    log-normalizes its own input because the speculative loops hand it raw
#    lm_head logits (dflash.py, eagle3.py, mtp.py) while the AR path hands it
#    logprobs, and exp() of an unnormalized row is not a probability vector --
#    without it top_p silently stops filtering on every speculative request.
#    make_sampler does not normalize, so the two disagree on any row whose
#    exp-mass is not 1. Two of the three modes are unaffected (top_n_sigma is
#    shift-invariant, p_less softmaxes), but typical_p is not: on a row that
#    already carries -inf entries -- a grammar mask, or logit_bias=-inf, or the
#    output of a previous mode -- upstream measures typicality against the
#    residual mass and this measures it against the renormalized distribution.
#    Measured: 8 tokens kept upstream vs 1 here on a 64-token row with 56
#    entries masked. Ours is the more defensible reading of "keep half the
#    probability mass", but it IS different, it is reachable through logit_bias,
#    and the alternative (normalize only for the rank filters, leave the modes
#    on the raw input) is an open design question, not a settled one.
#
# 3. TIE ORDERING AT ANY CUTOFF -- INCLUDING float32. Upstream sorts ascending
#    and this sorts descending, so equal-valued tokens straddling a threshold
#    fall on opposite sides. Same cardinality, same probability mass, different
#    members: log_softmax([3,3,3,3,0,-1,-2,-3]) at top_p=0.5 keeps {0,1,2} here
#    and {1,2,3} upstream. An earlier version of this note claimed float32 was
#    safe and that "the strict sweeps run in float32" made it a non-issue. That
#    was wrong. Rare on random rows (0 of 600) and constant on quantized or
#    rounded logits. Jittering the ties away restores exact parity, which is how
#    the mechanism was confirmed.
#
# 4. min_p EXACTLY ON THE BOUNDARY. Upstream's _apply_min_p is @mx.compile'd,
#    and the compiled kernel drops a token whose logprob equals
#    max + log(min_p) to the bit, while the same formula run eagerly -- or under
#    mx.disable_compile() -- keeps it. This matches upstream's uncompiled
#    semantics, so the divergence is against an MLX compile artifact rather than
#    against make_sampler's intent. Needs an exact-boundary coincidence: 0 of
#    1,800 random rows, but a Zipf row hits it every time at min_p=0.01.
#
# 5. REDUCED PRECISION beyond tie ordering. In float16/bfloat16, p_less and
#    typical_p differ systematically, not just at ties, because the statistics
#    they compute (collision probability, entropy) are accumulated at different
#    widths on each side. The strict sweeps therefore run in float32.
#
# 6. top_k OUTSIDE ITS DOMAIN. make_sampler raises for top_k >= vocab_size
#    (apply_top_k validates 0 < top_k < vocab_size); this treats any top_k at or
#    above the vocab as "keep everything", and clamps into int32 so an absurd
#    value cannot crash the GPU worker and take every co-batched request with
#    it. Deliberate, and the better behaviour for a server, but a difference.
#
# 7. TEMPERATURE EDGES. Greedy is decided by temperature < _GREEDY_EPS (1e-5)
#    rather than == 0, so 0 < temperature < 1e-5 is greedy here and sampled
#    upstream (token-equivalent in practice -- the argmax survives every filter
#    -- except where the upstream chain empties the row, which is entry 1).
#    Negative temperature is greedy here; upstream inverts the distribution.
#    Both are reachable now that the request schemas carry no bounds.


def _log_normalize(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _keep_of(masked_row):
    return {i for i, v in enumerate(masked_row.tolist()) if v != NEG_INF}


def _reference_keep(lp_row, cfg):
    """Exactly the chain make_sampler composes, in its order, with its gates."""
    x = lp_row
    if cfg.top_n_sigma > 0.0:
        x = apply_top_n_sigma(x, cfg.top_n_sigma)
    if cfg.p_less:
        x = apply_p_less(x, cfg.temperature)
    if 0.0 < cfg.typical_p < 1.0:
        x = apply_typical_p(x, cfg.typical_p)
    if 0.0 < cfg.top_p < 1.0:
        x = apply_top_p(x, cfg.top_p)
    if cfg.min_p != 0.0:
        x = apply_min_p(x, cfg.min_p)
    if cfg.top_k > 0:
        x = apply_top_k(x, cfg.top_k)
    mx.eval(x)
    return _keep_of(x)


def _sampler_keep(lp_row, cfg):
    """Tokens the real row sampler leaves alive, in ORIGINAL index space.

    Captured from the array handed to mx.random.categorical -- one forward pass,
    no sampling loop, and it observes the production code rather than a
    reimplementation of it.
    """
    captured = {}
    orders = []

    def cat_spy(logits, *args, **kwargs):
        captured["masked"] = logits
        return _REAL_CATEGORICAL(logits, *args, **kwargs)

    def sort_spy(*args, **kwargs):
        out = _REAL_ARGSORT(*args, **kwargs)
        orders.append(out)
        return out

    sampler = _PositionedTargetSampler([cfg])
    with (
        patch.object(mx.random, "categorical", cat_spy),
        patch.object(mx, "argsort", sort_spy),
    ):
        mx.eval(sampler(lp_row[None]))
    assert "masked" in captured, "sampler never reached the categorical draw"

    # The mask is in sorted space, and when a mode is active the chain sorts the
    # MODE-MASKED array -- so the permutation has to come from the run itself,
    # not from re-sorting the input. The last argsort before the draw is it.
    masked = captured["masked"][0]
    order = orders[-1][0]
    mx.eval(masked, order)
    order_list = order.tolist()
    return {order_list[i] for i, v in enumerate(masked.tolist()) if v != NEG_INF}


def _row(vocab, *, key, scale=2.0, dtype=mx.float32):
    return _log_normalize(
        mx.random.normal((vocab,), key=mx.random.key(key)) * scale
    ).astype(dtype)


TEMPERATURES = [0.3, 0.7, 1.0, 1.5, 2.0]
RANK_FILTERS = [
    dict(),
    dict(top_p=0.9),
    dict(top_p=0.5),
    dict(top_p=1.0),  # the server default -- must disable, not truncate
    dict(top_p=0.0),  # must disable, as make_sampler does -- not collapse
    dict(min_p=0.05),
    dict(min_p=0.3),
    dict(top_k=1),
    dict(top_k=40),
    dict(top_p=0.9, top_k=40),
    dict(top_p=0.9, min_p=0.05),
    dict(top_p=0.8, top_k=50, min_p=0.02),
]


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("filters", RANK_FILTERS, ids=lambda f: repr(sorted(f.items())))
def test_rank_filters_match_make_sampler_at_every_temperature(temperature, filters):
    """Contract 1: no mode active -> exact candidate-set equality.

    Temperature must not move the candidate set at all: make_sampler filters the
    unscaled logprobs and applies temperature only in categorical_sampling.
    """
    cfg = SamplingConfig(temperature=temperature, seed=1, **filters)
    for key in (1, 2, 3):
        lp = _row(512, key=key)
        assert _sampler_keep(lp, cfg) == _reference_keep(
            lp, cfg
        ), f"temperature={temperature} filters={filters} key={key}"


@pytest.mark.parametrize("vocab", [256, 4096, 32768])
@pytest.mark.parametrize("scale", [1.0, 4.0, 8.0])
def test_disabled_top_p_keeps_the_whole_vocabulary(vocab, scale):
    """top_p=1.0 is the server default and must be a no-op.

    Cumulative sums in float32 drift above 1.0 on peaked distributions, so a
    `<= top_p` cutoff silently truncates the tail exactly where the parameter is
    supposed to be inert. make_sampler never has this failure mode because it
    skips apply_top_p entirely unless 0 < top_p < 1.
    """
    lp = _log_normalize(mx.random.normal((vocab,), key=mx.random.key(7)) * scale)
    for top_p in (1.0, 0.0):
        cfg = SamplingConfig(temperature=1.0, top_p=top_p, seed=1)
        assert _sampler_keep(lp, cfg) == set(
            range(vocab)
        ), f"top_p={top_p} vocab={vocab}"


@pytest.mark.parametrize("mode", ["top_n_sigma", "p_less", "typical_p"])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_each_mode_alone_matches_its_reference(mode, temperature):
    """Contract 2: one mode, no rank filter -> exact candidate-set equality."""
    kw = (
        {"top_n_sigma": 1.0}
        if mode == "top_n_sigma"
        else ({"p_less": True} if mode == "p_less" else {"typical_p": 0.5})
    )
    cfg = SamplingConfig(temperature=temperature, seed=1, **kw)
    for key in (11, 12):
        lp = _row(512, key=key)
        assert _sampler_keep(lp, cfg) == _reference_keep(
            lp, cfg
        ), f"{mode} T={temperature}"


MODE_RANK_COMBOS = [
    dict(top_n_sigma=1.0, top_p=0.9),
    dict(top_n_sigma=1.5, min_p=0.05),
    dict(p_less=True, top_p=0.9),
    dict(p_less=True, top_k=40),
    dict(typical_p=0.5, top_p=0.9),
    dict(typical_p=0.5, min_p=0.05),
    dict(typical_p=0.5, top_p=0.8, top_k=30, min_p=0.02),
]


@pytest.mark.parametrize("temperature", [0.3, 0.7, 1.0, 1.5, 2.0])
@pytest.mark.parametrize(
    "combo", MODE_RANK_COMBOS, ids=lambda c: repr(sorted(c.items()))
)
def test_one_mode_plus_rank_filters_matches_make_sampler(temperature, combo):
    """A single mode composed with rank filters must match token for token.

    The only licensed escape is KNOWN_DIVERGENCE 1: where make_sampler rejects
    every token, this keeps the top survivor rather than drawing from an all
    -inf row.
    """
    cfg = SamplingConfig(temperature=temperature, seed=1, **combo)
    for key in (11, 12, 13):
        lp = _row(512, key=key)
        ours = _sampler_keep(lp, cfg)
        ref = _reference_keep(lp, cfg)
        if not ref:
            assert len(ours) == 1, "empty upstream row must fall back to one token"
            continue
        assert ours == ref, f"T={temperature} {combo} key={key}"


TWO_MODE_COMBOS = [
    dict(top_n_sigma=1.0, p_less=True),
    dict(top_n_sigma=1.5, p_less=True),
    dict(top_n_sigma=1.0, p_less=True, top_p=0.9),
    dict(top_n_sigma=1.0, p_less=True, top_k=40),
]


@pytest.mark.parametrize("temperature", [0.3, 0.7, 1.0, 1.5, 2.0])
@pytest.mark.parametrize(
    "combo", TWO_MODE_COMBOS, ids=lambda c: repr(sorted(c.items()))
)
def test_two_modes_without_typical_p_match_make_sampler(temperature, combo):
    """Modes chain sequentially, so a second mode filters what the first left.

    p_less's collision probability is a property of the distribution it is
    actually sampling from; computing it on the unfiltered row is a different
    filter, not a rounding difference. Nothing in this chain is NaN-poisoned
    upstream, so nothing licenses a divergence here.
    """
    cfg = SamplingConfig(temperature=temperature, seed=1, **combo)
    for key in (11, 12, 13, 14, 15):
        lp = _row(512, key=key)
        ref = _reference_keep(lp, cfg)
        if not ref:
            continue  # KNOWN_DIVERGENCE 1, covered elsewhere
        assert _sampler_keep(lp, cfg) == ref, f"T={temperature} {combo} key={key}"


def test_typical_p_after_another_mode_stays_well_defined():
    """KNOWN_DIVERGENCE 2, now narrowed to exactly one chain.

    typical_p is the only mode whose upstream form breaks when chained: its
    entropy sum takes 0 * -inf over the tokens a previous mode masked. This
    drops those terms instead of inheriting the NaN. The guard asserts upstream
    really is still poisoned, so the divergence gets revisited rather than
    inherited if that is ever fixed upstream.
    """
    lp = _row(512, key=11)
    stage1 = apply_top_n_sigma(lp, 1.0)
    poisoned = mx.exp(stage1) * stage1
    mx.eval(poisoned)
    assert any(math.isnan(v) for v in poisoned.tolist()), (
        "upstream's typical_p-after-a-mode chain is no longer NaN-poisoned -- "
        "re-evaluate KNOWN_DIVERGENCE 2, it may now be matchable"
    )

    cfg = SamplingConfig(temperature=1.0, top_n_sigma=1.0, typical_p=0.5, seed=1)
    ours = _sampler_keep(lp, cfg)
    assert ours, "chained modes emptied the row"
    # Chaining only removes, so the result stays inside the first mode's set.
    assert ours <= _reference_keep(
        lp, SamplingConfig(temperature=1.0, top_n_sigma=1.0, seed=1)
    )


def test_the_chained_mode_mask_is_never_empty():
    """The invariant that makes the all -inf row unreachable.

    Each mode keeps at least the best token the previous one left, so chaining
    cannot empty the mask. This is what actually protects the draw; the
    argmax backstop in batched_row_sample is insurance against a fourth mode
    being added later, not a live path.
    """
    empty = []
    for key in range(40):
        for scale in (0.5, 2.0, 8.0, 20.0):
            for ns in (0.0, 0.5, 1.0, 2.0):
                for pl in (False, True):
                    for typ in (1.0, 0.5, 0.1, 0.01):
                        if ns == 0.0 and not pl and typ >= 1.0:
                            continue
                        lp = _row(256, key=key, scale=scale)
                        mask = _new_modes_keep(
                            lp[None],
                            mx.array([0.7]),
                            mx.array([ns]),
                            mx.array([pl]),
                            mx.array([typ]),
                        )
                        mx.eval(mask)
                        if int(mx.sum(mask)) == 0:
                            empty.append((key, scale, ns, pl, typ))
    assert not empty, f"chained mode mask emptied: {empty[:5]}"


def test_an_empty_mode_mask_would_still_draw_a_real_token():
    """Exercise the backstop directly instead of hoping a draw reaches it.

    An all -inf row would make the rank-0 floor keep a -inf entry, and
    mx.random.categorical would then return a uniformly random token from the
    whole vocabulary -- silently. Force the mask empty and assert the argmax
    comes out instead.
    """
    lp = _row(512, key=3)
    argmax = int(mx.argmax(lp, axis=-1))
    cfg = SamplingConfig(temperature=0.7, top_n_sigma=1.0, typical_p=0.3, seed=1)

    def all_rejected(work, *args, **kwargs):
        return mx.zeros(work.shape, dtype=mx.bool_)

    drawn = set()
    with patch.object(ar_module, "_new_modes_keep", all_rejected):
        for seed in range(16):
            mx.random.seed(seed)
            token = _PositionedTargetSampler([cfg])(lp[None])
            mx.eval(token)
            drawn.add(int(token[0]))
    assert drawn == {argmax}, f"backstop drew {sorted(drawn)[:8]} instead of {argmax}"


@pytest.mark.parametrize("top_k", [2**31, 10**12, 10**18])
def test_enormous_top_k_does_not_crash_the_worker(top_k):
    """top_k has no upper bound on any request schema, here or on main.

    Above int32 the per-row array build raises std::bad_cast, and that raise
    lands in the GPU worker's except handler, which fails every co-batched
    request -- so one client's absurd top_k would end everyone else's
    in-flight generation.
    """
    lp = _row(256, key=1)
    token = _PositionedTargetSampler(
        [SamplingConfig(temperature=0.7, top_k=top_k, seed=1)]
    )(lp[None])
    mx.eval(token)
    assert 0 <= int(token[0]) < 256


@pytest.mark.parametrize("top_p", [0.5, 0.9])
def test_raw_logits_are_normalized_before_the_rank_filters(top_p):
    """The speculative loops pass raw lm_head logits, not log-normalized ones.

    exp() of an unnormalized row sums to e**logsumexp rather than 1, so the
    nucleus cutoff is satisfied almost everywhere and top_p stops filtering.
    dflash.py:153/276, eagle3.py:175/185/199 and mtp.py:172 all call the
    sampler this way, so the same request would filter differently depending on
    whether it was routed to speculative decoding.
    """
    raw = mx.random.normal((4096,), key=mx.random.key(5)) * 2.0
    lp = _log_normalize(raw)
    mx.eval(raw, lp)
    cfg = SamplingConfig(temperature=0.7, top_p=top_p, seed=1)
    assert _sampler_keep(raw, cfg) == _sampler_keep(lp, cfg)
    assert _sampler_keep(raw, cfg) == _reference_keep(lp, cfg)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_reduced_precision_matches_where_no_tie_straddles_the_cutoff(dtype):
    """KNOWN_DIVERGENCE 2: reduced precision may reorder ties at the boundary.

    Assert equality on rows whose reference cutoff has no duplicate value on the
    boundary, so a genuine semantic drift still fails here.
    """
    checked = 0
    for key in range(30):
        lp = _row(256, key=key, dtype=dtype)
        cfg = SamplingConfig(temperature=0.7, top_p=0.9, seed=1)
        ref = _reference_keep(lp.astype(mx.float32), cfg)
        vals = lp.astype(mx.float32).tolist()
        boundary = min((vals[i] for i in ref), default=None)
        if boundary is None or sum(1 for v in vals if v == boundary) != 1:
            continue  # a tie sits on the cutoff; skip, it is not a semantic case
        checked += 1
        assert _sampler_keep(lp, cfg) == ref, f"dtype={dtype} key={key}"
    assert checked >= 10, f"only {checked} tie-free rows exercised"


@pytest.mark.parametrize("filters", RANK_FILTERS, ids=lambda f: repr(sorted(f.items())))
def test_greedy_ignores_every_filter_like_make_sampler(filters):
    """make_sampler returns a bare argmax at temperature 0, whatever else is set."""
    lp = _log_normalize(mx.random.normal((3, 512), key=mx.random.key(31)) * 2.0)
    cfg = SamplingConfig(temperature=0.0, seed=1, **filters)
    ours = _PositionedTargetSampler([cfg] * 3)(lp)
    ref = make_sampler(temp=0)(lp)
    mx.eval(ours, ref)
    assert ours.tolist() == ref.tolist()


# --- the divergences above, pinned so they cannot drift silently ------------


def test_float32_ties_straddling_the_cutoff_keep_the_same_count():
    """KNOWN_DIVERGENCE 3. Different members, same cardinality and same mass.

    If this ever starts differing in SIZE it is no longer tie ordering and the
    entry no longer covers it.
    """
    row = _log_normalize(mx.array([3.0, 3.0, 3.0, 3.0, 0.0, -1.0, -2.0, -3.0]))
    mx.eval(row)
    cfg = SamplingConfig(temperature=1.0, top_p=0.5, seed=1)
    ours = _sampler_keep(row, cfg)
    ref = _reference_keep(row, cfg)
    assert len(ours) == len(ref), f"tie divergence changed the count: {ours} vs {ref}"


def test_typical_p_on_a_premasked_row_is_the_normalization_divergence():
    """KNOWN_DIVERGENCE 2, and the reachable half of it.

    A grammar mask or logit_bias=-inf produces a row whose exp-mass is not 1.
    Assert the divergence exists and is the normalization one, by showing it
    disappears once the row is renormalized before upstream sees it too.
    """
    base = _row(64, key=0)
    masked = mx.where(mx.arange(64) < 8, base, mx.array(NEG_INF))
    mx.eval(masked)
    cfg = SamplingConfig(temperature=0.7, typical_p=0.5, seed=1)

    assert _sampler_keep(masked, cfg) != _reference_keep(masked, cfg)
    # feed upstream the same normalized row and the disagreement goes away
    renormalized = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    mx.eval(renormalized)
    assert _sampler_keep(masked, cfg) == _reference_keep(renormalized, cfg)


@pytest.mark.parametrize("temperature", [1e-6, -1.0])
def test_temperature_edges_are_greedy_here(temperature):
    """KNOWN_DIVERGENCE 7. Reachable now that the schemas carry no bounds."""
    lp = _row(256, key=4)
    argmax = int(mx.argmax(lp, axis=-1))
    cfg = SamplingConfig(temperature=temperature, top_p=0.9, seed=1)
    drawn = set()
    for seed in range(8):
        mx.random.seed(seed)
        token = _PositionedTargetSampler([cfg])(lp[None])
        mx.eval(token)
        drawn.add(int(token[0]))
    assert drawn == {argmax}
