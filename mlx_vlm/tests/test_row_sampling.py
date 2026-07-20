import mlx.core as mx
import pytest
from mlx_vlm.generate.ar import SamplingConfig, batched_row_sample
from mlx_vlm.sample_utils import apply_top_k

NEG_INF = -float("inf")


def _keys(n):
    return mx.stack([mx.random.key(i) for i in range(n)])


def test_config_is_frozen_and_hashable():
    c = SamplingConfig(temperature=0.7, top_p=0.9, top_k=40, min_p=0.0, seed=1)
    with pytest.raises(Exception):
        c.temperature = 0.1  # frozen
    assert len({c, SamplingConfig(temperature=0.7, top_p=0.9, top_k=40, min_p=0.0, seed=1)}) == 1


def _sampled_set(row_logprobs, *, top_k, top_p=1.0, min_p=0.0, temperature=1.0, n=256):
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
        lp, temperature=mx.zeros(2), top_p=mx.ones(2),
        top_k=mx.zeros(2, dtype=mx.int32), min_p=mx.zeros(2), keys=_keys(2),
    )
    assert got.tolist() == [1, 0]  # argmax per row, no NaN from temp==0


def test_mixed_greedy_and_sampled_batch():
    lp = mx.array([[0.0, 9.0, 0.0], [1.0, 1.0, 1.0]])
    got = batched_row_sample(
        lp, temperature=mx.array([0.0, 1.0]), top_p=mx.ones(2),
        top_k=mx.zeros(2, dtype=mx.int32), min_p=mx.zeros(2), keys=_keys(2),
    )
    assert int(got[0].item()) == 1  # greedy row -> argmax
    assert 0 <= int(got[1].item()) < 3  # sampled row -> valid draw
