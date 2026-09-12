"""Target probabilities, reasoning boundaries, and isolated request metrics."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import BatchGenerator, generate_step
from mlx_vlm.speculative.sampling import accept_sampled
from mlx_vlm.tests import test_speculative_serving as serving
from mlx_vlm.tests.test_speculative_serving import NeverStop, wrapper
from mlx_vlm.thinking import ThinkingBudgetLogitsProcessor
from mlx_vlm.utils import ThinkingBudgetCriteria

pair = serving.pair


def collect(target, draft, prompt, **kwargs):
    holder = []
    kwargs.setdefault("max_tokens", 10)
    result = list(
        generate_step(
            mx.array([prompt]),
            wrapper(target),
            None,
            None,
            draft_model=draft,
            draft_kind="mtp",
            draft_block_size=4,
            prefill_step_size=2,
            speculative_cache_callback=holder.append,
            **kwargs,
        )
    )
    return result, holder[0] if holder else None


@pytest.mark.parametrize("temperature", [0, 0.8])
@pytest.mark.parametrize("enabled", [False, True])
def test_target_logprobs_match_ar_with_processors(pair, temperature, enabled):
    target, draft = pair
    prompt = [1, 2, 3, 4, 2]
    kwargs = dict(
        temperature=temperature,
        seed=17,
        top_p=0.9,
        repetition_penalty=1.1,
        presence_penalty=0.1,
        frequency_penalty=0.05,
        logit_bias={5: 0.3},
        logprobs=enabled,
    )
    expected, _ = collect(target, None, prompt, **kwargs)
    actual, _ = collect(target, draft, prompt, **kwargs)
    assert [x[0] for x in actual] == [x[0] for x in expected]
    for (_, observed), (_, reference) in zip(actual, expected):
        if enabled:
            assert observed is not None
            assert mx.allclose(observed, reference, atol=2e-5).item()
        else:
            assert observed is None


@pytest.mark.parametrize("enabled", [False, True])
def test_ragged_server_reports_probabilities_and_row_metrics(pair, enabled):
    target, draft = pair
    prompts = [[1, 2], [3, 4, 5, 6]]

    def run_batch(drafter):
        generator = BatchGenerator(
            target,
            SimpleNamespace(stopping_criteria=NeverStop()),
            draft_model=drafter,
            draft_kind="mtp" if drafter is not None else None,
            draft_block_size=3,
            compute_logprobs=enabled,
            top_logprobs_k=3 if enabled else 0,
            prefill_batch_size=2,
            completion_batch_size=2,
            prefill_step_size=2,
        )
        ids = generator.insert(
            prompts,
            max_tokens=[1, 10],
            prompt_kwargs=[
                {"inputs_embeds": target.model.embed_tokens(mx.array([prompt]))}
                for prompt in prompts
            ],
        )
        rows = {uid: [] for uid in ids}
        while generator.has_work:
            _, responses = generator.next()
            for response in responses:
                if response.token is not None:
                    rows[response.uid].append(response)
        generator.close()
        return ids, rows

    ids, expected = run_batch(None)
    _, rows = run_batch(draft)
    assert rows[ids[0]][-1].speculative_stats == (0, 0, 0)
    assert rows[ids[1]][-1].speculative_stats[0] > 0
    assert not hasattr(draft, "speculative_stats")
    for uid in ids:
        assert [r.token for r in rows[uid]] == [r.token for r in expected[uid]]
        for result, reference in zip(rows[uid], expected[uid]):
            if enabled:
                assert result.token_logprob == pytest.approx(
                    reference.token_logprob, abs=2e-5
                )
                assert [t for t, _ in result.top_logprobs] == [
                    t for t, _ in reference.top_logprobs
                ]
                assert [lp for _, lp in result.top_logprobs] == pytest.approx(
                    [lp for _, lp in reference.top_logprobs], abs=2e-5
                )
            else:
                assert result.token_logprob is None
                assert result.top_logprobs is None


def budget_criteria(budget):
    tokenizer = SimpleNamespace(
        encode=lambda text, **_: {
            "<think>": [27, 28],
            "</think>": [29, 30],
            "\n": [31],
        }[text]
    )
    return ThinkingBudgetCriteria(
        tokenizer,
        budget,
        thinking_start_token="<think>",
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )


@pytest.mark.parametrize("budget", [0, 2, 5])
@pytest.mark.parametrize("temperature", [0, 0.8])
def test_thinking_budget_masks_target_before_sampling(pair, budget, temperature):
    target, draft = pair
    prompt = [1, 2, 27, 28]
    kwargs = dict(
        logprobs=True, temperature=temperature, seed=17, logit_bias={5: 100.0}
    )
    expected, _ = collect(
        target, None, prompt, thinking_budget_criteria=budget_criteria(budget), **kwargs
    )
    actual, _ = collect(
        target,
        draft,
        prompt,
        thinking_budget_criteria=budget_criteria(budget),
        **kwargs,
    )
    tokens = [t for t, _ in actual]
    assert tokens == [5] * budget + [29, 30] + [5] * (8 - budget)
    assert tokens == [t for t, _ in expected]
    for index in (budget, budget + 1):
        assert actual[index][1][tokens[index]].item() == 0.0


def test_budget_inside_accepted_block_uses_only_accepted_prefix():
    policy = ThinkingBudgetLogitsProcessor(2, [27, 28], [29, 30], prompt_length=2)
    logits = mx.zeros((1, 4, 32)).at[:, :, 5].add(100)
    accepted = accept_sampled(
        mx.array([[5, 5, 5]]),
        logits,
        [4],
        None,
        [0],
        [1],
        processors=[[policy]],
        contexts=[[27, 28]],
        greedy=True,
        compute_logprobs=True,
    )
    assert accepted.tokens == [[5, 5, 29]]
    # A speculative mask computation cannot consume persistent thinking state.
    assert mx.argmax(policy(mx.array([27, 28]), logits[:, 0]), axis=-1).item() == 5
    assert (
        mx.argmax(policy(mx.array([27, 28, 5, 5, 29]), logits[:, 0]), axis=-1).item()
        == 30
    )


@pytest.mark.parametrize("draft_enabled", [False, True])
def test_server_thinking_budget_uses_same_sampling_policy(pair, draft_enabled):
    target, draft = pair
    generator = BatchGenerator(
        target,
        SimpleNamespace(stopping_criteria=NeverStop()),
        draft_model=draft if draft_enabled else None,
        draft_kind="mtp" if draft_enabled else None,
        draft_block_size=4,
        prefill_batch_size=2,
        completion_batch_size=2,
        prefill_step_size=2,
        compute_logprobs=True,
    )
    prompts = [[1, 27, 28], [2, 3, 27, 28]]

    def prefer_five(tokens, logits):
        return logits.at[:, 5].add(100)

    ids = generator.insert(
        prompts,
        max_tokens=8,
        prompt_kwargs=[
            {"inputs_embeds": target.model.embed_tokens(mx.array([p]))} for p in prompts
        ],
        logits_processors=[[prefer_five], [prefer_five]],
        thinking_budget_criteria=[budget_criteria(0), budget_criteria(3)],
    )
    rows = {uid: [] for uid in ids}
    while generator.has_work:
        _, responses = generator.next()
        for response in responses:
            if response.token is not None:
                rows[response.uid].append(response.token)
    generator.close()
    assert rows[ids[0]] == [29, 30] + [5] * 6
    assert rows[ids[1]] == [5] * 3 + [29, 30] + [5] * 3


def test_prefix_restore_starts_new_request_counters(pair):
    target, draft = pair
    _, first = collect(target, draft, [1, 2, 3])
    assert first.stats[0].rounds > 0
    restored = first.restore(first.checkpoint())
    assert restored.stats[0].snapshot() == (0, 0, 0)
    assert first.stats[0].rounds > 0


def test_interleaved_generators_share_weights_without_sharing_metrics(pair):
    target, draft = pair
    holder = []
    iterator = generate_step(
        mx.array([[1, 2, 3]]),
        wrapper(target),
        None,
        None,
        draft_model=draft,
        draft_kind="mtp",
        max_tokens=12,
        speculative_cache_callback=holder.append,
    )
    next(iterator)
    next(iterator)
    state = holder[0]
    before = state.stats[0].snapshot()
    _, other = collect(target, draft, [5, 6, 7], max_tokens=8)
    assert state.stats[0].snapshot() == before
    assert other.stats[0] is not state.stats[0]
    iterator.close()
    assert not hasattr(draft, "speculative_stats")


def test_natural_and_partial_reasoning_terminators():
    processor = ThinkingBudgetLogitsProcessor(2, [27, 28], [29, 30], prompt_length=2)
    scores = mx.arange(32)[None].astype(mx.float32)
    # Natural termination leaves the answer distribution unchanged.
    assert mx.array_equal(processor(mx.array([27, 28, 29, 30]), scores), scores).item()
    # A partial terminator at the boundary is completed, not repeated.
    masked = processor(mx.array([27, 28, 5, 29]), scores)
    assert mx.argmax(masked).item() == 30
    assert masked.shape == scores.shape


def test_forced_token_top_logprobs_serialize_as_valid_json():
    import json

    from mlx_vlm.server.app import _make_logprob_content

    tokenizer = SimpleNamespace(decode=lambda ids, **_: str(ids[0]))
    result = _make_logprob_content(
        tokenizer, 29, 0.0, top_logprobs=[(29, 0.0), (30, -float("inf"))], top_k=2
    )
    payload = json.loads(json.dumps(result.model_dump(), allow_nan=False))
    assert payload["logprob"] == 0.0
    assert len(payload["top_logprobs"]) == 1
