"""Cache lifetimes at target, drafter, and generator boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.models.glm5_next.language import LanguageModel as GlmLanguageModel
from mlx_vlm.models.qwen4_exp.language import LanguageModel as QwenLanguageModel
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.dflash import _dflash_rounds, _dflash_rounds_batch
from mlx_vlm.speculative.drafters.glm5_next_mtp import (
    Glm5NextMTPDraftModel,
    ModelConfig,
)
from mlx_vlm.speculative.eagle3 import _eagle3_rounds, _eagle3_rounds_batch
from mlx_vlm.speculative.mtp import _mtp_rounds, _mtp_rounds_batch
from mlx_vlm.speculative.targets import bind_speculative_target
from mlx_vlm.tests.test_qwen4_mtp import _outer_config, _tiny_text_config
from mlx_vlm.tests.test_speculative import _tiny_glm5_next_text_config


def test_ordinary_qwen4_decode_does_not_record_speculation():
    config = _tiny_text_config()
    config.linear_key_head_dim = config.linear_value_head_dim = 32
    model = QwenLanguageModel(config, _outer_config())
    model.eval()
    caches = model.make_cache()
    mx.eval(model(mx.array([[1, 2]]), cache=caches).logits)
    for token in (3, 4):
        output = model(mx.array([[token]]), cache=caches)
        mx.eval(output.logits)
        assert output.gdn_states is None
        for cache in caches:
            if isinstance(cache, ArraysCache):
                assert not cache.is_speculating
                assert cache.nbytes == sum(
                    x.nbytes for x in cache.state if x is not None
                )


def test_glm_sampler_failure_restores_temporal_and_append_caches():
    model = GlmLanguageModel(_tiny_glm5_next_text_config())
    model.eval()
    target = bind_speculative_target(model)
    caches = model.make_cache()
    mx.eval(model(mx.array([[1, 2]]), cache=caches).logits)
    initial = list(caches[0].state)

    def fail(_):
        raise RuntimeError("injected sampler failure")

    with pytest.raises(RuntimeError, match="injected sampler failure"):
        target.speculative_verify_logits(mx.array([[3, 4]]), caches, fail)
    assert not caches[0].is_speculating
    assert all(a is b for a, b in zip(initial, caches[0].state))
    assert caches[1][0].offset == 2


def test_glm_adapter_shares_weights_and_preserves_serving_model():
    model = GlmLanguageModel(_tiny_glm5_next_text_config())
    model.eval()
    inputs = mx.array([[1, 2, 3]])
    before = model(inputs).logits
    mx.eval(before)
    original_projection = model.model.layers[0].self_attn.qkv_proj
    target = bind_speculative_target(model)
    assert target.model is not model.model
    assert target.model.layers[0] is not model.model.layers[0]
    assert target.model.layers[0].self_attn.qkv_proj.module is original_projection
    assert not hasattr(model, "speculative_verify_hidden")
    caches = model.make_cache()
    hidden, _, transaction = target.speculative_verify_hidden(inputs, caches)
    mx.eval(hidden)
    transaction.commit(inputs.shape[1])
    after = model(inputs).logits
    mx.eval(after)
    assert mx.array_equal(before, after).item()
    assert model.model.layers[0].self_attn.qkv_proj is original_projection
    assert isinstance(original_projection, nn.Linear)


@pytest.mark.parametrize("batch", [1, 4])
def test_glm_block_verification_matches_stepwise_bfloat16(batch):
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hc_mult = 4
    model = GlmLanguageModel(config)
    # Retain the model's required FP32 parameters.
    from mlx.utils import tree_flatten

    model.load_weights(
        [
            (name, value.astype(mx.bfloat16) if model.cast_predicate(name) else value)
            for name, value in tree_flatten(model.parameters())
        ]
    )
    model.eval()
    target = bind_speculative_target(model)
    caches = [_make_cache(model, left_padding=[0] * batch) for _ in range(2)]
    prefix = mx.array([[1, 2, 3]] * batch)
    for cache in caches:
        mx.eval(model(prefix, cache=cache).logits)
    tokens = mx.array([[4, 5, 6, 7]] * batch)
    steps = []
    for position in range(tokens.shape[1]):
        hidden, _, transaction = target.speculative_verify_hidden(
            tokens[:, position : position + 1], caches[0]
        )
        mx.eval(hidden)
        transaction.commit(1)
        steps.append(hidden)
    actual, _, transaction = target.speculative_verify_hidden(tokens, caches[1])
    mx.eval(actual)
    transaction.commit(tokens.shape[1])
    expected = mx.concatenate(steps, axis=1)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [1, 2])
def test_glm_drafter_rejection_restores_pool_after_multiple_appends(batch):
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    model = GlmLanguageModel(config)
    drafter = Glm5NextMTPDraftModel(ModelConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.reset(model, left_padding=[0] * batch if batch > 1 else None)
    hidden = mx.random.normal((batch, 1, config.hidden_size))
    mx.eval(drafter._forward_tokens(mx.array([[1]] * batch), hidden, mx.int32))
    reference = deepcopy(drafter)
    sampler = lambda logits: mx.argmax(logits, axis=-1)
    tokens = drafter.draft_block(
        2 if batch == 1 else mx.full((batch,), 2), hidden, None, 4, sampler, greedy=True
    )
    mx.eval(tokens)
    assert drafter._round_appended == 3
    transaction = drafter._round_transaction
    verified = mx.random.normal((batch, 4, config.hidden_size))
    for candidate in (drafter, reference):
        candidate.accept_verified_tokens_batch(
            verified, tokens, [0] * batch, [[8]] * batch, sampler, greedy=True
        )
        mx.eval(candidate.draft_eval_state())
    assert not transaction.active
    actual_pool, expected_pool = drafter._cache[0][2], reference._cache[0][2]
    assert mx.array_equal(
        mx.array(actual_pool.offset), mx.array(expected_pool.offset)
    ).item()
    assert actual_pool.remainder == expected_pool.remainder
    for actual, expected in zip(actual_pool.state, expected_pool.state):
        if actual is None or expected is None:
            assert actual is expected
        else:
            assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(drafter._seed_token, reference._seed_token).item()
    assert mx.array_equal(drafter._seed_hidden, reference._seed_hidden).item()


class _Target:
    """Minimal target that updates real recurrent and KV caches."""

    def __init__(self, token):
        self.token = token
        self.transaction = None

    def __call__(self, inputs, cache, **kwargs):
        batch, length = inputs.shape
        self.transaction = start_speculative_cache(cache, length)
        initial = cache[0][0]
        states = initial[:, None] + mx.arange(1, length + 1)[None, :, None]
        cache[0][0] = states[:, -1]
        cache[0].record_speculative_states(0, states[:, :-1], states[:, -1])
        kv = mx.zeros((batch, 1, length, 1))
        cache[1].update_and_fetch(kv, kv)
        hidden = mx.zeros((batch, length, 4))
        logits = mx.broadcast_to(mx.eye(8)[self.token], (batch, length, 8))
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden],
            shared_kv_states={},
            gdn_states=self.transaction,
        )

    def speculative_verify_logits(self, inputs, cache, sampler):
        output = self(inputs, cache)
        try:
            return (
                output.hidden_states[0],
                {},
                output.gdn_states,
                sampler(output.logits),
            )
        except BaseException:
            output.gdn_states.abort()
            raise

    def rollback_speculative_cache(self, *args):
        raise AssertionError("transactions must own cache commit")


class _Drafter:
    prefer_requested_block_size = True

    def __init__(self):
        self.config = SimpleNamespace(block_size=2, target_layer_ids=[0])
        self.accept_lens = []
        self.draft_lens = []

    def reset(self, model, left_padding=None):
        return []

    def make_cache(self):
        return []

    def set_shared_kv(self, *args, **kwargs):
        pass

    def draft_block(
        self, bonus, hidden, cache, block_size, sampler, token_dtype, **kwargs
    ):
        return mx.full((hidden.shape[0], block_size - 1), 4, dtype=token_dtype)


def _round_generator(kind, batch, token, sampler):
    target = _Target(token)
    caches = [ArraysCache(1), BatchKVCache([0] * batch) if batch > 1 else KVCache()]
    caches[0][0] = mx.zeros((batch, 1))
    initial = mx.zeros((batch, 1, 2, 1))
    caches[1].update_and_fetch(initial, initial)
    functions = {
        "mtp": (_mtp_rounds, _mtp_rounds_batch),
        "dflash": (_dflash_rounds, _dflash_rounds_batch),
        "eagle3": (_eagle3_rounds, _eagle3_rounds_batch),
    }
    fn = functions[kind][batch > 1]
    kwargs = dict(
        first_bonus=1 if batch == 1 else mx.ones((batch,), dtype=mx.int32),
        max_tokens=4,
        sampler=sampler,
        draft_block_size=2,
    )
    if kind == "mtp":
        kwargs["shared_kv_states"] = {}
    generator = fn(target, _Drafter(), caches, mx.zeros((batch, 1, 4)), **kwargs)
    return generator, target, caches


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("token", [4, 7])
def test_round_commits_before_yield_and_generator_close(kind, batch, token):
    generator, target, caches = _round_generator(
        kind, batch, token, lambda x: mx.argmax(x, axis=-1)
    )
    next(generator)
    assert not target.transaction.active
    assert not caches[0].is_speculating
    retained = 2 if token == 4 else 1
    assert caches[0][0].tolist() == [[float(retained)]] * batch
    offset = caches[1].offset
    assert (offset.tolist() if isinstance(offset, mx.array) else [offset]) == [
        2 + retained
    ] * batch
    generator.close()


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
@pytest.mark.parametrize("batch", [1, 2])
def test_failed_target_sampling_aborts_round(kind, batch):
    def fail(_):
        raise RuntimeError("injected sampler failure")

    generator, target, caches = _round_generator(kind, batch, 4, fail)
    with pytest.raises(RuntimeError, match="injected sampler failure"):
        next(generator)
    assert not target.transaction.active
    assert not caches[0].is_speculating
    assert caches[0][0].tolist() == [[0.0]] * batch
    offset = caches[1].offset
    assert (offset.tolist() if isinstance(offset, mx.array) else [offset]) == [
        2
    ] * batch


def test_mtp_target_failure_aborts_glm_draft_round(monkeypatch):
    import mlx_vlm.speculative.mtp as mtp

    config = _tiny_glm5_next_text_config()
    model = GlmLanguageModel(config)
    model.eval()
    caches = model.make_cache()
    prompt = mx.array([[1, 2]])
    output = model(prompt, cache=caches, return_hidden=True)
    mx.eval(output.logits, output.hidden_states)
    assert mtp._mtp_cache_offset_max(caches) == 2
    drafter = Glm5NextMTPDraftModel(ModelConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.prefer_requested_block_size = True

    def fail(*args, **kwargs):
        assert drafter._round_appended == 2
        assert drafter._round_transaction.active
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(mtp, "_mtp_verify_target", fail)
    generator = mtp._mtp_rounds(
        model,
        drafter,
        caches,
        output.hidden_states[-1],
        {},
        prompt_tokens=prompt,
        first_bonus=3,
        max_tokens=8,
        sampler=lambda x: mx.argmax(x, axis=-1),
        draft_block_size=4,
        greedy_sampling=True,
    )
    with pytest.raises(RuntimeError, match="injected target failure"):
        next(generator)
    assert drafter._round_transaction is None
    assert drafter._next_position == 2
    assert drafter._seed_token is not None
    assert not drafter._cache[0][2].is_speculating
    assert mtp._mtp_cache_offset_max(caches) == 2


def test_transaction_scope_aborts_uncommitted_temporal_and_kv_updates():
    recurrent = ArraysCache(1)
    recurrent[0] = mx.zeros((1, 1))
    kv = KVCache()
    initial = mx.zeros((1, 1, 2, 1))
    kv.update_and_fetch(initial, initial)
    with pytest.raises(RuntimeError, match="injected failure"):
        with start_speculative_cache([recurrent, kv], 2):
            recurrent[0] = mx.ones((1, 1))
            kv.update_and_fetch(initial, initial)
            raise RuntimeError("injected failure")
    assert recurrent[0].item() == 0
    assert not recurrent.is_speculating
    assert kv.offset == 2


def test_unsupported_target_argmax_reuses_verified_hidden():
    from mlx_vlm.speculative.mtp import _mtp_verify_target

    calls = []
    caches = [ArraysCache(1)]
    caches[0][0] = mx.zeros((1, 1))

    def forward(inputs, cache):
        calls.append(inputs)
        transaction = start_speculative_cache(cache, inputs.shape[1])
        return mx.ones((1, 2, 4)), {}, transaction

    target = SimpleNamespace(
        speculative_verify_hidden=forward,
        speculative_argmax_from_hidden=lambda hidden: None,
        speculative_logits_from_hidden=lambda hidden: hidden,
    )
    result = _mtp_verify_target(
        target, mx.array([[1, 2]]), caches, lambda x: mx.argmax(x, axis=-1)
    )
    assert len(calls) == 1
    assert result.target_tokens.tolist() == [[0, 0]]
    result.abort()
    assert not caches[0].is_speculating


def test_glm_mtp_generation_matches_ordinary_decode():
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hc_mult = 4
    model = GlmLanguageModel(config)
    model.eval()
    prompt = mx.array([[1, 2, 3]])
    ordinary_cache, speculative_cache = model.make_cache(), model.make_cache()
    expected = []
    inputs = prompt
    for _ in range(8):
        token = mx.argmax(model(inputs, cache=ordinary_cache).logits[:, -1], axis=-1)
        expected.append(token.item())
        inputs = token[:, None]
    output = model(prompt, cache=speculative_cache, return_hidden=True)
    first = mx.argmax(output.logits[:, -1], axis=-1).item()
    drafter = Glm5NextMTPDraftModel(ModelConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.prefer_requested_block_size = True
    actual = [first] + [
        token
        for token, _ in _mtp_rounds(
            model,
            drafter,
            speculative_cache,
            output.hidden_states[-1],
            {},
            prompt_tokens=prompt,
            first_bonus=first,
            max_tokens=8,
            sampler=lambda x: mx.argmax(x, axis=-1),
            draft_block_size=4,
            greedy_sampling=True,
        )
    ]
    assert actual == expected
    assert not speculative_cache[0].is_speculating
    assert not speculative_cache[1][2].is_speculating
    assert not drafter._cache[0][2].is_speculating
