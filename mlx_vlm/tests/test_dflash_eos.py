from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.speculative.utils as speculative_utils


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "config_fields,expected_eos",
    [
        ({"eos_token_id": 99}, {99}),
        ({"eos_token_id": [98, 99]}, {98, 99}),
        ({}, None),
        (None, None),
    ],
)
def test_dflash_direct_dispatch_propagates_model_eos(
    monkeypatch, batch_size, config_fields, expected_eos
):
    calls = []

    def single(*args, **kwargs):
        calls.append(("single", kwargs))
        yield 11, None

    def batch(*args, **kwargs):
        calls.append(("batch", kwargs))
        yield [11] * batch_size, None

    monkeypatch.setattr(speculative_utils, "_dflash_rounds", single)
    monkeypatch.setattr(speculative_utils, "_dflash_rounds_batch", batch)
    output = list(
        speculative_utils.run_speculative_rounds(
            (
                SimpleNamespace(config=SimpleNamespace(**config_fields))
                if config_fields is not None
                else SimpleNamespace()
            ),
            SimpleNamespace(),
            [],
            mx.ones((batch_size, 1), dtype=mx.int32),
            mx.full((batch_size, 1), 10, dtype=mx.int32),
            mx.zeros((batch_size, 128)),
            SimpleNamespace(hidden_states=[mx.zeros((batch_size, 1, 1))]),
            draft_kind="dflash",
            max_tokens=8,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            sampler_is_greedy=True,
        )
    )
    assert len(calls) == 1
    route, kwargs = calls[0]
    assert route == ("single" if batch_size == 1 else "batch")
    assert kwargs["eos_token_ids"] == expected_eos
    assert kwargs["greedy_sampling"] is True
    bonus = kwargs["first_bonus"]
    assert (bonus if batch_size == 1 else bonus.tolist()) == (
        10 if batch_size == 1 else [10, 10]
    )
    assert [tokens for tokens, _ in output] == (
        [10, 11] if batch_size == 1 else [[10, 10], [11, 11]]
    )


@pytest.mark.parametrize("first_bonus,expected,calls", [(99, [], 0), (10, [99], 1)])
def test_dflash_singleton_stops_on_first_bonus_or_emitted_eos(
    first_bonus, expected, calls
):
    class LanguageModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                logits=mx.zeros((1, inputs.shape[1], 128)),
                hidden_states=[mx.zeros((1, inputs.shape[1], 1))],
                gdn_states=None,
            )

        def rollback_speculative_cache(self, *args):
            pass

    class Drafter:
        config = SimpleNamespace(target_layer_ids=[0], block_size=4)

        def reset(self, model):
            self.accept_lens = []
            self.draft_lens = []
            return []

        def draft_block(self, *args):
            return mx.array([[1, 2, 3]])

    lm = LanguageModel()
    output = list(
        speculative_utils._dflash_rounds(
            SimpleNamespace(language_model=lm),
            Drafter(),
            [],
            mx.zeros((1, 1, 1)),
            first_bonus=first_bonus,
            max_tokens=8,
            sampler=lambda logits: mx.array([[99, 0, 0, 0]]),
            eos_token_ids={99},
        )
    )
    assert [token for token, _ in output] == expected
    assert lm.calls == calls


def test_dflash_batch_round_stops_on_first_bonus_or_emitted_eos():
    class LanguageModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, cache, capture_layer_ids, speculative_verify):
            assert speculative_verify is True
            del cache, capture_layer_ids
            self.calls += 1
            batch, length = inputs.shape
            return SimpleNamespace(
                logits=mx.zeros((batch, length, 128)),
                hidden_states=[mx.zeros((batch, length, 1))],
                gdn_states=None,
            )

        def rollback_speculative_cache(self, *args):
            del args

    class Drafter:
        config = SimpleNamespace(target_layer_ids=[0], block_size=4)
        accept_lens = []
        draft_lens = []

        def reset(self, model):
            del model

        def make_cache(self):
            return []

        def draft_block(self, *args):
            del args
            return mx.array([[1, 2, 3]])

    first_bonus_lm = LanguageModel()
    first_bonus_rounds = list(
        speculative_utils._dflash_rounds_batch(
            SimpleNamespace(language_model=first_bonus_lm),
            Drafter(),
            [],
            mx.zeros((1, 1, 1)),
            first_bonus=mx.array([99]),
            max_tokens=8,
            sampler=lambda logits: mx.array([[99, 0, 0, 0]]),
            eos_token_ids={99},
        )
    )
    assert first_bonus_rounds == []
    assert first_bonus_lm.calls == 0

    emitted_lm = LanguageModel()
    emitted_rounds = list(
        speculative_utils._dflash_rounds_batch(
            SimpleNamespace(language_model=emitted_lm),
            Drafter(),
            [],
            mx.zeros((1, 1, 1)),
            first_bonus=mx.array([10]),
            max_tokens=8,
            sampler=lambda logits: mx.array([[99, 0, 0, 0]]),
            eos_token_ids={99},
        )
    )
    assert [tokens for tokens, _ in emitted_rounds] == [[99]]
    assert emitted_lm.calls == 1


def test_dflash_batch_filters_initial_eos_rows_before_verify():
    class Cache:
        def __init__(self):
            self.filtered = []

        def filter(self, keep):
            self.filtered.append(keep.tolist())

    class LanguageModel:
        def __init__(self, cache):
            self.cache = cache
            self.batch_sizes = []

        def __call__(self, inputs, cache, capture_layer_ids, speculative_verify):
            assert speculative_verify is True
            del capture_layer_ids
            assert cache == [self.cache]
            assert self.cache.filtered == [[1]]
            self.batch_sizes.append(inputs.shape[0])
            return SimpleNamespace(
                logits=mx.zeros((1, inputs.shape[1], 128)),
                hidden_states=[mx.zeros((1, inputs.shape[1], 1))],
                gdn_states=None,
            )

        def rollback_speculative_cache(self, *args):
            del args

    class Drafter:
        config = SimpleNamespace(target_layer_ids=[0], block_size=4)
        accept_lens = []
        draft_lens = []

        def reset(self, model):
            del model

        def make_cache(self):
            return []

        def draft_block(self, *args):
            del args
            return mx.array([[1, 2, 3]])

    cache = Cache()
    language_model = LanguageModel(cache)
    rounds = list(
        speculative_utils._dflash_rounds_batch(
            SimpleNamespace(language_model=language_model),
            Drafter(),
            [cache],
            mx.zeros((2, 1, 1)),
            first_bonus=mx.array([99, 10]),
            max_tokens=8,
            sampler=lambda logits: mx.array([[99, 0, 0, 0]]),
            eos_token_ids={99},
        )
    )

    assert [tokens for tokens, _ in rounds] == [[None, 99]]
    assert cache.filtered == [[1]]
    assert language_model.batch_sizes == [1]
