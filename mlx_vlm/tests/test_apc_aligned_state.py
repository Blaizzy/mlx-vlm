import pytest

from mlx_vlm.apc_aligned_state import checkpoint_schedule
from mlx_vlm.generate.ar import _pending_checkpoint_lens

IMAGE = 900


def test_schedule_walks_the_stride():
    tokens = list(range(100))

    assert checkpoint_schedule(tokens, stride=25) == [25, 50, 75]


def test_schedule_leaves_room_to_generate():
    tokens = list(range(100))

    assert checkpoint_schedule(tokens, stride=25, guard_tokens=30) == [25, 50]


def test_short_prompt_has_no_boundary():
    assert checkpoint_schedule(list(range(10)), stride=25) == []


def test_schedule_keeps_the_earliest_boundaries_under_a_limit():
    tokens = list(range(1000))

    assert checkpoint_schedule(tokens, stride=100, limit=3) == [100, 200, 300]


def test_a_limit_of_zero_disables_checkpointing():
    assert checkpoint_schedule(list(range(100)), stride=25, limit=0) == []


def test_boundaries_move_past_media_so_the_suffix_is_text_only():
    tokens = list(range(40)) + [IMAGE] * 20 + list(range(40))

    schedule = checkpoint_schedule(tokens, stride=25, media_token_ids=[IMAGE])

    assert schedule
    assert all(IMAGE not in tokens[boundary:] for boundary in schedule)


def test_boundaries_are_strictly_increasing():
    tokens = list(range(30)) + [IMAGE] * 30 + list(range(40))

    schedule = checkpoint_schedule(tokens, stride=10, media_token_ids=[IMAGE])

    assert schedule == sorted(set(schedule))


def test_stride_must_be_positive():
    with pytest.raises(ValueError):
        checkpoint_schedule([1, 2, 3], stride=0)


def test_pending_lengths_accept_one_value_or_many():
    def store(prefix_len, cache):
        return None

    assert _pending_checkpoint_lens(store, 32, prompt_len=100) == [32]
    assert _pending_checkpoint_lens(store, [64, 32], prompt_len=100) == [32, 64]


def test_pending_lengths_drop_values_outside_the_prompt():
    def store(prefix_len, cache):
        return None

    assert _pending_checkpoint_lens(store, [0, 50, 100, 250], prompt_len=100) == [50]


def test_pending_lengths_are_empty_without_a_store_callback():
    assert _pending_checkpoint_lens(None, [10, 20], prompt_len=100) == []
