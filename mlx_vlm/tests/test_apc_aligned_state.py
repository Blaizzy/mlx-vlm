import pytest

from mlx_vlm.apc_aligned_state import StateStep, plan_state_motions


def test_fresh_request_zeroes_its_first_node():
    plan = plan_state_motions([StateStep([7, 8], computed=0, scheduled=4)], stride=16)

    assert plan.zero == [7]
    assert plan.copy == []
    assert plan.targets == [7]


def test_step_inside_one_node_moves_nothing():
    plan = plan_state_motions([StateStep([3, 4], computed=2, scheduled=4)], stride=16)

    assert plan.zero == []
    assert plan.copy == []
    assert plan.targets == [3]


def test_crossing_a_boundary_carries_state_forward():
    plan = plan_state_motions([StateStep([3, 9], computed=15, scheduled=2)], stride=16)

    assert plan.copy == [(3, 9)]
    assert plan.targets == [9]


def test_prefix_hit_resumes_from_the_checkpoint_it_matched():
    hit = StateStep([4, 5, 6], computed=32, scheduled=1)

    plan = plan_state_motions([hit], stride=16)

    assert plan.zero == []
    assert plan.copy == [(5, 6)]
    assert plan.targets == [6]


def test_private_nodes_advance_in_place_under_copy_on_write():
    private = StateStep([3, 9], computed=15, scheduled=2, shareable=[False, False])

    assert plan_state_motions([private], stride=16).copy == []
    assert plan_state_motions([private], stride=16, copy_on_write=False).copy == [
        (3, 9)
    ]


def test_shareable_nodes_are_always_carried_forward():
    shared = StateStep([3, 9], computed=15, scheduled=2, shareable=[True, False])

    assert plan_state_motions([shared], stride=16).copy == [(3, 9)]


def test_motions_batch_across_requests():
    steps = [
        StateStep([1, 2], computed=0, scheduled=1),
        StateStep([3, 4], computed=15, scheduled=2),
        StateStep([5, 6], computed=4, scheduled=2),
    ]

    plan = plan_state_motions(steps, stride=16)

    assert plan.zero == [1]
    assert plan.copy == [(3, 4)]
    assert plan.targets == [1, 4, 5]


def test_a_long_step_lands_on_the_node_holding_its_last_token():
    plan = plan_state_motions(
        [StateStep([1, 2, 3], computed=0, scheduled=40)], stride=16
    )

    assert plan.targets == [3]
    assert plan.zero == [3]


def test_stride_must_be_positive():
    with pytest.raises(ValueError):
        plan_state_motions([StateStep([1], computed=0, scheduled=1)], stride=0)


def test_empty_step_is_rejected():
    with pytest.raises(ValueError):
        plan_state_motions([StateStep([1], computed=0, scheduled=0)], stride=16)


def test_missing_node_for_target_is_rejected():
    with pytest.raises(ValueError):
        plan_state_motions([StateStep([1], computed=0, scheduled=40)], stride=16)
