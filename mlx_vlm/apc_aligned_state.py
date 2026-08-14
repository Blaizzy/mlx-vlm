"""Block-aligned reuse for non-pageable cache state.

Recurrent and convolution state summarises every token before it, so it cannot
be sliced to an arbitrary position the way attention KV can. It can, however,
be checkpointed: a node that covers a block boundary can hold the state as it
stood at that boundary, and a later request sharing that prefix resumes from
it.

The planner below turns one scheduling step into the two motions that keep such
state coherent:

``zero`` starts a request whose prefix is empty, because the node it lands on
may still hold another request's bytes. ``copy`` carries state from the node a
request just left into the node it is entering, leaving the old node intact as
a checkpoint.

A prefix hit needs no third motion: the request is admitted with its matched
length, so the first copy of the step reads the checkpoint it hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = ["StatePlan", "StateStep", "plan_state_motions"]


@dataclass(frozen=True)
class StateStep:
    """One request's position in a step: what it had, and what it runs now."""

    node_ids: Sequence[int]
    computed: int
    scheduled: int
    shareable: Optional[Sequence[bool]] = None


@dataclass
class StatePlan:
    """Node ids to clear, node pairs to carry forward, and each step's target."""

    zero: List[int] = field(default_factory=list)
    copy: List[Tuple[int, int]] = field(default_factory=list)
    targets: List[int] = field(default_factory=list)


def _node_index(position: int, stride: int) -> int:
    return position // stride


def plan_state_motions(
    steps: Sequence[StateStep],
    stride: int,
    copy_on_write: bool = True,
) -> StatePlan:
    """Plan state motion for one scheduling step.

    ``stride`` is the checkpoint interval in tokens. With ``copy_on_write`` a
    node is only carried forward when it is shareable, since a node no other
    request can reach is free to advance in place.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    plan = StatePlan()
    for step in steps:
        if step.scheduled <= 0:
            raise ValueError("every scheduled step must run at least one token")

        last_position = step.computed + step.scheduled - 1
        target_index = _node_index(last_position, stride)
        if target_index >= len(step.node_ids):
            raise ValueError(
                f"step needs node index {target_index} but was given "
                f"{len(step.node_ids)} nodes"
            )

        target = step.node_ids[target_index]
        plan.targets.append(target)

        if step.computed == 0:
            plan.zero.append(target)
            continue

        source_index = _node_index(step.computed - 1, stride)
        source = step.node_ids[source_index]
        if source == target:
            continue

        if copy_on_write and not _is_shareable(step, source_index):
            continue

        plan.copy.append((source, target))

    return plan


def _is_shareable(step: StateStep, index: int) -> bool:
    if step.shareable is None:
        return True
    if index >= len(step.shareable):
        return True
    return bool(step.shareable[index])
