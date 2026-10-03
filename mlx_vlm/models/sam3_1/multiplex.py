"""Multiplex state management for SAM 3.1 video tracking.

Port of sam3/model/multiplex_utils.py (inference subset).

A multiplex state maps objects (data space, batch dim) to slots in buckets
(multiplex space, num_buckets x multiplex_count). The mask decoder runs on
whole buckets at once; mux/demux convert between the two spaces via
precomputed gather/scatter indices (exact copies, unlike permutation matmuls).
"""

import math
from typing import Dict, List, Optional

import mlx.core as mx

# Special values for object tracking
PADDING_NUM = -1  # Marks empty slots in buckets
REMOVED_NUM = -1116  # Marks objects that have been removed


class MultiplexState:
    """Records the assignment of each object to a (bucket, slot) pair.

    Supports:
        mux:   (total_valid_entries, ...) -> (num_buckets, multiplex_count, ...)
        demux: (num_buckets, multiplex_count, ...) -> (total_valid_entries, ...)
        add_objects / remove_objects for dynamic object management
    """

    def __init__(
        self,
        assignments: List[List[int]],
        allowed_bucket_capacity: int,
        *,
        object_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            assignments: list of buckets; each bucket is a list of object indices
                (0..total_valid-1), PADDING_NUM for empty slots, or REMOVED_NUM
                for removed objects.
            allowed_bucket_capacity: max non-padding entries per bucket.
            object_ids: optional bookkeeping of global object IDs.
        """
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(self, assignments, *, object_ids=None):
        self.assignments = [list(b) for b in assignments]
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            raise ValueError("No buckets found in the state")
        self.multiplex_count = len(self.assignments[0])
        assert all(len(b) == self.multiplex_count for b in self.assignments)

        flat = [x for bucket in self.assignments for x in bucket]
        self.total_valid_entries = sum(x >= 0 for x in flat)
        self.total_non_padding_entries = sum(x != PADDING_NUM for x in flat)

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries

        # Precompute the slot<->object index maps for mux/demux. Gather
        # indices are used instead of permutation matrices so the operations
        # are exact copies (GPU matmul is not bit-exact).
        slot_to_obj = [max(x, PADDING_NUM) for x in flat]
        obj_to_slot = [0] * self.total_valid_entries
        for slot, obj_idx in enumerate(slot_to_obj):
            if obj_idx >= 0:
                obj_to_slot[obj_idx] = slot
        self._slot_to_obj = mx.array(slot_to_obj)
        self._slot_gather_idx = mx.maximum(self._slot_to_obj, 0)
        self._slot_is_valid = self._slot_to_obj >= 0
        self._obj_to_slot = mx.array(obj_to_slot)

    @property
    def available_slots(self) -> int:
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> List[int]:
        """Return the next consecutive object indices available in the state."""
        assert num_objects > 0
        if not allow_new_buckets:
            assert (
                self.available_slots >= num_objects
            ), f"not enough available slots {self.available_slots} < {num_objects}"
        return list(
            range(self.total_valid_entries, self.total_valid_entries + num_objects)
        )

    def add_objects(
        self,
        object_indices: List[int],
        *,
        object_ids: Optional[List[int]] = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        """Add new objects, filling empty slots first, then new buckets."""
        num_new_objects = len(object_indices)
        if num_new_objects == 0:
            return
        assert object_indices == sorted(object_indices)
        assert (object_ids is None) == (self.object_ids is None)
        if object_ids is not None:
            assert len(object_ids) == num_new_objects
        if prefer_new_buckets:
            assert allow_new_buckets

        pending = list(zip(object_indices, object_ids or [None] * num_new_objects))

        def _place(bucket, i):
            obj_idx, obj_id = pending.pop(0)
            bucket[i] = obj_idx
            if object_ids is not None:
                self.object_ids.append(obj_id)

        if not prefer_new_buckets:
            # Fill empty slots in existing buckets first
            for bucket in self.assignments:
                for i in range(self.allowed_bucket_capacity):
                    if pending and bucket[i] == PADDING_NUM:
                        _place(bucket, i)
                if not pending:
                    break

        if pending and not allow_new_buckets:
            raise ValueError(
                f"Cannot place objects {[p[0] for p in pending]} "
                "without creating new buckets"
            )

        # Create new buckets for remaining objects
        while pending:
            new_bucket = [PADDING_NUM] * self.multiplex_count
            for i in range(self.allowed_bucket_capacity):
                if not pending:
                    break
                _place(new_bucket, i)
            self.assignments.append(new_bucket)

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects

    def remove_objects(self, object_indices: List[int], strict: bool = True):
        """Mark objects as removed; drop buckets that become empty.

        Returns the list of bucket indices that are kept.
        """
        remaining = set(object_indices)
        for bucket in self.assignments:
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in remaining:
                    bucket[slot_idx] = REMOVED_NUM
                    remaining.discard(obj_id)
        if strict:
            assert not remaining, f"Failed to remove objects: {sorted(remaining)}"

        # A bucket is dead once it holds no valid (non-negative) objects
        buckets_to_keep = [
            i
            for i, bucket in enumerate(self.assignments)
            if any(obj_id >= 0 for obj_id in bucket)
        ]
        self.assignments = [self.assignments[i] for i in buckets_to_keep]

        if not buckets_to_keep:
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        # Remap remaining object indices to be sequential
        all_positive_ids = sorted(
            {x for bucket in self.assignments for x in bucket if x >= 0}
        )
        id_mapping = {old: new for new, old in enumerate(all_positive_ids)}
        for bucket in self.assignments:
            for i, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[i] = id_mapping[obj_id]

        if self.object_ids is not None:
            self.object_ids = [self.object_ids[old] for old in all_positive_ids]

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def mux(self, x: mx.array) -> mx.array:
        """(total_valid_entries, ...) -> (num_buckets, multiplex_count, ...).

        Padding slots are filled with zeros."""
        num_valid = x.shape[0]
        assert (
            num_valid == self.total_valid_entries
        ), f"{num_valid=} != {self.total_valid_entries=}"
        result = mx.take(x.reshape(num_valid, -1), self._slot_gather_idx, axis=0)
        result = mx.where(self._slot_is_valid[:, None], result, 0)
        return result.reshape(self.num_buckets, self.multiplex_count, *x.shape[1:])

    def demux(self, x: mx.array) -> mx.array:
        """(num_buckets, multiplex_count, ...) -> (total_valid_entries, ...)."""
        num_buckets, multiplex_count = x.shape[:2]
        assert num_buckets == self.num_buckets
        assert multiplex_count == self.multiplex_count
        x_flat = x.reshape(num_buckets * multiplex_count, -1)
        result = mx.take(x_flat, self._obj_to_slot, axis=0)
        return result.reshape(self.total_valid_entries, *x.shape[2:])

    def get_valid_object_mask(self) -> mx.array:
        """(num_buckets, multiplex_count) bool mask of valid (non-padding) slots."""
        return self._slot_is_valid.reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self) -> set:
        """All valid internal object indices in the state."""
        return {
            obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0
        }


class MultiplexController:
    """Creates multiplex states by bucketing objects (inference: no shuffle)."""

    def __init__(self, multiplex_count: int, eval_multiplex_count: int = -1):
        assert multiplex_count >= 1
        self.multiplex_count = multiplex_count
        self.eval_multiplex_count = (
            multiplex_count if eval_multiplex_count < 0 else eval_multiplex_count
        )

    @property
    def allowed_bucket_capacity(self) -> int:
        # inference only (eval capacity)
        return self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        random: bool = False,
        *,
        object_ids: Optional[List[int]] = None,
    ) -> MultiplexState:
        """Map `num_valid_entries` objects into buckets of size multiplex_count."""
        capacity = self.allowed_bucket_capacity
        num_buckets = math.ceil(num_valid_entries / capacity)

        ids = (
            mx.random.permutation(num_valid_entries).tolist()
            if random
            else list(range(num_valid_entries))
        )
        ids += [PADDING_NUM] * (num_buckets * capacity - len(ids))
        assignments = [
            ids[i * capacity : (i + 1) * capacity]
            + [PADDING_NUM] * (self.multiplex_count - capacity)
            for i in range(num_buckets)
        ]
        return MultiplexState(
            assignments, allowed_bucket_capacity=capacity, object_ids=object_ids
        )


class MultiplexTrackerState:
    """Inference state for a multiplex video tracking session.

    Holds the multiplex assignment (buckets) and the per-frame outputs that
    serve as memory for future frames.
    """

    def __init__(self, multiplex_state: MultiplexState):
        self.multiplex_state = multiplex_state
        self.cond_frame_outputs: Dict[int, dict] = {}
        self.non_cond_frame_outputs: Dict[int, dict] = {}

    @property
    def output_dict(self) -> dict:
        return {
            "cond_frame_outputs": self.cond_frame_outputs,
            "non_cond_frame_outputs": self.non_cond_frame_outputs,
        }

    @property
    def num_objects(self) -> int:
        return self.multiplex_state.total_valid_entries
