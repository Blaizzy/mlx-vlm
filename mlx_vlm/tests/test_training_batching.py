"""Shared batch index ordering and distributed coverage."""

import itertools
import unittest

from mlx_vlm.trainer.common.utils import iterate_batch_indices


class BatchingTest(unittest.TestCase):
    def test_distributed_coverage_and_seed(self):
        streams = [
            list(
                itertools.islice(
                    iterate_batch_indices(
                        13,
                        4,
                        length_key=lambda i: i,
                        train=True,
                        rank=rank,
                        world_size=2,
                        seed=42,
                    ),
                    6,
                )
            )
            for rank in range(2)
        ]
        for epoch in range(2):
            seen = []
            for step in range(epoch * 3, (epoch + 1) * 3):
                left, right = streams[0][step], streams[1][step]
                self.assertFalse(set(left) & set(right))
                self.assertEqual(left[-1] + 1, right[0])
                seen.extend(left + right)
            self.assertEqual(sorted(seen), list(range(12)))
        repeated = list(
            itertools.islice(
                iterate_batch_indices(
                    13,
                    4,
                    length_key=lambda i: i,
                    train=True,
                    rank=0,
                    world_size=2,
                    seed=42,
                ),
                6,
            )
        )
        self.assertEqual(streams[0], repeated)

    def test_sorted_finite_evaluation(self):
        indices = list(iterate_batch_indices(5, 2, length_key=lambda i: -i))
        self.assertEqual(indices, [[4, 3], [2, 1]])
        for kwargs in (
            {"batch_size": 0},
            {"batch_size": 8},
            {"batch_size": 3, "world_size": 2},
            {"batch_size": 2, "rank": 1},
        ):
            with self.assertRaises(ValueError):
                list(iterate_batch_indices(5, length_key=lambda i: i, **kwargs))


if __name__ == "__main__":
    unittest.main()
