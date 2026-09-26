"""Shared semantic prefix lookup and checkpoint boundaries."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class PrefixContext:
    token_ids: list[int]
    spans: tuple[tuple[int, int], ...]
    hashes: list[int]

    @staticmethod
    def supports_overrides(kwargs, input_ids, mask):
        # Opaque embeddings, positions and mutable caches cannot be identified
        # by the processed media tensors. Never store them in this namespace.
        if any(
            kwargs.get(key) is not None
            for key in (
                "position_ids",
                "rope_deltas",
                "cached_image_features",
                "inputs_embeds",
                "prompt_cache",
                "draft_model",
                "max_kv_size",
                "kv_bits",
                "kv_key_bits",
                "kv_value_bits",
            )
        ):
            return False
        return mask is None or (
            mask.ndim == 2
            and mask.shape == input_ids.shape
            and bool(mx.all(mask == 1).item())
        )

    def prefix_hash(self, prefix_len: int) -> int:
        completed = 0
        for start, end in self.spans:
            if start < prefix_len < end:
                raise ValueError("A media prefix cannot end inside a media span")
            completed += end <= prefix_len
        return self.hashes[completed]

    def lookup(self, manager) -> Optional[dict]:
        # The manager's lower bound is exclusive and its upper bound inclusive.
        # In particular max_prefix_tokens=0 means unbounded, not an empty range.
        for k in range(len(self.spans), -1, -1):
            lower = self.spans[k - 1][1] - 1 if k else 0
            upper = self.spans[k][0] if k < len(self.spans) else len(self.token_ids) - 1
            if upper <= lower or upper <= 0:
                continue
            cache, length = manager.lookup_exact_cache(
                self.token_ids,
                extra_hash=self.hashes[k],
                min_prefix_tokens=lower,
                max_prefix_tokens=upper,
            )
            if cache is not None:
                return {
                    "warm_cache": cache,
                    "prefix_len": length,
                    "matched_blocks": [],
                    "extra_hash": self.hashes[k],
                    "full_input_ids": self.token_ids,
                }
        return None

    def checkpoint_lengths(self, coordinator) -> list[int]:
        # Ordinary checkpoint spacing still applies, but a boundary only needs
        # to pass the media occurrence containing it, not every occurrence.
        lengths = coordinator.checkpoint_lengths(self.token_ids, set())
        safe = set()
        for length in lengths:
            for start, end in self.spans:
                if start < length < end:
                    length = end
                    break
            if 0 < length < len(self.token_ids):
                safe.add(length)
        return sorted(safe)
