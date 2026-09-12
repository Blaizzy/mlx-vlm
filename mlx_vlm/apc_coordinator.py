"""Model-agnostic coordination for Automatic Prefix Caching.

The storage manager owns hashes, blocks, eviction and persistence.  This
coordinator owns the model cache plan and is the only layer generation code
needs to call.  It follows vLLM's split between cache specs/groups and a
coordinator, adapted to MLX's contiguous runtime cache objects.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from .apc_adapters import (
    PrefixCachePlan,
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
)


class APCCoordinator:
    """Coordinate prefix reuse across every cache group in a model.

    Native dense K/V layouts use the block pool.  Windowed, recurrent,
    composite and custom layouts use restorable checkpoints at the same token
    boundary.  The distinction is deliberately private to this class.
    """

    def __init__(self, manager: Any, model: Any, *, max_kv_size: Optional[int] = None):
        self.manager = manager
        self.model = model
        self.max_kv_size = max_kv_size
        self.plan: PrefixCachePlan = (
            build_prefix_cache_plan(model)
            if max_kv_size is None
            else build_prefix_cache_plan_from_caches(self.fresh_cache())
        )

    def scope_hash(self, extra_hash: int) -> int:
        if self.max_kv_size is None:
            return extra_hash
        from .apc import semantic_extra_hash

        return semantic_extra_hash(
            image_hash=extra_hash, media={"max_kv_size": self.max_kv_size}
        )

    def prepare_prefill(self, token_count: int) -> None:
        if self.enabled:
            self.manager.prepare_prefill(token_count)

    @property
    def enabled(self) -> bool:
        return self.manager is not None and self.plan.restorable

    @property
    def strategy(self) -> Optional[str]:
        return self.plan.strategy if self.enabled else None

    @property
    def is_checkpoint(self) -> bool:
        return self.strategy == "checkpoint"

    @property
    def legacy_mode(self) -> Optional[str]:
        return self.plan.legacy_mode if self.enabled else None

    def fresh_cache(self) -> List[Any]:
        language_model = getattr(self.model, "language_model", self.model)
        if self.max_kv_size is not None:
            from .models.cache import make_prompt_cache

            return list(make_prompt_cache(language_model, max_kv_size=self.max_kv_size))
        make_cache = getattr(language_model, "make_cache", None) or getattr(
            self.model, "make_cache", None
        )
        if callable(make_cache):
            return list(make_cache())
        from .models.cache import make_prompt_cache

        return list(make_prompt_cache(language_model))

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int,
        safe_lookup_min: int,
        suffix_is_text_only: Callable[[int], bool],
        prefix_has_media: Callable[[int], bool],
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        from .apc import apc_lookup_plan

        hit = apc_lookup_plan(
            self.manager,
            token_ids,
            extra_hash=extra_hash,
            apc_mode=self.legacy_mode,
            safe_lookup_min=safe_lookup_min,
            suffix_is_text_only=suffix_is_text_only,
            prefix_has_media=prefix_has_media,
        )
        if hit is not None:
            hit["cache_plan"] = self.plan
        return hit

    def checkpoint_len(
        self, token_ids: Sequence[int], media_token_ids: set[int]
    ) -> int:
        """Reusable checkpoint before the final guard token(s)."""
        if not self.enabled or not self.is_checkpoint:
            return 0
        from .apc import adjust_prefix_to_text_suffix_boundary

        return adjust_prefix_to_text_suffix_boundary(
            token_ids,
            len(token_ids) - self.manager.exact_cache_guard_tokens,
            media_token_ids,
            max_prefix_tokens=len(token_ids) - 1,
        )

    def checkpoint_lengths(
        self, token_ids: Sequence[int], media_token_ids: set[int]
    ) -> List[int]:
        """Bounded intermediate states plus the final conversation checkpoint.

        Stateful caches cannot roll back the final snapshot to a divergence.
        Capture earlier states while prefilling, aligned across requests. Limit
        captures to the resident entry budget (two for a disk-only manager),
        rather than copying an ever-growing cache at every prefill chunk.
        """
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        interval = self.manager.checkpoint_interval_tokens
        budget = self.manager._exact_cache_max or (2 if self.manager.disk else 1)
        if interval <= 0 or budget <= 1:
            return [final]
        from .apc import adjust_prefix_to_text_suffix_boundary

        block_size = self.manager.block_size
        interval = ((interval + block_size - 1) // block_size) * block_size
        last = ((final - 1) // interval) * interval
        first = max(interval, last - (budget - 2) * interval)
        lengths = {final}
        for boundary in range(first, last + 1, interval):
            boundary = adjust_prefix_to_text_suffix_boundary(
                token_ids, boundary, media_token_ids, max_prefix_tokens=final
            )
            if self.manager.exact_cache_min_tokens <= boundary < final:
                lengths.add(boundary)
        return sorted(lengths)

    def merge_rows(
        self,
        picks: Sequence[Optional[dict]],
        prefix_lens: Sequence[int],
        *,
        kv_quant_config: Optional[dict] = None,
    ) -> Tuple[Optional[List[Any]], int]:
        """Materialize a mixed warm/cold batch at the common prefix boundary."""
        from .apc import (
            make_warm_batch_exact_cache_multi,
            make_warm_batch_kv_cache_multi,
        )

        if self.is_checkpoint:
            row_caches = [
                pick["warm_cache"] if pick is not None else self.fresh_cache()
                for pick in picks
            ]
            return make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
                kv_quant_config=kv_quant_config,
            )
        return make_warm_batch_kv_cache_multi(
            list(picks),
            num_layers=len(self.plan.components),
            kv_quant_config=kv_quant_config,
        )

    def materialize_single(
        self,
        hit: dict,
        *,
        min_capacity_tokens: int,
        kv_quant_config: Optional[dict] = None,
    ) -> List[Any]:
        warm_cache = hit.get("warm_cache")
        if warm_cache is not None:
            return warm_cache
        from .apc import make_warm_kv_cache

        return make_warm_kv_cache(
            hit.get("matched_blocks", []),
            min_capacity_tokens=min_capacity_tokens,
            kv_quant_config=kv_quant_config,
        )

    def store_checkpoint(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        batch_idx: Optional[int] = None,
    ) -> bool:
        if not self.enabled or not self.is_checkpoint:
            return False
        from .apc import (
            _cache_nbytes,
            _prompt_cache_is_batch_shaped,
            snapshot_prompt_cache_row,
        )

        # Batch extraction can itself allocate a full row before store_exact_cache
        # decides whether it can afford another retained snapshot.
        if _prompt_cache_is_batch_shaped(prompt_cache):
            if self.manager.disk is not None:
                self.manager.disk.flush()
            if not self.manager._make_room(_cache_nbytes(prompt_cache)):
                with self.manager.lock:
                    self.manager.stats.memory_skips += 1
                return False
        snapshot = snapshot_prompt_cache_row(prompt_cache, batch_idx or 0, clone=False)
        if snapshot is None:
            return False
        return self.manager.store_exact_cache(
            token_ids, snapshot, extra_hash=extra_hash
        )

    def commit(
        self,
        prompt_cache: Sequence[Any],
        token_ids: Sequence[int],
        *,
        batch_idx: Optional[int] = None,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
        blocks_in_use: Sequence[Any] = (),
    ) -> bool:
        """Store one completed prefix and release any block leases."""
        if not self.enabled:
            return False
        if self.is_checkpoint:
            try:
                return self.store_checkpoint(
                    token_ids,
                    prompt_cache,
                    batch_idx=batch_idx,
                    extra_hash=extra_hash,
                )
            finally:
                self.manager.release(blocks_in_use)

        from .apc import commit_prefix_blocks

        commit_prefix_blocks(
            self.manager,
            list(prompt_cache),
            token_ids,
            batch_idx=batch_idx,
            extra_hash=extra_hash,
            skip_first_n_tokens=skip_first_n_tokens,
            blocks_in_use=blocks_in_use,
        )
        return True

    def release_hit(self, hit: Optional[dict]) -> None:
        if hit is not None and self.manager is not None:
            self.manager.release(hit.get("matched_blocks", ()))
