"""Model-agnostic coordination for Automatic Prefix Caching.

The storage manager owns hashes, blocks, eviction and persistence.  This
coordinator owns the model cache plan and is the only layer generation code
needs to call.  It follows vLLM's split between cache specs/groups and a
coordinator, adapted to MLX's contiguous runtime cache objects.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from .apc_adapters import PrefixCachePlan, build_prefix_cache_plan


class APCCoordinator:
    """Coordinate prefix reuse across every cache group in a model.

    Native dense K/V layouts use the block pool.  Windowed, recurrent,
    composite and custom layouts use restorable checkpoints at the same token
    boundary.  The distinction is deliberately private to this class.
    """

    def __init__(self, manager: Any, model: Any):
        self.manager = manager
        self.model = model
        self.plan: PrefixCachePlan = build_prefix_cache_plan(model)

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
                (
                    list(pick.pop("warm_cache"))
                    if pick is not None
                    else self.fresh_cache()
                )
                for pick in picks
            ]
            return make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
                kv_quant_config=kv_quant_config,
                consume_sources=True,
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
        from .apc import snapshot_prompt_cache_row

        snapshot = snapshot_prompt_cache_row(prompt_cache, batch_idx or 0)
        if snapshot is None:
            return False
        return self.manager.store_exact_cache(
            token_ids,
            snapshot,
            extra_hash=extra_hash,
            take_ownership=True,
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
