# Cross-request prefix reuse (APC) for the DFlash B=1 speculative path.
#
# The server's _run_speculative loop never consulted the automatic-prefix-cache
# subsystem (apc.py) — it always prefilled the whole prompt and reported
# cached_tokens=0. For agent traffic (stable system prompt + tools + growing
# conversation) that re-prefills a large shared prefix every turn.
#
# This module adds exact-mode prefix reuse for the DFlash path. At B=1 a cold
# Muse request runs on a plain per-layer cache (make_speculative_prompt_cache);
# on an APC warm hit it decodes on a BatchRotatingKVCache/BatchKVCache restored
# from the snapshot. That warm decode is correct (no miscompute — three reviews
# traced the mask/offset/rollback math) and deterministic (warm-vs-warm
# identical); it is usually byte-identical to the cold path and can occasionally
# diverge at a near-tie greedy fork on the 4-bit target (numerical tie-flip, same
# class as spec-vs-plain — see utils.py and the wiki lever record). It is a no-op
# unless an APCManager is present (APC_ENABLED=1), so with APC off the serving
# path is unchanged.
#
# Design notes:
#   * We compute our OWN extra_hash here, from the image payload only (0 for
#     text), so store and lookup always agree WITHOUT depending on the shared
#     _apc_extra_hash, which folds in inputs_embeds and is therefore unstable
#     across requests (the salt bug that makes the CB path miss). We do not
#     touch that shared path.
#   * Text-only guard: if the request carries pixel_values we skip APC
#     entirely (DFlash image handling is unsettled, and multimodal prefixes
#     need the media-safe machinery we're deliberately not invoking here).
#   * Every entry point is wrapped by the caller in try/except so an APC fault
#     degrades to a cold prefill, never a failed request.
#
# Caveat to measure, not a correctness bug: on a warm restore the target hidden
# states captured during suffix prefill cover only the suffix, so the drafter's
# cross-attention context is short for the first block — expect a one-block
# acceptance dip right after a hit. The same shortened-drafter-context caveat
# applies to COLD requests when a mid-prefill static-prefix checkpoint is
# captured — the head/tail prefill split means DFlash hidden states cover only
# the tail, so a first-block acceptance dip is expected there too (milder than
# the warm single-position case).
import logging
import os
from typing import Any, List, Optional, Sequence

from .. import apc as _apc

logger = logging.getLogger("mlx_vlm.apc")

# Store-size floor for post-prefill stores (review #2): with the in-memory LRU
# capped at APC_EXACT_CACHE_ENTRIES=2, tiny entries just churn it. At Muse's
# measured ~600 tok/s M5 prefill a 1024-token hit saves ~1.7 s, and a
# 1024-token entry is ~55 MB post-compaction — below that the entry churns the
# small LRU for sub-second savings. Deliberately NOT applied to
# store_prefix_checkpoint, which is already gated by STATIC_PREFIX_MIN_TOKENS
# (2048) via static_prefix_boundary.
STORE_MIN_TOKENS = max(0, int(os.environ.get("APC_DFLASH_STORE_MIN_TOKENS", "1024")))


def _text_only(prompt_kwargs: Optional[dict]) -> bool:
    if not prompt_kwargs:
        return True
    # The server's _gpu_embed runs the vision encoder and REPLACES pixel_values
    # with inputs_embeds, setting `_apc_image_hash` iff the request carried an
    # image (absent for pure text). So `_apc_image_hash` is the signal that
    # survives; checking only pixel_values would miss every real image request
    # and let its KV be stored/reused under a text key (Codex #1).
    if prompt_kwargs.get("_apc_image_hash") is not None:
        return False
    for k in ("pixel_values", "pixel_values_videos", "input_features"):
        if prompt_kwargs.get(k) is not None:
            return False
    return True


def _stable_extra_hash(model: Any, processor: Any) -> int:
    """Request-independent salt: image payload None (text) + model/processor
    deps. Identical for every text request against this model, so exact-prefix
    keys line up across requests."""
    return _apc.semantic_extra_hash(
        tenant=None,
        image_hash=_apc.hash_image_payload(pixel_values=None, image_ref=None),
        media={},
        model=model,
        processor=processor,
    )


def dflash_apc_mode(language_model: Any) -> Optional[str]:
    """Only exact mode is wired here (Muse uses RotatingKVCache/KVCache ->
    exact). Return None for block-only models so the caller skips APC.

    Builds ``make_cache()`` exactly ONCE and derives both the mode
    classification and the sink guard from that single list (Codex r1 #1:
    ``model_apc_mode`` plus a second probe built two throwaway per-layer cache
    lists per row per lookup — ~1.6k empty objects on a 16-row batch). The
    classification mirrors ``model_apc_mode``'s ordering: all-block-eligible
    is "block" (refused here), else all-exact-eligible is "exact".
    """
    make_cache = getattr(language_model, "make_cache", None)
    if not callable(make_cache):
        return None  # model_apc_mode calls this "block"; block is refused here
    try:
        caches = make_cache()
        if not caches:
            return None
        if all(_apc._cache_entry_supports_block_apc(c) for c in caches):
            return None  # "block" — not wired on this path
        if not all(_apc._cache_entry_supports_exact_apc(c) for c in caches):
            return None
        # keep>0 guard (review #4): BatchRotatingKVCache — used on every warm
        # restore — has no `keep` support (merge takes the last window tokens,
        # extract drops `keep`), so an attention-sink model would be silently
        # corrupted on reuse. Refuse APC outright for such models. Recursive
        # (Codex r2 #3): exact eligibility admits CacheList/tuple composites,
        # so the sink probe must look inside them too. Any fault while probing
        # degrades to None (cold path), never a failed request.
        for c in caches:
            if _has_sink(c):
                return None
    except Exception:
        return None
    return "exact"


def _has_sink(c) -> bool:
    """True if ``c`` or any nested sub-cache carries an attention sink
    (keep > 0). Exact-mode eligibility recurses through CacheList/tuple
    composites, so this check must recurse too (Codex r2 #3) — a composite
    hiding a RotatingKVCache(keep>0) would otherwise slip past the flat
    guard and be corrupted by the keep-blind batch merge/extract."""
    if c is None:
        return False
    if getattr(c, "keep", 0):
        return True
    subs = getattr(c, "caches", None)
    if subs is not None:
        return any(_has_sink(s) for s in subs)
    if isinstance(c, tuple):
        return any(_has_sink(s) for s in c)
    return False


def _snapshot_reuse_safe(snap) -> bool:
    """False if any cache entry (at any nesting depth) carries an attention
    sink (keep > 0). Batch merge/extract are keep-blind (review #4), so sink
    caches must never enter the exact store — a warm restore through
    BatchRotatingKVCache would silently drop or misplace the sink tokens."""
    return not any(_has_sink(c) for c in snap)


def lookup(
    apc_manager: Any,
    language_model: Any,
    model: Any,
    processor: Any,
    ids_list: Sequence[int],
    prompt_kwargs: Optional[dict],
) -> Optional[dict]:
    """Return an APC plan ({'warm_cache', 'prefix_len', 'extra_hash'}) for a
    reusable exact prefix, or None. Text-only, exact-mode, B=1.

    Cheap request-local guards run BEFORE the mode probe (Codex r2 #4):
    probing calls ``language_model.make_cache()``, which a disabled-APC or
    media-bearing request should never trigger.
    """
    if apc_manager is None or not _text_only(prompt_kwargs):
        return None
    if not ids_list or len(ids_list) < 2:
        return None
    if dflash_apc_mode(language_model) != "exact":
        return None
    return _lookup_row(apc_manager, model, processor, ids_list, prompt_kwargs)


def _lookup_row(
    apc_manager: Any,
    model: Any,
    processor: Any,
    ids_list: Sequence[int],
    prompt_kwargs: Optional[dict],
) -> Optional[dict]:
    """``lookup`` minus the mode gate — for callers that have already checked
    ``dflash_apc_mode`` once for the whole batch (Codex r1 #1: the per-row
    gate rebuilt the model's cache list B times per coalesced lookup)."""
    if apc_manager is None or not _text_only(prompt_kwargs):
        return None
    if not ids_list or len(ids_list) < 2:
        return None
    extra_hash = _stable_extra_hash(model, processor)
    plan = _apc.apc_lookup_plan(
        apc_manager,
        list(ids_list),
        extra_hash=extra_hash,
        apc_mode="exact",
        safe_lookup_min=0,           # text-only: no media-safe floor
        suffix_is_text_only=lambda pl: True,
        prefix_has_media=lambda pl: False,
    )
    if plan is not None:
        plan["extra_hash"] = extra_hash
    return plan


def store(
    apc_manager: Any,
    model: Any,
    processor: Any,
    ids_list: Sequence[int],
    prompt_cache: Sequence[Any],
    extra_hash: Optional[int] = None,
    prompt_kwargs: Optional[dict] = None,
) -> bool:
    """Snapshot the post-prefill cache for exact reuse.

    Stores two entries:
      * full length L — reused when this whole prompt is a literal prefix of a
        later (strictly longer) request, i.e. a growing multi-turn conversation.
      * checkpoint L-guard — a copy trimmed back by the manager's guard tokens.
        lookup_exact_cache rejects an entry whose length equals the query
        length (no suffix to generate from), and the generation-prompt marker
        makes a full prior prompt rarely a literal prefix of the next one; the
        guard checkpoint is what lets an identical / small-tail-variation
        re-request hit. Trimming a clone is exact for KVCache/RotatingKVCache
        (guard << window), and avoids a second prefill pass.

    A static-prefix checkpoint is pinned at static_prefix_boundary (end of the
    FIRST rendered message, min STATIC_PREFIX_MIN_TOKENS) when one exists — NOT
    at the first user-turn marker, which sits after the context gateway's
    per-request control packet and therefore never repeats (see
    static_prefix_boundary).
    """
    if apc_manager is None or not ids_list:
        return False
    # Symmetric with lookup(): never snapshot a media-bearing prompt under a
    # text-ignoring key. Dead today (only store_batch is wired), but keeps this
    # B=1 entry point from becoming a media-KV footgun for a future caller
    # (Fable verify #1a).
    if prompt_kwargs is not None and not _text_only(prompt_kwargs):
        return False
    if extra_hash is None:
        extra_hash = _stable_extra_hash(model, processor)
    boundary = static_prefix_boundary(processor, ids_list)
    return _store_row(
        apc_manager, ids_list, prompt_cache, 0, extra_hash,
        boundaries=[boundary] if boundary else None,
    )


def store_prefix_checkpoint(
    apc_manager: Any,
    model: Any,
    processor: Any,
    ids_list: Sequence[int],
    prompt_cache: Sequence[Any],
    cp: int,
    extra_hash: Optional[int] = None,
    row_idx: int = 0,
    prompt_kwargs: Optional[dict] = None,
) -> bool:
    """Store a static-prefix checkpoint captured DURING prefill, at length ``cp``.

    _store_trimmed builds checkpoints after the fact by trimming the finished
    cache, which it must refuse once a RotatingKVCache has wrapped -- the
    rotated-out tokens are gone and the trim would key an incomplete window as
    tokens[:cp]. Muse's sliding layers are 2048 wide, so on any real prompt
    every checkpoint is refused and only the full-length entry survives; since
    lookup rejects an entry whose length equals the query length, an identical
    or same-system re-request can never hit.

    Captured mid-prefill there is nothing to recover: at the moment the cache
    offset IS cp, the sliding layers physically hold the true window ending at
    cp -- the same state a cold prefill of tokens[:cp] leaves behind. No trim,
    so the wrap guard does not apply.

    Fail-safe: any fault returns False. A missed checkpoint costs reuse, never
    correctness.
    """
    if apc_manager is None or not ids_list:
        return False
    if prompt_kwargs is not None and not _text_only(prompt_kwargs):
        return False  # symmetric with lookup()/store(): never key media under text
    n = len(ids_list)
    cp = int(cp)
    if not 1 <= cp < n:
        return False
    if extra_hash is None:
        extra_hash = _stable_extra_hash(model, processor)
    snap = _apc.snapshot_prompt_cache_row(prompt_cache, row_idx)
    if snap is None:
        return False
    # Batch merge/extract are keep-blind (review #4): sink caches must never
    # enter the exact store, or a warm restore would silently corrupt them.
    if not _snapshot_reuse_safe(snap):
        return False
    # Snapshot is fresh and never reused by us -> manager may own it without
    # recloning (review #5).
    return bool(
        apc_manager.store_exact_cache(
            [int(t) for t in ids_list[:cp]], snap, extra_hash=extra_hash, clone=False
        )
    )


# --- static-prefix (system+tools) boundary detection ------------------------
# Fallback ONLY (review #6): keyed by id(tok), so after GC a recycled id could
# serve a stale header id and the dict grows unbounded. Used solely for
# tokenizers that reject attribute writes; the primary cache lives on the
# tokenizer object itself and dies with it.
_START_HEADER_CACHE: dict = {}
_UNSET = object()  # distinguishes "not computed" from a computed None
# Below this the checkpoint is not worth ~1.5 GB of cache for a 30B: the prefill
# it saves is a rounding error, and storing it would churn the disk tier.
STATIC_PREFIX_MIN_TOKENS = 2048


def _start_header_id(processor) -> Optional[int]:
    """Token id of the harmony "<|start|>" message header, if it is a single id."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    cached = getattr(tok, "_apc_start_header_id", _UNSET)
    if cached is not _UNSET:
        return cached
    key = id(tok)
    if key in _START_HEADER_CACHE:  # fallback hit: attribute-rejecting tokenizer
        return _START_HEADER_CACHE[key]
    out = None
    try:
        ids = list(tok.encode("<|start|>"))
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        if len(ids) == 1:
            out = int(ids[0])
    except Exception:
        out = None
    try:
        setattr(tok, "_apc_start_header_id", out)
    except Exception:
        # Some tokenizers (__slots__/frozen wrappers) reject attribute writes;
        # only those fall back to the id()-keyed module dict.
        _START_HEADER_CACHE[key] = out
    return out


def static_prefix_boundary(
    processor, ids_list: Sequence[int], min_tokens: int = STATIC_PREFIX_MIN_TOKENS
) -> Optional[int]:
    """End of the FIRST rendered message -- the reusable static prefix.

    Deliberately not the first user-turn marker. Anything sitting between the
    system prompt and the user turn lands inside that prefix, and the context
    gateway puts a per-request control packet exactly there (as a developer
    message, which _qwen_compatible_body rewrites to system, so it is inside
    the system block). That packet embeds the request, so a user-marker
    boundary changes on every single call and never hits -- while still costing
    a full checkpoint store each time.

    The first message is the big stable one (the agent's system prompt), so
    cutting at the second "<|start|>" header keeps the prefix identical across
    both a varying gateway packet and a varying question. Verified: two
    renders differing in packet AND question produced byte-identical prefixes.

    Returns None when the model is not harmony-framed, when there is only one
    message, or when the prefix is shorter than ``min_tokens``.
    """
    start_id = _start_header_id(processor)
    if start_id is None:
        return None
    seen = 0
    for i, tid in enumerate(ids_list):
        if tid == start_id:
            seen += 1
            if seen == 2:
                return i if i >= int(min_tokens) else None
    return None


def _ckpt_key(ids_list, cp, extra_hash):
    # Exact tuple, not hash(), so dedup never suffers a hash collision that would
    # silently drop a distinct checkpoint (Codex #5). The set is per-store_batch
    # call and holds at most a few short-lived entries per row.
    return (int(extra_hash), int(cp), tuple(int(t) for t in ids_list[:cp]))


def _store_row(
    apc_manager, ids_list, prompt_cache, row_idx, extra_hash, boundaries=None, seen=None
) -> bool:
    """Store one row of a (batched or single) prompt cache: the full length,
    a guard-checkpoint at L-guard, and any pinned boundary checkpoints (e.g. the
    static system+tools boundary). Each checkpoint is a clone trimmed to that
    length. ``seen`` dedups identical checkpoints across rows in one store_batch
    (rows sharing a system prefix produce an identical boundary checkpoint —
    storing it once avoids double-counting disk bytes; Codex #5). A key is marked
    seen ONLY after a successful store, so a row that skips a checkpoint (e.g. a
    wrapped rotating cache) does not suppress a later row that could store it.

    Checkpoints are stored FIRST and the full-length entry LAST (review #1):
    the manager's in-memory LRU holds APC_EXACT_CACHE_ENTRIES=2 and evicts
    oldest-inserted, so full-then-checkpoints made a row evict its OWN full
    entry — the one a growing conversation needs next turn."""
    n = len(ids_list)
    # Store-size floor (review #2, rationale at STORE_MIN_TOKENS): tiny entries
    # churn the 2-entry LRU for sub-second prefill savings.
    if n < STORE_MIN_TOKENS:
        return False
    # Batch merge/extract are keep-blind (review #4): sink caches must never
    # enter the exact store, or a warm restore through BatchRotatingKVCache
    # would silently corrupt them.
    if not _snapshot_reuse_safe(prompt_cache):
        return False
    # Collect checkpoint lengths: guard + pinned boundaries, deduped, valid.
    cps = set()
    guard = int(getattr(apc_manager, "exact_cache_guard_tokens", 0) or 0)
    if guard > 0:
        cps.add(n - guard)
    for b in boundaries or []:
        if b:
            cps.add(int(b))
    for cp in sorted(c for c in cps if 1 <= c < n):
        k = _ckpt_key(ids_list, cp, extra_hash)
        if seen is not None and k in seen:
            continue
        stored = _store_trimmed(
            apc_manager, ids_list, prompt_cache, row_idx, extra_hash, cp, n
        )
        if stored and seen is not None:
            seen.add(k)
    ok = False
    fk = _ckpt_key(ids_list, n, extra_hash)
    if seen is None or fk not in seen:
        snap_full = _apc.snapshot_prompt_cache_row(prompt_cache, row_idx)
        if snap_full is not None:
            # Snapshot is fresh and never reused by us -> manager may own it
            # without recloning (review #5).
            # Memory tier only. A full-length entry is hit exactly when this
            # whole prompt is a literal prefix of a later, longer one -- i.e.
            # the next turn of THIS conversation, in this process. It is never
            # hit by a new session, whose prompt diverges before the end (a
            # fresh question, and a fresh gateway control packet). Persisting
            # it wrote ~870 MB per session that nothing could ever read, and
            # that pressure evicts the static-prefix checkpoint, which every
            # new session does hit. The checkpoint still goes to disk.
            ok = apc_manager.store_exact_cache(
                list(ids_list), snap_full, extra_hash=extra_hash, clone=False,
                disk=False,
            )
            if ok and seen is not None:
                seen.add(fk)
    return ok


def _store_trimmed(apc_manager, ids_list, prompt_cache, row_idx, extra_hash, cp, n) -> bool:
    """Store a checkpoint at length ``cp`` (a snapshot trimmed by n-cp). Returns
    True iff the checkpoint was actually stored (False if skipped due to a
    wrapped rotating cache or an untrimmable cache). Trimming is exact for
    KVCache/RotatingKVCache when the cache has not wrapped."""
    snap = _apc.snapshot_prompt_cache_row(prompt_cache, row_idx)
    if snap is None:
        return False
    trim_n = n - cp
    if trim_n <= 0:
        return False
    # Correctness guard (Codex #2): a trimmed checkpoint at cp is only lossless
    # if every sliding layer still physically holds the full window ending at cp.
    # Once a RotatingKVCache has WRAPPED (offset > max_size at length n), a trim
    # cannot recover the rotated-out tokens — it would silently store an
    # incomplete window keyed as tokens[:cp] (e.g. a far-back system boundary in
    # a 32K multi-turn prompt). Skip the checkpoint in that case; the full-length
    # store (untrimmed, window-correct at n) still handles growing conversations.
    # (Deliberately conservative: an oversized one-shot prefill that still
    # retains all tokens is skipped too — a missed checkpoint, never a wrong one.)
    for c in snap:
        if c is None:
            continue
        max_size = getattr(c, "max_size", None)
        if max_size is not None:
            off = getattr(c, "offset", 0)
            off = int(off) if not hasattr(off, "shape") else int(off.max())
            if off > int(max_size):  # wrapped -> checkpoint would be corrupt
                return False
    for c in snap:
        if c is None:
            continue
        trim = getattr(c, "trim", None)
        if trim is None:
            return False  # can't trim this cache type; skip the checkpoint
        trim(trim_n)
    # Return the manager's actual result: if the store is rejected (e.g. an
    # unclonable cache, or APC disabled), the caller must NOT mark this key
    # `seen`, or it would suppress a later identical row that could store it
    # (Fable verify #5).
    # Snapshot is fresh and never reused by us -> manager may own it without
    # recloning (review #5).
    return bool(
        apc_manager.store_exact_cache(
            list(ids_list[:cp]), snap, extra_hash=extra_hash, clone=False
        )
    )


def lookup_batch(
    apc_manager: Any,
    language_model: Any,
    model: Any,
    processor: Any,
    all_input_ids: Sequence[Sequence[int]],
    prompt_kwargs_list: Sequence[Optional[dict]],
) -> Optional[dict]:
    """Per-row exact-prefix lookup, merged into a batched warm cache.

    Returns {'warm_cache' (batched, per-row offset=prefix_len), 'prefix_lens',
    'extra_hashes', 'right_pad', 'lengths', 'suffixes'} or None if no row hits.
    Each hit row contributes its exact-prefix cache; miss rows get an empty
    cache. make_warm_batch_exact_cache_multi aligns them into batch caches.
    """
    if apc_manager is None:
        return None
    # One mode/sink probe for the whole batch (Codex r1 #1) — the per-row
    # lookup below skips it via _lookup_row.
    if dflash_apc_mode(language_model) != "exact":
        return None
    B = len(all_input_ids)
    picks = []
    extra_hashes = []
    for i in range(B):
        pk = prompt_kwargs_list[i] if i < len(prompt_kwargs_list) else None
        if not _text_only(pk):
            picks.append(None)
            extra_hashes.append(_stable_extra_hash(model, processor))
            continue
        plan = _lookup_row(apc_manager, model, processor, all_input_ids[i], pk)
        picks.append(plan)
        extra_hashes.append(
            plan.get("extra_hash") if plan else _stable_extra_hash(model, processor)
        )
    prefix_lens = []
    for i in range(B):
        p = picks[i]
        pl = int(p["prefix_len"]) if (p and p.get("warm_cache") is not None) else 0
        if pl >= len(all_input_ids[i]):  # need >=1 suffix token
            pl = 0
            picks[i] = None
        prefix_lens.append(pl)
    if not any(pl > 0 for pl in prefix_lens):
        return None
    row_caches = [
        picks[i]["warm_cache"] if (picks[i] and picks[i].get("warm_cache") is not None)
        else language_model.make_cache()
        for i in range(B)
    ]
    merged, _ = _apc.make_warm_batch_exact_cache_multi(row_caches, prefix_lens)
    if merged is None:
        return None
    suffixes = [list(all_input_ids[i][prefix_lens[i]:]) for i in range(B)]
    lengths = [len(s) for s in suffixes]
    max_suf = max(lengths)
    right_pad = [max_suf - l for l in lengths]
    return {
        "warm_cache": merged,
        "prefix_lens": prefix_lens,
        "extra_hashes": extra_hashes,
        "right_pad": right_pad,
        "lengths": lengths,
        "suffixes": suffixes,
    }


def store_batch(
    apc_manager: Any,
    model: Any,
    processor: Any,
    all_input_ids: Sequence[Sequence[int]],
    prompt_cache: Sequence[Any],
    extra_hashes: Sequence[Optional[int]],
    prompt_kwargs_list: Optional[Sequence[Optional[dict]]] = None,
) -> None:
    """Store each row of the post-prefill batched cache for future reuse.

    Skips non-text-only rows (image/audio/video): the extra_hash intentionally
    ignores media, so storing a media-bearing prompt under a text key would let a
    later text request with matching token ids reuse media-derived KV (Codex #1).
    Boundary checkpoints are pinned at static_prefix_boundary (end of the FIRST
    rendered message, min STATIC_PREFIX_MIN_TOKENS) — not the first user-turn
    marker, which sits after the context gateway's per-request control packet
    and therefore never repeats (see static_prefix_boundary).
    ``seen`` dedups identical checkpoints across rows (Codex #5)."""
    if apc_manager is None:
        return
    seen: set = set()
    for i in range(len(all_input_ids)):
        try:
            pk = (
                prompt_kwargs_list[i]
                if prompt_kwargs_list and i < len(prompt_kwargs_list)
                else None
            )
            if not _text_only(pk):
                continue  # never store media-bearing prompts under a text key
            eh = (
                extra_hashes[i]
                if extra_hashes and i < len(extra_hashes) and extra_hashes[i] is not None
                else _stable_extra_hash(model, processor)
            )
            boundary = static_prefix_boundary(processor, all_input_ids[i])
            _store_row(
                apc_manager, all_input_ids[i], prompt_cache, i, eh,
                boundaries=[boundary] if boundary else None, seen=seen,
            )
        except Exception as e:
            # Non-fatal: a failed store just means missed future reuse, never a
            # failed request. But it must not be silent — a store regression
            # (e.g. an unclonable cache) would otherwise be invisible in
            # production (Fable F5). debug-level to avoid hot-path spam.
            logger.debug("APC store_batch row %d failed: %s", i, e, exc_info=True)
