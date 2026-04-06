"""Pre-defined spatial operations on mask metadata for the grounded reasoning agent.

These deterministic functions replace numerical reasoning that would otherwise
require the VLM to sort floats in natural language — an unreliable operation for
small quantised models.  The VLM picks the right function name and parameters;
the function handles all arithmetic and returns a clean JSON-serialisable result.

All functions take ``all_masks`` as their first argument — the flat dict of
{mask_id: meta} maintained by the agent loop — plus any per-tool parameters.
"""

import math


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slot_masks(all_masks, slot):
    """Return a list of mask metadata dicts for the given slot, or raise."""
    ms = [m for m in all_masks.values() if m.get("slot") == slot]
    if not ms:
        available = sorted({m.get("slot") for m in all_masks.values()})
        raise KeyError(
            f"Slot '{slot}' has no active masks. "
            f"Available slots: {available}"
        )
    return ms


def _sort_key(direction):
    """Return a (key_fn, reverse) pair for sorting by direction keyword."""
    d = direction.lower()
    if d in ("left", "leftmost"):
        return lambda m: m["centroid_norm"]["x"], False
    if d in ("right", "rightmost"):
        return lambda m: m["centroid_norm"]["x"], True
    if d in ("top", "topmost"):
        return lambda m: m["centroid_norm"]["y"], False
    if d in ("bottom", "bottommost"):
        return lambda m: m["centroid_norm"]["y"], True
    raise ValueError(
        f"Unknown direction '{direction}'. "
        "Use: left, right, top, bottom (or leftmost/rightmost/topmost/bottommost)."
    )


def _summary(m):
    """Compact, JSON-serialisable representation of one mask."""
    return {
        "id": m["id"],
        "slot": m.get("slot"),
        "centroid_norm": m["centroid_norm"],
        "area_fraction": m["area_fraction"],
        "image_region": m["image_region"],
    }


# ---------------------------------------------------------------------------
# Ranking / ordering
# ---------------------------------------------------------------------------

def rank_by_x(all_masks, slot):
    """Rank all masks in *slot* by x-coordinate, left to right.

    Returns a list sorted left→right with an added ``rank`` field
    (rank 1 = leftmost).  Use for ordering questions along the horizontal axis
    (e.g. "which player is furthest left?", offside line comparisons).
    """
    ms = _slot_masks(all_masks, slot)
    ms_sorted = sorted(ms, key=lambda m: m["centroid_norm"]["x"])
    ranked = []
    for rank, m in enumerate(ms_sorted, start=1):
        entry = _summary(m)
        entry["rank"] = rank
        ranked.append(entry)
    return {"slot": slot, "axis": "x", "order": "left_to_right", "masks": ranked}


def rank_by_y(all_masks, slot):
    """Rank all masks in *slot* by y-coordinate, top to bottom.

    Returns a list sorted top→bottom with an added ``rank`` field
    (rank 1 = topmost / highest in the image, since y=0 is the top).
    Use for "which is highest/lowest?" type questions.
    """
    ms = _slot_masks(all_masks, slot)
    ms_sorted = sorted(ms, key=lambda m: m["centroid_norm"]["y"])
    ranked = []
    for rank, m in enumerate(ms_sorted, start=1):
        entry = _summary(m)
        entry["rank"] = rank
        ranked.append(entry)
    return {"slot": slot, "axis": "y", "order": "top_to_bottom", "masks": ranked}


# ---------------------------------------------------------------------------
# Single-mask extremes
# ---------------------------------------------------------------------------

def extreme_mask(all_masks, slot, direction):
    """Return the single mask at the given positional extreme.

    direction: ``'leftmost'`` | ``'rightmost'`` | ``'topmost'`` | ``'bottommost'``

    Use when you need exactly one mask (e.g. "the furthest-forward attacker").
    """
    ms = _slot_masks(all_masks, slot)
    key_fn, reverse = _sort_key(direction)
    best = sorted(ms, key=key_fn, reverse=reverse)[0]
    result = _summary(best)
    result["direction"] = direction
    return result


def nth_from(all_masks, slot, n, direction):
    """Return the n-th mask in *slot* sorted by *direction* (1-indexed).

    direction: ``'left'`` | ``'right'`` | ``'top'`` | ``'bottom'``

    n=1 is equivalent to extreme_mask.  n=2 gives the second from the extreme,
    etc.  Typical use: ``nth_from(slot="blue", n=2, direction="right")`` returns
    the second-rightmost blue player — i.e. the last *outfield* defender when
    the goalkeeper is the rightmost mask.
    """
    ms = _slot_masks(all_masks, slot)
    key_fn, reverse = _sort_key(direction)
    ms_sorted = sorted(ms, key=key_fn, reverse=reverse)
    if n < 1 or n > len(ms_sorted):
        raise IndexError(
            f"n={n} is out of range for slot '{slot}' which has "
            f"{len(ms_sorted)} mask(s)."
        )
    result = _summary(ms_sorted[n - 1])
    result["n"] = n
    result["direction"] = direction
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def exclude_extremes(all_masks, slot, axis="x", n=1):
    """Remove the *n* masks from each end along *axis*, return the rest.

    Typical use: remove goalkeepers from a player slot.  The GK is always the
    mask at the extreme end of the x-axis on their defensive side, so
    ``exclude_extremes(slot="blue", axis="x", n=1)`` drops the single most
    extreme-x blue mask (the GK) and returns the outfield defenders.

    Returns a dict with ``remaining`` masks and the IDs that were excluded.
    """
    ms = _slot_masks(all_masks, slot)
    if axis == "x":
        ms_sorted = sorted(ms, key=lambda m: m["centroid_norm"]["x"])
    elif axis == "y":
        ms_sorted = sorted(ms, key=lambda m: m["centroid_norm"]["y"])
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'.")

    if 2 * n >= len(ms_sorted):
        raise ValueError(
            f"Cannot exclude {n} from each end of slot '{slot}': "
            f"only {len(ms_sorted)} mask(s) available."
        )

    excluded = ms_sorted[:n] + ms_sorted[len(ms_sorted) - n:]
    kept = ms_sorted[n: len(ms_sorted) - n]
    return {
        "slot": slot,
        "axis": axis,
        "n_excluded_per_end": n,
        "excluded_ids": [m["id"] for m in excluded],
        "remaining": [_summary(m) for m in kept],
    }


def filter_by_size(all_masks, slot, top_n=None, min_area=None, max_area=None):
    """Filter masks in *slot* by ``area_fraction``.

    Provide at least one of:
      - ``top_n``   — keep only the N largest masks
      - ``min_area`` — keep masks with area_fraction >= threshold (0–1)
      - ``max_area`` — keep masks with area_fraction <= threshold (0–1)

    Useful for discarding spurious tiny detections or focusing on the
    dominant objects in a slot.
    """
    ms = _slot_masks(all_masks, slot)
    filtered = ms
    if min_area is not None:
        filtered = [m for m in filtered if m["area_fraction"] >= min_area]
    if max_area is not None:
        filtered = [m for m in filtered if m["area_fraction"] <= max_area]
    filtered = sorted(filtered, key=lambda m: m["area_fraction"], reverse=True)
    if top_n is not None:
        filtered = filtered[:top_n]
    return {
        "slot": slot,
        "n_remaining": len(filtered),
        "masks": [_summary(m) for m in filtered],
    }


# ---------------------------------------------------------------------------
# Cross-slot comparison
# ---------------------------------------------------------------------------

def compare_slot_positions(all_masks, slot_a, slot_b, axis="x"):
    """Compare the positional distributions of two slots along *axis*.

    Returns mean/min/max for each slot and a boolean indicating which slot
    is further in the positive direction (right for x, down for y).

    Use for "which team is further forward?", "which group is higher?",
    and as a quick sanity check before a more precise nth_from comparison.
    """
    ms_a = _slot_masks(all_masks, slot_a)
    ms_b = _slot_masks(all_masks, slot_b)

    if axis == "x":
        vals_a = [m["centroid_norm"]["x"] for m in ms_a]
        vals_b = [m["centroid_norm"]["x"] for m in ms_b]
        positive_label = "right"
    elif axis == "y":
        vals_a = [m["centroid_norm"]["y"] for m in ms_a]
        vals_b = [m["centroid_norm"]["y"] for m in ms_b]
        positive_label = "down"
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'.")

    mean_a = sum(vals_a) / len(vals_a)
    mean_b = sum(vals_b) / len(vals_b)

    return {
        "axis": axis,
        slot_a: {
            "mean": round(mean_a, 4),
            "min": round(min(vals_a), 4),
            "max": round(max(vals_a), 4),
            "n": len(vals_a),
        },
        slot_b: {
            "mean": round(mean_b, 4),
            "min": round(min(vals_b), 4),
            "max": round(max(vals_b), 4),
            "n": len(vals_b),
        },
        f"{slot_a}_mean_further_{positive_label}_than_{slot_b}": mean_a > mean_b,
    }


def closest_pair(all_masks, slot_a, slot_b):
    """Return the nearest pair of masks (one from each slot) by centroid distance.

    Useful for "which red player is closest to which blue player?" or
    detecting man-marking situations.
    """
    ms_a = _slot_masks(all_masks, slot_a)
    ms_b = _slot_masks(all_masks, slot_b)

    best_dist = float("inf")
    best_pair = None

    for ma in ms_a:
        for mb in ms_b:
            dx = ma["centroid_norm"]["x"] - mb["centroid_norm"]["x"]
            dy = ma["centroid_norm"]["y"] - mb["centroid_norm"]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_pair = (ma, mb)

    if best_pair is None:
        return {"error": "No masks found in one or both slots."}

    ma, mb = best_pair
    return {
        "distance_norm": round(best_dist, 4),
        slot_a: _summary(ma),
        slot_b: _summary(mb),
    }


# ---------------------------------------------------------------------------
# Registry — used by the agent loop to dispatch tool calls by name
# ---------------------------------------------------------------------------

# Maps tool name → (function, [required_param_names])
SPATIAL_OPS = {
    "rank_by_x": (rank_by_x, ["slot"]),
    "rank_by_y": (rank_by_y, ["slot"]),
    "extreme_mask": (extreme_mask, ["slot", "direction"]),
    "nth_from": (nth_from, ["slot", "n", "direction"]),
    "exclude_extremes": (exclude_extremes, ["slot"]),
    "filter_by_size": (filter_by_size, ["slot"]),
    "compare_slot_positions": (compare_slot_positions, ["slot_a", "slot_b"]),
    "closest_pair": (closest_pair, ["slot_a", "slot_b"]),
}


def dispatch(tool_name, all_masks, params):
    """Call the spatial operation *tool_name* with *params*.

    Returns a JSON-serialisable result dict, or raises KeyError / ValueError /
    IndexError with a descriptive message that the agent loop can forward to the VLM.
    """
    if tool_name not in SPATIAL_OPS:
        raise KeyError(
            f"'{tool_name}' is not a spatial operation. "
            f"Available: {sorted(SPATIAL_OPS.keys())}"
        )

    fn, _ = SPATIAL_OPS[tool_name]

    # Build kwargs from params (all are optional at the Python level since
    # functions have defaults; missing required args will raise naturally)
    kwargs = {k: v for k, v in params.items()}
    return fn(all_masks, **kwargs)
