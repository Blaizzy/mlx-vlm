"""BFL bounding-box scaffolding and reconciliation."""

import json
import math
import re

SCALE = 1000
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def to_canvas(boxes: list[dict]) -> list[dict]:
    """0-1 boxes -> upsampler rows on the 0-1000 canvas."""

    def grid(box):
        if not box:
            return None
        if len(box) != 4:
            raise ValueError(f"a box is [y_min, x_min, y_max, x_max], got {box}")
        return [round(max(0.0, min(1.0, float(v))) * SCALE) for v in box]

    rows = []
    for i, box in enumerate(boxes):
        row_id, desc = str(box.get("id") or f"obj_{i + 1}"), str(box.get("desc") or "")
        if box.get("bbox"):
            rows.append({"id": row_id, "bbox": grid(box["bbox"]), "desc": desc})
        elif box.get("src_bbox") or box.get("tgt_bbox"):
            rows.append(
                {
                    "id": row_id,
                    "from": str(box.get("from") or ""),
                    "src_bbox": grid(box.get("src_bbox")),
                    "tgt_bbox": grid(box.get("tgt_bbox")),
                    "desc": desc,
                }
            )
        else:
            raise ValueError(f"box {i} has neither bbox nor src_bbox/tgt_bbox")
    return rows


def detect(prompt: str) -> tuple[str, list[dict]] | None:
    """(prompt, rows) when the prompt already carries boxes on the 0-1000 canvas, else None. Two forms:
    "<caption> [rows]" (the DiT prompt format) and the upsampler's own "ORIGINAL PROMPT: ... LOCKED
    BOUNDING-BOX ANNOTATIONS ...:\\n[rows]" input. A trailing list that is not all box rows (e.g. a
    caption ending in "[SALE]") is not a box table."""
    text, rows = split(prompt)
    if not rows or not all(
        isinstance(r, dict)
        and isinstance(r.get("id"), str)
        and ID.fullmatch(r["id"])
        and any(valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox"))
        for r in rows
    ):
        return None
    locked = re.fullmatch(
        r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*",
        text,
        re.DOTALL,
    )
    return (locked.group(1) if locked else text), rows


def is_finished(prompt: str) -> bool:
    """An upsampler answer, not a request: its box table is cited in full as <id>s in the caption."""
    found = detect(prompt)
    return found is not None and all(f"<{row['id']}>" in found[0] for row in found[1])


def plan(
    prompt: str, boxes: list[dict] | None, n_images: int
) -> tuple[str, str, list[dict] | None]:
    """(system prompt key, user text, locked rows) for the upsampler. --boxes wins; otherwise boxes found in
    the prompt switch it to l3b by themselves."""
    task = "i2i" if n_images else "t2i"
    if boxes:
        locked = to_canvas(boxes)
    elif found := detect(prompt):
        prompt, locked = found
    else:
        return task, prompt, None
    return f"{task}_bbox", upsampling_input(prompt, locked), locked


def upsampling_input(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


def split(text: str) -> tuple[str, list | None]:
    """caption, rows: the last '[' from which the rest parses as a JSON list. Not a regex: descs may
    contain brackets and escaped quotes."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_box(box) -> list | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in box
    ):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if (
        y1 < 0
        or x1 < 0
        or y2 <= y1
        or x2 <= x1
        or y2 > SCALE
        or x2 > SCALE
        or max(box) <= 1
    ):
        return None
    return list(box)


def valid_added(row, caption: str) -> dict | None:
    """A row the upsampler added: kept only if its id is cited as <id> and it is well formed."""
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (
        not isinstance(row_id, str)
        or not ID.fullmatch(row_id)
        or f"<{row_id}>" not in caption
        or not isinstance(desc, str)
        or not desc.strip()
    ):
        return None
    if (box := valid_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_box(src) if src is not None else None
    tgt_box = valid_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {
        "id": row_id,
        "from": origin,
        "src_bbox": src_box,
        "tgt_bbox": tgt_box,
        "desc": desc.strip(),
    }


def iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reconcile(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed.

    1. A row re-planning a locked box under its own id (IoU >= 0.9, e.g. balloon_1 for obj_1) takes the
       locked id, in the rows and in the caption's <id> citations.
    2. Locked rows are kept verbatim, adopting the upsampler's desc; its own rows follow if valid.
    """
    caption, rows = split(text)
    rows = rows or []
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("id") in renamed
                or row.get("id") in locked_ids
            ):
                continue
            box = valid_box(
                row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox")
            )
            if box is not None and iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {
        r.get("id"): r
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append(
            {
                **lock,
                "desc": (
                    desc.strip()
                    if isinstance(desc, str) and desc.strip()
                    else lock.get("desc", "")
                ),
            }
        )
    for row in rows:
        added = valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"
