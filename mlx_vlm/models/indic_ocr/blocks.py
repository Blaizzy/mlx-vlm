"""Layout cleanup: geometry, duplicate suppression, nested-equation resolution.

Ported from upstream ``idp_blocks.py`` (bodhan-ai/indic-ocr). Pure geometry
on blocks -- no PIL, no model -- so the rules that decide what gets
transcribed stay testable in isolation.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

from .processing_indic_ocr import (
    CLASSES,
    DROP_TYPES,
    KEPT_BLOCK_TYPES,
    PP_DOCLAYOUT_LABEL_TO_TYPE,
    is_transcribed,
    map_label,
)

TEXTLIKE = frozenset({"Text", "Title", "SectionHeader", "Caption", "Footnote"})


class LayoutSchemaError(ValueError):
    """An incoming layout record cannot be interpreted unambiguously."""


def _unknown(value, candidates, kind: str) -> str:
    import difflib

    match = difflib.get_close_matches(str(value), list(candidates), n=1, cutoff=0.6)
    hint = f"; did you mean {match[0]!r}?" if match else ""
    return f"unknown {kind} {value!r}{hint}"


@dataclass
class Block:
    """One detected region. ``text`` is None before OCR, and "" for blocks
    deliberately not transcribed (kept in place rather than deleted)."""

    order: int
    label: str
    type: str
    bbox_xyxy: list
    conf: float
    text: Optional[str] = None

    def as_record(self) -> dict:
        # Key order is load-bearing: json.dump writes insertion order.
        record: dict = {
            "order": self.order,
            "label": self.label,
            "type": self.type,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
            "conf": round(float(self.conf), 3),
        }
        if self.text is not None:
            record["text"] = self.text
        return record

    @classmethod
    def problems(cls, record: dict) -> list:
        found: list = []
        for key in ("order", "bbox_xyxy"):
            if key not in record:
                found.append(f"missing required key {key!r}")

        if "order" in record and (
            not isinstance(record["order"], int)
            or isinstance(record["order"], bool)
            or record["order"] < 0
        ):
            found.append(
                f"order must be a nonnegative integer, got {record['order']!r}"
            )

        bbox = record.get("bbox_xyxy")
        if bbox is not None:
            try:
                if len([float(v) for v in bbox]) != 4:
                    found.append(
                        f"bbox_xyxy must be 4 numbers [x0, y0, x1, y1], got {bbox!r}"
                    )
            except (TypeError, ValueError):
                found.append(
                    f"bbox_xyxy must be 4 numbers [x0, y0, x1, y1], got {bbox!r}"
                )

        declared = record.get("type")
        if declared:
            valid = tuple(KEPT_BLOCK_TYPES) + tuple(sorted(DROP_TYPES))
            if str(declared) not in valid:
                found.append(_unknown(declared, valid, "type"))
        else:
            label = str(record.get("label", "")).strip()
            known = {
                c.strip().lower() for c in CLASSES
            } | PP_DOCLAYOUT_LABEL_TO_TYPE.keys()
            if label.lower() not in known:
                found.append(
                    _unknown(
                        label, list(CLASSES) + list(PP_DOCLAYOUT_LABEL_TO_TYPE), "label"
                    )
                    + ' -- use an IndicDocLayout or PP-DocLayoutV3 label, or declare "type"'
                    " explicitly if your detector has its own taxonomy"
                )
        return found

    @classmethod
    def from_record(cls, record: dict, strict: bool = True) -> "Block":
        if strict:
            found = cls.problems(record)
            if found:
                raise LayoutSchemaError("; ".join(found))

        label = record.get("label", "")
        return cls(
            order=int(record["order"]),
            label=str(label),
            type=str(record.get("type") or map_label(label)),
            bbox_xyxy=[float(v) for v in record["bbox_xyxy"]],
            conf=float(record.get("conf", 1.0)),
            text=record.get("text"),
        )

    def copy(self) -> "Block":
        return dataclasses.replace(self, bbox_xyxy=list(self.bbox_xyxy))


def area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def contained_frac(small, big) -> float:
    ix0, iy0 = max(small[0], big[0]), max(small[1], big[1])
    ix1, iy1 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a = area(small)
    return inter / a if a > 0 else 0.0


def clamp_to_page(bbox, width: int, height: int) -> list:
    return [
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(float(width), bbox[2]),
        min(float(height), bbox[3]),
    ]


def clean_layout(blocks: list, contain: float = 0.90, wrap: float = 0.5) -> list:
    """Drop duplicate and spurious boxes. Survivors keep input order.

    1. Nested duplicates -- a box ``contain``-inside a larger box goes.
    2. One header, one footer -- only the largest of each survives.
    3. Empty frames -- a header/footer wrapping nothing is dropped.
    """
    from .processing_indic_ocr import HEAD_FOOT, MARGINALIA

    n = len(blocks)
    box = lambda i: blocks[i].bbox_xyxy  # noqa: E731
    drop: set = set()
    marginalia = {map_label(label) for label in MARGINALIA}
    head_foot = tuple(map_label(label) for label in HEAD_FOOT)

    def transcribed(block):
        return block.type not in DROP_TYPES and is_transcribed(block.label)

    groups = (
        lambda block: block.type not in marginalia,
        lambda block: block.type in marginalia and block.type not in head_foot,
    )
    for in_group in groups:
        idxs = sorted(
            (i for i in range(n) if in_group(blocks[i])),
            key=lambda i: area(box(i)),
            reverse=True,
        )
        kept: list = []
        for i in idxs:
            has_text = transcribed(blocks[i])
            if any(
                contained_frac(box(i), box(j)) >= contain
                and (transcribed(blocks[j]) or not has_text)
                for j in kept
            ):
                drop.add(i)
            else:
                kept.append(i)

    for block_type in head_foot:
        group = [i for i in range(n) if blocks[i].type == block_type and i not in drop]
        if not group:
            continue
        biggest = max(group, key=lambda i: area(box(i)))
        drop.update(i for i in group if i != biggest)
        wraps = any(
            blocks[k].type != block_type
            and contained_frac(box(k), box(biggest)) >= wrap
            for k in range(n)
        )
        if not wraps:
            drop.add(biggest)

    return [blocks[i] for i in range(n) if i not in drop]


def _nested_in(inner, outer, thresh: float) -> bool:
    inner_area = area(inner)
    if inner_area <= 0:
        return False
    return contained_frac(inner, outer) >= thresh and inner_area < 0.95 * area(outer)


def _absorbs_equation(container: Block, mode: str) -> bool:
    if container.type == "Equation":
        return mode != "text_only"
    if container.type in TEXTLIKE:
        return mode != "eq_only"
    return False


def resolve_nested_equations(
    blocks: list, nest: bool = True, mode: str = "both", nested: float = 0.70
) -> list:
    """Drop equation boxes nested in a container that already covers them."""
    if not nest:
        return list(blocks)

    drop: set = set()
    for i, inner in enumerate(blocks):
        if inner.type != "Equation":
            continue
        for j, container in enumerate(blocks):
            if (
                i != j
                and _nested_in(inner.bbox_xyxy, container.bbox_xyxy, nested)
                and _absorbs_equation(container, mode)
            ):
                drop.add(i)
                break
    return [b for k, b in enumerate(blocks) if k not in drop]


__all__ = [
    "Block",
    "LayoutSchemaError",
    "TEXTLIKE",
    "area",
    "contained_frac",
    "clamp_to_page",
    "clean_layout",
    "resolve_nested_equations",
]
