"""End-to-end IndicOCR page pipeline in MLX.

``IndicOCRParser`` mirrors upstream ``IndicOCR`` (``idp_offline.py``):

    page image -> layout detect -> dedup -> per-block OCR -> Markdown + JSON

Stage 1 runs the MLX layout detector; stage 2 the MLX OCR model. A
layout produced elsewhere (torch upstream, hand-edited JSON, another
detector) can be replayed via ``JsonLayoutBackend``.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import mlx.nn as nn
from PIL import Image

from .blocks import (
    Block,
    LayoutSchemaError,
    clamp_to_page,
    clean_layout,
    resolve_nested_equations,
)
from .processing_indic_ocr import is_transcribed
from .reconstruct import reconstruct


@dataclass(frozen=True)
class LayoutOptions:
    conf: float = 0.5
    img_size: int = 1024


@dataclass(frozen=True)
class CropOptions:
    min_px_side: int = 256
    max_px_side: int = 1536
    pad_px: int = 0

    @property
    def min_px(self) -> int:
        return self.min_px_side**2

    @property
    def max_px(self) -> int:
        return self.max_px_side**2


@dataclass(frozen=True)
class DedupOptions:
    nest: bool = True
    mode: str = "both"
    contain: float = 0.90
    wrap: float = 0.5
    nested: float = 0.70

    def __post_init__(self):
        if self.mode not in ("both", "text_only", "eq_only"):
            raise ValueError(
                f"DedupOptions.mode must be both/text_only/eq_only, got {self.mode!r}"
            )


@dataclass(frozen=True)
class RecognizerOptions:
    max_tokens: int = 2048
    temperature: float = 0.0  # greedy: the only reproducible setting
    table_format: str = "html"

    def __post_init__(self):
        if self.table_format not in ("html", "markdown"):
            raise ValueError(
                "RecognizerOptions.table_format must be 'html' or 'markdown', "
                f"got {self.table_format!r}"
            )


@dataclass
class PageResult:
    image: str
    width: int
    height: int
    blocks: list = field(default_factory=list)
    markdown: Optional[str] = None

    def as_record(self) -> dict:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "blocks": [b.as_record() for b in self.blocks],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_record(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def from_record(cls, record: dict, strict: bool = True) -> "PageResult":
        blocks = list(record.get("blocks", []))
        if strict:
            errors = []
            seen_orders = set()
            for i, block in enumerate(blocks):
                if not isinstance(block, dict):
                    errors.append(f"  block[{i}]: expected a block record")
                    continue
                errors.extend(
                    f"  block[{i}] (order={block.get('order', '?')!r}): {problem}"
                    for problem in Block.problems(block)
                )
                order = block.get("order")
                if isinstance(order, int):
                    if order in seen_orders:
                        errors.append(f"  block[{i}]: duplicate order {order}")
                    seen_orders.add(order)
            if errors:
                raise LayoutSchemaError(
                    f"{len(errors)} problem(s) in layout for "
                    f"{record.get('image', '<unknown>')!r}:\n" + "\n".join(errors)
                )
        return cls(
            image=str(record["image"]),
            width=int(record["width"]),
            height=int(record["height"]),
            blocks=[Block.from_record(b, strict=False) for b in blocks],
        )


def _open(image_path: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(image_path).convert("RGB")


def _densify(blocks: list) -> list:
    """Sort by detector reading order, renumber to a gap-free 0-based rank."""
    ordered = sorted(blocks, key=lambda b: b.order)
    for rank, block in enumerate(ordered):
        block.order = rank
    return ordered


def viewer_records_to_blocks(records: list, width: int, height: int) -> list:
    """Convert detector viewer records to pixel-coordinate OCR blocks."""
    blocks = []
    for rec in records:
        order = rec["reading_order"]
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise LayoutSchemaError(
                f"reading_order must be a positive integer, got {order!r}"
            )
    for rec in sorted(records, key=lambda item: item["reading_order"]):
        y0, x0, y1, x1 = rec["bbox"]
        label = str(rec["label"])
        block = {
            "order": rec["reading_order"] - 1,
            "label": label,
            "bbox_xyxy": [
                x0 / 1000 * width,
                y0 / 1000 * height,
                x1 / 1000 * width,
                y1 / 1000 * height,
            ],
            "conf": float(rec.get("score", 1.0)),
        }
        if "type" in rec:
            block["type"] = rec["type"]
        blocks.append(Block.from_record(block))
    return blocks


def _resolve_repo_root(
    repo_or_path: str,
    revision: Optional[str] = None,
    force_download: bool = False,
):
    from mlx_vlm.utils import get_model_path

    root = get_model_path(
        repo_or_path, revision=revision, force_download=force_download
    )
    with open(root / "config.json", encoding="utf-8") as fh:
        config = json.load(fh)
    stages = dict(config.get("stages", {}))
    if "text_config" in config and "vision_config" in config:
        stages.setdefault("ocr", {"config": "config.json"})
    return root, stages


def _stage_dir(root, stages: dict, name: str, default: str):
    from pathlib import Path

    return root / Path(stages.get(name, {}).get("config", default)).parent


def _resolve_layout_source(root, stages: dict, layout_repo: Optional[str] = None):
    """Where to load the layout stage from: an explicit repo/path, else
    the single-repo ``weights/layout`` subdir, else an error."""
    from pathlib import Path

    if layout_repo is not None:
        return layout_repo
    candidate = _stage_dir(root, stages, "layout", "weights/layout/config.json")
    if (Path(candidate) / "config.json").is_file():
        return str(candidate)
    raise ValueError(
        "No layout stage found: pass layout_repo=<hf-id-or-path> (e.g. "
        "'HashNuke/pp-doclayout-v3-mlx'), or use a single repo containing "
        "weights/layout (e.g. 'HashNuke/indic-ocr-mlx')."
    )


class MLXLayoutBackend(nn.Module):
    """Stage 1 with the MLX layout detector."""

    def __init__(
        self,
        model,
        options: Optional[LayoutOptions] = None,
        dedup: Optional[DedupOptions] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.options = options or LayoutOptions()
        self.dedup = dedup or DedupOptions()

    def detect(self, image: Image.Image) -> list:
        if self.model is None:
            raise RuntimeError("Layout backend is closed.")
        width, height = image.size
        records = self.model.detect(
            image, conf=self.options.conf, img_size=self.options.img_size
        )
        blocks = viewer_records_to_blocks(records, width, height)
        for block in blocks:
            block.bbox_xyxy = [
                round(v, 1) for v in clamp_to_page(block.bbox_xyxy, width, height)
            ]
            block.conf = round(block.conf, 3)
        return _densify(
            clean_layout(blocks, contain=self.dedup.contain, wrap=self.dedup.wrap)
        )

    def close(self) -> None:
        self.model = None


class JsonLayoutBackend:
    """Replay a layout produced elsewhere. Assumed clean; only renumbered."""

    def __init__(self, layout, strict: bool = True) -> None:
        if isinstance(layout, PageResult):
            self.page = layout
        else:
            if isinstance(layout, str):
                with open(layout, encoding="utf-8") as fh:
                    layout = json.load(fh)
            self.page = PageResult.from_record(layout, strict=strict)

    def detect(self, image: Image.Image) -> list:
        return _densify([b.copy() for b in self.page.blocks])

    def close(self) -> None:
        return None


class BlockOCRRunner(nn.Module):
    """Stage 2: per-block transcription against a layout (any backend)."""

    def __init__(
        self,
        model,
        processor,
        options: Optional[RecognizerOptions] = None,
        dedup: Optional[DedupOptions] = None,
        crop: Optional[CropOptions] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.options = options or RecognizerOptions()
        self.dedup = dedup or DedupOptions()
        self.crop = crop or CropOptions()

    def run(self, image_path: str, layout) -> PageResult:
        """Every block of the layout comes back, in its original order.
        Blocks that were not transcribed carry ``text: ""``."""
        from .processing_indic_ocr import transcribe_blocks

        if self.model is None or self.processor is None:
            raise RuntimeError("OCR backend is closed.")
        image = _open(image_path)
        page = PageResult.from_record(
            layout.as_record()
            if isinstance(layout, PageResult)
            else (
                json.load(open(layout, encoding="utf-8"))
                if isinstance(layout, str)
                else layout
            )
        )
        blocks = [b.copy() for b in page.blocks]

        eligible = resolve_nested_equations(
            [b for b in blocks if is_transcribed(b.label)],
            nest=self.dedup.nest,
            mode=self.dedup.mode,
            nested=self.dedup.nested,
        )
        transcribe_blocks(
            self.model,
            self.processor,
            image,
            eligible,
            table_format=self.options.table_format,
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            pad_px=self.crop.pad_px,
            min_px=self.crop.min_px,
            max_px=self.crop.max_px,
        )
        by_order = {b.order: (b.text or "") for b in eligible}
        for block in blocks:
            block.text = (by_order.get(block.order) or "").strip()

        return PageResult(
            image=page.image,
            width=page.width,
            height=page.height,
            blocks=blocks,
            markdown=reconstruct(blocks),
        )

    def close(self) -> None:
        self.model = None
        self.processor = None


class IndicOCRParser(nn.Module):
    """Both stages in one object: ``parse("page.png")`` -> PageResult."""

    def __init__(
        self,
        layout_model,
        ocr_model,
        ocr_processor,
        layout_options: Optional[LayoutOptions] = None,
        recognizer_options: Optional[RecognizerOptions] = None,
        dedup: Optional[DedupOptions] = None,
        crop: Optional[CropOptions] = None,
    ) -> None:
        super().__init__()
        dedup = dedup or DedupOptions()
        self.layout = MLXLayoutBackend(layout_model, layout_options, dedup)
        self.ocr = BlockOCRRunner(
            ocr_model, ocr_processor, recognizer_options, dedup, crop
        )

    @classmethod
    def from_config(cls, config):
        from ..pp_doclayout_v3 import Model as LayoutModel
        from .indic_ocr import Model as OCRModel

        parser = cls(
            LayoutModel(config.layout_config),
            OCRModel(config.ocr_config),
            None,
        )
        parser.config = config
        return parser

    def sanitize(self, weights):
        """Route stage-prefixed tensors through each submodel's sanitizer.

        Input prefixes are ``layout_model.`` and ``ocr_model.``; converted
        checkpoints use the module paths ``layout.model.`` and ``ocr.model.``.
        Unknown keys are preserved so strict loading can report them.
        """
        remaining = dict(weights)
        mapping = getattr(getattr(self, "config", None), "weight_mapping", None) or {}
        for key in list(remaining):
            if key in mapping:
                destination = f"{mapping[key]}_model.{key}"
                if destination in remaining:
                    raise ValueError(f"Duplicate stage weight: {key}")
                remaining[destination] = remaining.pop(key)
        result = {}
        for name, backend in (("layout", self.layout), ("ocr", self.ocr)):
            stage_weights = {}
            for key in list(remaining):
                for prefix in (f"{name}_model.", f"{name}.model."):
                    if key.startswith(prefix):
                        stage_key = key[len(prefix) :]
                        if stage_key in stage_weights:
                            raise ValueError(f"Duplicate {name} weight: {stage_key}")
                        stage_weights[stage_key] = remaining.pop(key)
                        break
            stage_weights = backend.model.sanitize(stage_weights)
            if name == "ocr" and backend.model.vision_tower is not None:
                stage_weights = backend.model.vision_tower.sanitize(stage_weights)
            result.update(
                (f"{name}.model.{key}", value) for key, value in stage_weights.items()
            )
        result.update(remaining)
        return result

    @classmethod
    def _from_models(cls, *args, **kwargs) -> "IndicOCRParser":
        return cls(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        layout_options: Optional[LayoutOptions] = None,
        recognizer_options: Optional[RecognizerOptions] = None,
        dedup: Optional[DedupOptions] = None,
        crop: Optional[CropOptions] = None,
        revision: Optional[str] = None,
        layout_repo: Optional[str] = None,
        lazy: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> "IndicOCRParser":
        """Load a two-stage repository or a flat OCR stage (HF id or local path).

        ``layout_repo`` overrides the bundled layout stage and is required
        for a flat OCR model. Both IndicDocLayout and stock PP-DocLayoutV3
        label taxonomies are supported. ``lazy``, ``strict``, and additional
        loader arguments are forwarded to both stages.
        """
        from mlx_vlm import load as mlx_load
        from mlx_vlm.utils import get_model_path, load_config
        from mlx_vlm.utils import load_model as mlx_load_model

        root, stages = _resolve_repo_root(
            repo_or_path, revision, force_download=kwargs.get("force_download", False)
        )
        config = load_config(root)
        if "layout_config" in config or "ocr_config" in config:
            from mlx_vlm.utils import load_processor

            parser = mlx_load_model(root, lazy=lazy, strict=strict, **kwargs)
            if layout_repo is not None:
                parser.layout.model = mlx_load_model(
                    get_model_path(layout_repo), lazy=lazy, strict=strict, **kwargs
                )
            parser.ocr.processor = load_processor(
                root / (parser.config.ocr_model_path or "."),
                eos_token_ids=parser.config.ocr_config.eos_token_id,
                **kwargs,
            )
            parser.layout.options = layout_options or LayoutOptions()
            parser.ocr.options = recognizer_options or RecognizerOptions()
            parser.layout.dedup = parser.ocr.dedup = dedup or DedupOptions()
            parser.ocr.crop = crop or CropOptions()
            return parser

        ocr_dir = _stage_dir(root, stages, "ocr", "weights/ocr/config.json")
        if not (ocr_dir / "config.json").is_file():
            raise FileNotFoundError(f"No OCR config found at {ocr_dir / 'config.json'}")
        layout_source = _resolve_layout_source(root, stages, layout_repo)
        layout_path = get_model_path(
            layout_source, force_download=kwargs.get("force_download", False)
        )
        for name, stage_path in (("layout", layout_path), ("ocr", ocr_dir)):
            stage_config = load_config(stage_path)
            if (
                (stage_config.get("model_type") or "").lower() == "indic_ocr"
                and "stages" in stage_config
                and "text_config" not in stage_config
            ):
                raise ValueError(
                    f"The {name} stage at {stage_path} points to a two-stage "
                    "wrapper; each stage must point to an individual model."
                )

        layout_model = mlx_load_model(
            layout_path,
            lazy=lazy,
            strict=strict,
            **kwargs,
        )
        if hasattr(layout_model, "eval"):
            layout_model.eval()
        ocr_model, ocr_processor = mlx_load(
            str(ocr_dir), lazy=lazy, strict=strict, **kwargs
        )
        return cls._from_models(
            layout_model,
            ocr_model,
            ocr_processor,
            layout_options,
            recognizer_options,
            dedup,
            crop,
        )

    def detect(self, image_path: str) -> PageResult:
        image = _open(image_path)
        return PageResult(
            image=os.path.basename(image_path),
            width=image.width,
            height=image.height,
            blocks=self.layout.detect(image),
        )

    def parse(self, image_path: str) -> PageResult:
        if self.ocr.processor is None and self.ocr.model is not None:
            from pathlib import Path

            from mlx_vlm.utils import load_processor

            config = getattr(self, "config", None)
            processor_path = getattr(config, "ocr_model_path", None)
            model_path = getattr(self, "model_path", None)
            if processor_path is not None and not Path(processor_path).is_absolute():
                processor_path = (
                    Path(model_path) / processor_path
                    if model_path is not None
                    else None
                )
            elif processor_path is None:
                processor_path = model_path
            if processor_path is not None:
                self.ocr.processor = load_processor(
                    Path(processor_path),
                    eos_token_ids=self.ocr.model.config.eos_token_id,
                )
        return self.ocr.run(image_path, self.detect(image_path))

    def close(self) -> None:
        self.layout.close()
        self.ocr.close()

    def __enter__(self) -> "IndicOCRParser":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


__all__ = [
    "LayoutOptions",
    "CropOptions",
    "DedupOptions",
    "RecognizerOptions",
    "PageResult",
    "viewer_records_to_blocks",
    "MLXLayoutBackend",
    "JsonLayoutBackend",
    "BlockOCRRunner",
    "IndicOCRParser",
]
