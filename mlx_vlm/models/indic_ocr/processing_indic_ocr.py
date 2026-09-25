"""Block-level prompts and crop helpers for IndicOCR.

Prompts and the label taxonomy are ported from the upstream repo
(``idp_contract.py`` / ``idp_crops.py``, bodhan-ai/indic-ocr).
Image encoding itself uses the standard Qwen3VLProcessor.
"""

import math
from typing import Optional

from PIL import Image

from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

# Prompts (verbatim from idp_contract.py)
TEXT_PROMPT = (
    "Transcribe the text in this image. Write any mathematical expressions in LaTeX, "
    "using $...$ for inline math and $$...$$ for display equations."
)
EQUATION_PROMPT = "Output only the LaTeX for this equation image."
TABLE_PROMPT_HTML = (
    "Convert this table image to HTML. Preserve the structure exactly, using colspan and "
    "rowspan for merged cells and <br/> for line breaks within a cell. "
    "Output only the HTML table."
)
TABLE_PROMPT_MARKDOWN = "Convert this table image to a GitHub-flavored markdown table. Output only the table."

# Label -> pipeline type (from idp_contract.LABEL_TO_TYPE)
LABEL_TO_TYPE = {
    "table": "Table",
    "table-caption": "Caption",
    "equation": "Equation",
    "expression": "Equation",
    "diagram": "Figure",
    "chart": "Figure",
    "image": "Picture",
    "image-caption": "Caption",
    "title": "Title",
    "chapter-title": "Title",
    "section-title": "SectionHeader",
    "sub-section-title": "SectionHeader",
    "sub-sub-section-title": "SectionHeader",
    "header": "PageHeader",
    "footer": "PageFooter",
    "page-number": "PageNumber",
    "folio": "PageNumber",
    "footnote": "Footnote",
}

# Stock PP-DocLayoutV3 labels differ from the IndicDocLayout fine-tune.
# Keep their original spelling in records while selecting the OCR role here.
PP_DOCLAYOUT_LABEL_TO_TYPE = {
    "abstract": "Text",
    "algorithm": "Text",
    "aside_text": "Text",
    "chart": "Figure",
    "content": "Text",
    "formula": "Equation",
    "doc_title": "Title",
    "figure_title": "Caption",
    "footer": "PageFooter",
    "footnote": "Footnote",
    "formula_number": "Text",
    "header": "PageHeader",
    "image": "Picture",
    "number": "PageNumber",
    "paragraph_title": "SectionHeader",
    "reference": "Text",
    "reference_content": "Text",
    "seal": "Text",
    "table": "Table",
    "text": "Text",
    "vision_footnote": "Footnote",
}

# Blocks never sent to the recognizer (from idp_contract.DROP_TYPES)
DROP_TYPES = frozenset({"Figure", "Picture"})

# Pipeline types map_label can produce, less DROP_TYPES
KEPT_BLOCK_TYPES = (
    "Text",
    "Title",
    "SectionHeader",
    "Table",
    "Equation",
    "Caption",
    "Footnote",
    "PageHeader",
    "PageFooter",
    "PageNumber",
)

# 37-class IndicDocLayout taxonomy (from idp_model_labels.CLASSES)
CLASSES = [
    "Question",
    "Paragraph",
    "Answer",
    "List",
    "Title",
    "Section-title",
    "Equation",
    "Table",
    "Diagram",
    "Image",
    "MCQ",
    "Infobox",
    "Sub-section-title",
    "Expression",
    "Image-caption",
    "Placeholder-text",
    "Chart",
    "Solved-example",
    "Footnote",
    "Table-caption",
    "Sub-sub-section-title",
    "Footer",
    "Header",
    "Code",
    "Page-number",
    "Chapter-title",
    "Chapter-end-section",
    "Folio",
    "Reference",
    "Table-of-contents",
    "Index",
    "Advertisement",
    "Author",
    "Dateline",
    "Contact-info",
    "Website-link",
    "Flag",
]

# Cleaned as their own group so a page-spanning paragraph cannot swallow
# a page number (from idp_contract.MARGINALIA / HEAD_FOOT)
MARGINALIA = frozenset({"Header", "Footer", "Page-number", "Folio"})
HEAD_FOOT = ("Header", "Footer")

# Never sent to the recognizer but kept in place with text "".
OCR_SKIP_LABELS = frozenset(
    {"header", "footer", "diagram", "image", "chart", "advertisement"}
)


def is_transcribed(label) -> bool:
    return str(label).strip().lower() not in OCR_SKIP_LABELS


def map_label(label) -> str:
    label = str(label).strip().lower()
    return LABEL_TO_TYPE.get(label, PP_DOCLAYOUT_LABEL_TO_TYPE.get(label, "Text"))


def prompt_for(block_type: str, table_format: str = "html") -> str:
    if block_type == "Table":
        if table_format == "markdown":
            return TABLE_PROMPT_MARKDOWN
        if table_format != "html":
            raise ValueError(
                f"table_format must be 'html' or 'markdown', got {table_format!r}"
            )
        return TABLE_PROMPT_HTML
    if block_type == "Equation":
        return EQUATION_PROMPT
    return TEXT_PROMPT


def area_clamp(
    image: Image.Image, min_px: int = 65536, max_px: int = 2359296
) -> Image.Image:
    """Scale a crop so its area lands in [min_px, max_px], preserving aspect."""
    width, height = image.size
    pixels = width * height
    if pixels <= 0:
        return image
    if pixels < min_px:
        scale = math.sqrt(min_px / pixels)
    elif pixels > max_px:
        scale = math.sqrt(max_px / pixels)
    else:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
    )


def crop_for_block(
    bbox_xyxy: list,
    page: Image.Image,
    pad_px: int = 0,
) -> Optional[Image.Image]:
    """Crop one layout box from the page, or None if it collapses."""
    width, height = page.size
    x0, y0, x1, y1 = (round(v) for v in bbox_xyxy)
    if pad_px:
        x0, y0 = x0 - pad_px, y0 - pad_px
        x1, y1 = x1 + pad_px, y1 + pad_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return page.crop((x0, y0, x1, y1)).convert("RGB")


install_auto_processor_patch("indic_ocr", Qwen3VLProcessor)


def build_ocr_requests(
    blocks,
    page,
    table_format: str = "html",
    pad_px: int = 0,
    min_px: int = 65536,
    max_px: int = 2359296,
) -> list:
    """Pair each transcribable block with its (crop, prompt).

    Returns ``[(block, crop, prompt)]`` in input order. Blocks that yield
    no crop are absent -- the caller sets their text to "".
    """
    from .blocks import Block  # noqa: F401  (type reference only)

    requests = []
    for block in blocks:
        if block.type in DROP_TYPES or not is_transcribed(block.label):
            continue
        crop = crop_for_block(block.bbox_xyxy, page, pad_px=pad_px)
        if crop is None:
            continue
        requests.append(
            (
                block,
                area_clamp(crop, min_px, max_px),
                prompt_for(block.type, table_format),
            )
        )
    return requests


def transcribe_blocks(
    model,
    processor,
    page,
    blocks,
    table_format: str = "html",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    use_tqdm: bool = True,
    pad_px: int = 0,
    min_px: int = 65536,
    max_px: int = 2359296,
) -> list:
    """Transcribe layout blocks in place via the standard generate flow.

    ``blocks`` carry ``label``/``type``/``bbox_xyxy`` (see ``blocks.Block``);
    each transcribable block gets ``text``, skipped ones get ``""``.
    Greedy decoding matches upstream's reproducible setting.
    """
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    requests = build_ocr_requests(
        blocks, page, table_format, pad_px=pad_px, min_px=min_px, max_px=max_px
    )
    transcribed = {id(block) for block, _, _ in requests}

    iterator = requests
    if use_tqdm:
        try:
            from tqdm import tqdm

            iterator = tqdm(requests, desc="OCR", unit="block")
        except ImportError:
            pass

    for block, crop, prompt in iterator:
        formatted = apply_chat_template(processor, model.config, prompt, num_images=1)
        block.text = mlx_generate(
            model,
            processor,
            formatted,
            [crop],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        ).text.strip()

    for block in blocks:
        if id(block) not in transcribed:
            block.text = ""
    return blocks


__all__ = [
    "TEXT_PROMPT",
    "EQUATION_PROMPT",
    "TABLE_PROMPT_HTML",
    "TABLE_PROMPT_MARKDOWN",
    "LABEL_TO_TYPE",
    "PP_DOCLAYOUT_LABEL_TO_TYPE",
    "DROP_TYPES",
    "KEPT_BLOCK_TYPES",
    "CLASSES",
    "MARGINALIA",
    "HEAD_FOOT",
    "OCR_SKIP_LABELS",
    "is_transcribed",
    "map_label",
    "prompt_for",
    "area_clamp",
    "crop_for_block",
    "build_ocr_requests",
    "transcribe_blocks",
]
