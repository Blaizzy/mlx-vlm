from __future__ import annotations

from PIL import Image

from mlx_vlm.models.dots_ocr.processor import RECOMMENDED_DPI


def pdf_to_images(path: str, dpi: int = RECOMMENDED_DPI) -> list[Image.Image]:
    """Convert a PDF into RGB `PIL.Image` pages at the requested DPI."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "pdf2image is required to convert PDFs; install via `pip install pdf2image`"
        ) from exc

    pages = convert_from_path(path, dpi=dpi)
    return [page.convert("RGB") for page in pages]
