"""Processor registration for MiniMax-M3 VL.

`minimax_m3_vl` is not in installed transformers yet, so the upstream processor + fast image
processor (vendored alongside as `processing_minimax.py` / `image_processor.py`) are registered
with the Auto* factories here. Importing this module installs them; the mlx-vlm loader's
`AutoProcessor.from_pretrained` then resolves them for VL inference. Text-only use needs only the
tokenizer and does not depend on this.
"""
from .image_processor import MiniMaxM3VLImageProcessor
from .processing_minimax import MiniMaxVLProcessor

try:
    from transformers import AutoImageProcessor, AutoProcessor

    AutoImageProcessor.register("MiniMaxM3VLImageProcessor", fast_image_processor_class=MiniMaxM3VLImageProcessor)
    AutoProcessor.register("MiniMaxVLProcessor", MiniMaxVLProcessor)
except Exception as e:  # already registered, or a transformers that ships it natively
    print(f"minimax_m3 processor registration skipped: {e}")
