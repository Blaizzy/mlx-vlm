"""SAM 3.1 Processor — same preprocessing as SAM 3."""

from ..base import install_auto_processor_patch
from ..sam3.processing_sam3 import Sam3Processor

# SAM 3.1 uses identical preprocessing (1008x1008, same normalization)
Sam31Processor = Sam3Processor

install_auto_processor_patch(["sam3.1_video", "sam3_1"], Sam31Processor)
