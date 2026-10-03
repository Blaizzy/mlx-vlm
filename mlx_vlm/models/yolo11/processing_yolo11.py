"""Image preparation for the YOLO11 detector."""

from ..base import install_auto_processor_patch
from .inference import prepare_image


class YOLO11Processor:
    """Letterboxes an image to a stride-aligned detector input."""

    def __init__(self, size=None, stride: int = 32):
        self.size = size
        self.stride = stride

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, image, **kwargs):
        return prepare_image(image, size=self.size, stride=self.stride)


install_auto_processor_patch("yolo11", YOLO11Processor)
