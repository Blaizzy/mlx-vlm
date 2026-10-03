"""Image preparation for SAM 3D Body."""

from ..base import install_auto_processor_patch
from .batch_prep import prepare_image


class SAM3DBodyProcessor:
    """Crops an image to a person box and normalizes it for the estimator."""

    def __init__(self, image_size=(512, 384)):
        self.image_size = image_size

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, image, bbox=None, **kwargs):
        if bbox is None:
            height, width = image.shape[:2]
            bbox = [0, 0, width, height]
        kwargs.setdefault("image_size", self.image_size)
        return prepare_image(image, bbox, **kwargs)


install_auto_processor_patch("sam3d_body", SAM3DBodyProcessor)
