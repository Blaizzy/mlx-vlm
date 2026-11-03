# Adapted from speridlabs/idol-sdk (Apache-2.0).
import numpy as np
from PIL import Image, ImageFilter

# How much room the square crop leaves around the object: crop side = object bounding box x margin, so the
# object spans 1 / margin of the image. Renders of single objects work best tight; photos with a looser crop.
CROP_MARGIN = {"synthetic": 1.2, "real": 1.7}


def has_real_alpha(image: Image.Image) -> bool:
    return (
        image.mode == "RGBA"
        and np.array(image.getchannel(3), dtype=np.int32).min() < 255
    )


def preprocess_image(
    image: Image.Image,
    background_remover,
    image_size: int = 1024,
    crop_margin: float = 1.2,
) -> Image.Image:
    """Return the RGB image the encoders see.

    Resize so the short side is ``image_size``; if the image has no real alpha, ``background_remover`` (BiRefNet)
    predicts one; erode the alpha by 1 px so no background-coloured fringe survives; crop a square of
    side ``crop_margin`` x the object's bounding box; resize to ``image_size``; composite on black.
    """
    if crop_margin <= 0 or not np.isfinite(crop_margin):
        raise ValueError("crop_margin must be positive and finite")
    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.LANCZOS,
    )
    if not has_real_alpha(image):
        if background_remover is None:
            raise ValueError(
                "An RGB input needs background removal; pass an RGBA image or enable BiRefNet."
            )
        image = background_remover.remove_background(image.convert("RGB"))
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise ValueError("no object found in the image (the alpha channel is empty)")
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(1, right - left, bottom - top) / 2 * crop_margin
    image = image.crop(
        [
            int(center_x - half),
            int(center_y - half),
            int(center_x + half),
            int(center_y + half),
        ]
    )
    image = image.resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite
