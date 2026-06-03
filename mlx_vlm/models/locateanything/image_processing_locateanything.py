import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
from PIL import Image

_materialize = mx.eval

LOCATEANYTHING_IMAGE_MEAN = (0.5, 0.5, 0.5)
LOCATEANYTHING_IMAGE_STD = (0.5, 0.5, 0.5)


def _base_image_processor():
    from transformers.image_processing_utils import BaseImageProcessor

    return BaseImageProcessor


class LocateAnythingImageProcessor(_base_image_processor()):

    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: Tuple[float, float, float] = LOCATEANYTHING_IMAGE_MEAN,
        image_std: Tuple[float, float, float] = LOCATEANYTHING_IMAGE_STD,
        in_token_limit: int = 25600,
        merge_kernel_size: List[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = (
            merge_kernel_size if merge_kernel_size is not None else [2, 2]
        )

    def rescale(
        self, image: Image.Image, merge_kernel_size: List[int] = None
    ) -> Image.Image:
        if merge_kernel_size is None:
            merge_kernel_size = self.merge_kernel_size

        w, h = image.size
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(
                self.in_token_limit / ((w // patch_size) * (h // patch_size))
            )
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        new_w, new_h = image.size
        pad_w = merge_kernel_size[1] * patch_size
        pad_h = merge_kernel_size[0] * patch_size
        target_w = math.ceil(new_w / pad_w) * pad_w
        target_h = math.ceil(new_h / pad_h) * pad_h
        if (target_w, target_h) != (new_w, new_h):
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_mlx(self, image: Image.Image) -> mx.array:
        image = image.convert("RGB")
        w, h = image.size
        arr = mx.array(list(image.getdata()), dtype=mx.float32).reshape(h, w, 3) / 255.0
        return arr.transpose(2, 0, 1)

    def normalize(self, image: mx.array) -> mx.array:
        mean = mx.array(self.image_mean, dtype=mx.float32).reshape(3, 1, 1)
        std = mx.array(self.image_std, dtype=mx.float32).reshape(3, 1, 1)
        return (image - mean) / std

    def patchify(self, image: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        patch_size = self.patch_size
        C, H, W = image.shape

        patches = image.reshape(
            C, H // patch_size, patch_size, W // patch_size, patch_size
        )
        patches = patches.transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, C, patch_size, patch_size)

        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, mx.array):
            arr = image
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                arr = (arr * 255).astype(mx.uint8)
            h, w, _ = arr.shape
            return Image.frombytes("RGB", (w, h), bytes(arr.reshape(-1).tolist()))
        raise ValueError(
            f"Invalid image type {type(image)}. Expected PIL.Image.Image or mx.array."
        )

    def _preprocess(self, image) -> Tuple[mx.array, Tuple[int, int]]:
        image = self.rescale(image, self.merge_kernel_size)
        image = self.to_mlx(image)
        image = self.normalize(image)
        return self.patchify(image)

    def preprocess(
        self,
        images,
        return_tensors: Optional[Union[str, object]] = None,
        **kwargs,
    ):
        from transformers.feature_extraction_utils import BatchFeature

        if isinstance(images, (mx.array, Image.Image)):
            images = [images]

        pixel_values_list = []
        image_grid_hws = []

        for image in images:
            patches, image_grid_hw = self._preprocess(self._to_pil(image))
            pixel_values_list.append(patches)
            image_grid_hws.append(image_grid_hw)

        pixel_values = mx.concatenate(pixel_values_list, axis=0)
        grid_shapes = [(int(h), int(w)) for h, w in image_grid_hws]
        image_grid_hws = mx.array(image_grid_hws)
        _materialize(pixel_values, image_grid_hws)

        data = {
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
            "_grid_shapes": grid_shapes,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(
        self,
        images,
        return_tensors: Optional[Union[str, object]] = None,
        **kwargs,
    ):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)
