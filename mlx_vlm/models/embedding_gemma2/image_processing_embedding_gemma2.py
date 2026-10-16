"""NumPy/Pillow patch preprocessing for EmbeddingGemma 2 images and frames."""

import copy

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
)

from ..gemma4.processing_gemma4 import (
    _SUPPORTED_SOFT_TOKENS,
    Gemma4ImageProcessor,
    _convert_to_rgb,
    _convert_video_to_patches,
    _pad_video_patches,
    _to_channel_first,
    get_aspect_ratio_preserving_size,
)


class EmbeddingGemma2ImageProcessor(Gemma4ImageProcessor):
    model_input_names = [
        "pixel_values",
        "image_position_ids",
        "num_soft_tokens_per_image",
    ]

    def __init__(self, **kwargs):
        kwargs.setdefault("image_mean", [0.0, 0.0, 0.0])
        kwargs.setdefault("image_std", [1.0, 1.0, 1.0])
        super().__init__(**kwargs)
        self._validate_budget(self.max_soft_tokens)

    @staticmethod
    def _validate_budget(max_soft_tokens):
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )

    def preprocess(self, images, return_tensors=None, input_data_format=None, **kwargs):
        # Per-call options must not change saved processor settings.
        processor = copy.copy(self)
        for name in (
            "patch_size",
            "max_soft_tokens",
            "pooling_kernel_size",
            "do_convert_rgb",
            "do_resize",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
        ):
            if kwargs.get(name) is not None:
                setattr(processor, name, kwargs[name])
        processor._validate_budget(processor.max_soft_tokens)
        max_patches = processor.max_soft_tokens * processor.pooling_kernel_size**2
        pixels, positions, counts = [], [], []
        for image in make_flat_list_of_images(images):
            # PIL stores pixels height-major, so its layout is known before
            # conversion. Inferring it from shape mistakes a 3-pixel-tall RGB
            # image, (3, W, 3), for channels-first and cannot handle a
            # 1-pixel-tall one at all.
            is_pil = isinstance(image, Image.Image)
            if processor.do_convert_rgb:
                image = _convert_to_rgb(image)
            image = to_numpy_array(image)
            if image.ndim == 2:
                image = image[None]
                layout = ChannelDimension.FIRST
            elif input_data_format is not None:
                layout = input_data_format
            elif is_pil:
                layout = ChannelDimension.LAST
            else:
                layout = infer_channel_dimension_format(image)
            image = _to_channel_first(image, layout)
            if processor.do_resize:
                height, width = image.shape[-2:]
                target_h, target_w = get_aspect_ratio_preserving_size(
                    height,
                    width,
                    processor.patch_size,
                    max_patches,
                    processor.pooling_kernel_size,
                )
                if (target_h, target_w) != (height, width):
                    # Floating inputs retain their range and precision; converting
                    # them to uint8 would corrupt already-rescaled images.
                    if np.issubdtype(image.dtype, np.floating):
                        image = np.stack(
                            [
                                np.asarray(
                                    Image.fromarray(channel.astype(np.float32)).resize(
                                        (target_w, target_h), processor.resample
                                    )
                                )
                                for channel in image
                            ]
                        )
                    else:
                        array = image.transpose(1, 2, 0)
                        if array.shape[-1] == 1:
                            array = array[..., 0]
                        array = np.asarray(
                            Image.fromarray(array).resize(
                                (target_w, target_h), processor.resample
                            )
                        )
                        image = (
                            array[None] if array.ndim == 2 else array.transpose(2, 0, 1)
                        )
            image = image.astype(np.float32)
            if processor.do_rescale:
                image *= processor.rescale_factor
            if processor.do_normalize:
                mean = np.asarray(processor.image_mean, dtype=np.float32)
                std = np.asarray(processor.image_std, dtype=np.float32)
                image = (image - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
            patches, ids = _convert_video_to_patches(image[None], processor.patch_size)
            counts.append(patches.shape[1] // processor.pooling_kernel_size**2)
            # Like the source, pad short grids without truncating oversized ones.
            patches, ids = _pad_video_patches(
                patches, ids, max(max_patches, patches.shape[1])
            )
            pixels.append(patches[0])
            positions.append(ids[0])
        return BatchFeature(
            {
                "pixel_values": np.stack(pixels),
                "image_position_ids": np.stack(positions),
                "num_soft_tokens_per_image": counts,
            },
            tensor_type=return_tensors,
        )
