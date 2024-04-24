import glob
import inspect
import json
import re
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterable, Any, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import to_numpy_array

from ..base import BaseImageProcessor
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel

from .image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from .image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)

import PIL

@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_hidden_size: int
    mm_vision_tower: str
    mm_projector_type: str = "mlp2x_gelu"
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def get_resize_output_image_size(image, size, input_data_format) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        The output size of the image after resizing.
    """
    height, width = get_image_size(image, channel_dim=input_data_format)

    min_len = size["shortest_edge"]
    max_len = size["longest_edge"]
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def make_list_of_images(images: ImageInput) -> List[List[np.ndarray]]:
    """
    Convert a single image or a list of images to a list of numpy arrays.

    Args:
        images (`ImageInput`):
            A single image or a list of images.

    Returns:
        A list of numpy arrays.
    """
    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        images = [[images]]
    # If it's a list of images, it's a single batch, so convert it to a list of lists
    elif isinstance(images, (list, tuple)) and len(images) > 0 and is_valid_image(images[0]):
        images = [images]
    # If it's a list of batches, it's already in the right format
    elif (
        isinstance(images, (list, tuple))
        and len(images) > 0
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        pass
    else:
        raise ValueError(
            "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
        )
    return images


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(
    images_list: List[List[np.ndarray]], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images_list[0][0])

    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(get_image_size(image, channel_dim=input_data_format))

    max_height, max_width = max_across_indices(image_sizes)
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# FIXME Amy: merge this function with the one in image_transforms.py
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    if not isinstance(image, PIL.Image.Image):
        return image

    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class ImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        do_image_splitting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 378, "longest_edge": 980}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.do_image_splitting = do_image_splitting

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(image, size, input_data_format)
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "size must be a dictionary with keys 'shortest_edge' and 'longest_edge' or 'height' and 'width'."
            )
        return resize(
            image, size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        For a list of images, for each images, pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width.
        For each sample in the batch, pads the sample with empty images to the max_number of images per sample in the batch. Optionally returns a pixel mask.

        Args:
            images (`np.ndarray`):
                List of list of images to pad. Pads to the largest height and width in the batch.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        batch_size = len(images)
        max_num_images = max(len(images_) for images_ in images)
        input_data_format = (
            infer_channel_dimension_format(images[0][0]) if input_data_format is None else input_data_format
        )
        data_format = input_data_format if data_format is None else data_format

        def empty_image(size, input_data_format):
            if input_data_format == ChannelDimension.FIRST:
                return np.zeros((3, *size), dtype=np.uint8)
            elif input_data_format == ChannelDimension.LAST:
                return np.zeros((*size, 3), dtype=np.uint8)
            raise ValueError("Invalid channel dimension format.")

        padded_images_list = [
            [empty_image(pad_size, data_format) for _ in range(max_num_images)] for _ in range(batch_size)
        ]
        padded_masks = [[np.zeros(pad_size) for _ in range(max_num_images)] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for sample_idx, image in enumerate(images[batch_idx]):
                padded_images_list[batch_idx][sample_idx] = self._pad_image(
                    image,
                    pad_size,
                    constant_values=constant_values,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                padded_masks[batch_idx][sample_idx] = make_pixel_mask(
                    image, output_size=pad_size, input_data_format=input_data_format
                )

        padded_masks = padded_masks if return_pixel_mask else None
        return padded_images_list, padded_masks

    def _crop(
        self,
        im: np.ndarray,
        w1: int,
        h1: int,
        w2: int,
        h2: int,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        if input_data_format == ChannelDimension.FIRST:
            return im[:, h1:h2, w1:w2]
        elif input_data_format == ChannelDimension.LAST:
            return im[h1:h2, w1:w2, :]

    def split_image(
        self,
        image: np.ndarray,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Split an image into 4 equal sub-images, and the concatenate that sequence with the original image.
        That means that a single image becomes a sequence of 5 images.
        This is a "trick" to spend more compute on each image with no changes in the vision encoder.

        Args:
            image (`np.ndarray`):
                Images to split.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image, input_data_format)

        mid_width = width // 2
        mid_height = height // 2
        return [
            self._crop(image, 0, 0, mid_width, mid_height, input_data_format),
            self._crop(image, mid_width, 0, width, mid_height, input_data_format),
            self._crop(image, 0, mid_height, mid_width, height, input_data_format),
            self._crop(image, mid_width, mid_height, width, height, input_data_format),
            image,
        ]

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_image_splitting: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ):
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether or not to pad the images to the largest height and width in the batch.
            do_image_splitting (`bool`, *optional*, defaults to `self.do_image_splitting`):
                Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
                strategy was first introduced in https://arxiv.org/abs/2311.06607.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.do_image_splitting

        images_list = make_list_of_images(images)

        if not valid_images(images_list[0]):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images_list = [[convert_to_rgb(image) for image in images] for images in images_list]

        # All transformations expect numpy arrays.
        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        if is_scaled_image(images_list[0][0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images_list[0][0])

        if do_image_splitting:
            new_images_list = []
            for images in images_list:
                new_images = []
                for image in images:
                    new_images.extend(self.split_image(image, input_data_format))
                new_images_list.append(new_images)
            images_list = new_images_list

        if do_resize:
            images_list = [
                [
                    self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_rescale:
            images_list = [
                [
                    self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_normalize:
            images_list = [
                [
                    self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        pixel_attention_mask = None
        if do_pad:
            images_list, pixel_attention_mask = self.pad(
                images_list, return_pixel_mask=True, return_tensors=return_tensors, input_data_format=input_data_format
            )

        if data_format is not None:
            images_list = [
                [
                    to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        data = {"pixel_values": np.array(images_list) if do_pad else images_list}  # Faster tensor conversion
        if pixel_attention_mask is not None:
            data["pixel_attention_mask"] = np.array(pixel_attention_mask) if do_pad else pixel_attention_mask

        return data


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class SigLipVisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_tower(x, output_hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = SigLipVisionTower(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = LlavaMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        *_, hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = hidden_state[-1].astype(pixel_values.dtype)
        assert image_features.shape[-2] == 729

        image_features = self.mm_projector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

        if len(image_positions) != num_images:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f" match the number of image inputs ({num_images})."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(self, input_ids: mx.array, pixel_values: mx.array, cache=None):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        siglip_config = AutoConfig.from_pretrained(config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(config["language_model"])
        siglip_config = siglip_config.to_dict()
        text_config = text_config.to_dict()
        config["vision_config"] = siglip_config["vision_config"]
        config["text_config"] = text_config

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.vision_tower", k)
                else (
                    f"mm_projector.linear_1.{k.split('.')[-1]}"
                    if re.match(r"^model\.mm_projector\.0", k)
                    else (
                        f"mm_projector.linear_2.{k.split('.')[-1]}"
                        if re.match(r"^model\.mm_projector\.2", k)
                        else (
                            f"language_model.model.{k}"
                            if re.match(r"^lm_head", k)
                            else (
                                f"language_model.{k}"
                                if re.match(r"^model\.(embed_tokens|norm|layers)", k)
                                else k
                            )
                        )
                    )
                )
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"vision_tower.vision_tower.vision_model.head.attention.in_proj.bias"
                if re.match(
                    r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_bias",
                    k,
                )
                else (
                    f"vision_tower.vision_tower.vision_model.head.attention.in_proj.weight"
                    if re.match(
                        r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_weight",
                        k,
                    )
                    else k
                )
            ): v
            for k, v in weights.items()
        }

        return weights


 python -m mlx_lm.convert --hf-path Qwen/CodeQwen1.5-7B -q --upload-repo mlx-community/CodeQwen1.5-7B-4bit


Skip to left side bar




Filter files by name
/OpenELM03B/
Name
Last Modified
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

# this import has to be relative, otherwise, when setting trust_remote_code=True
# huggingface transformers won't be able to load the module correctly
from .configuration_openelm import OpenELMConfig, make_divisible


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying OpenELMRMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"



#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

# this import has to be relative, otherwise, when setting trust_remote_code=True
# huggingface transformers won't be able to load the module correctly
from .configuration_openelm import OpenELMConfig, make_divisible


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying OpenELMRMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"
        )


class OpenELMPreTrainedModel(PreTrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, OpenELMRMSNorm):
            module.weight.data.fill_(1.0)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_device: torch.device = torch.device("cpu"),
        key_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.

        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            device: Device where the key embeddings are stored.
            key_dtype: Data type of the key embeddings.

        Returns:
            None

        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None and self._cached_cos.device != key_device)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None and self._cached_sin.device != key_device)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = torch.arange(
                self._cached_seq_length,
                dtype=torch.float32,
                device=self.inv_freq.device,
            )
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            pos_index_theta = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)

            # the shape of cos and sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.cos().to(dtype=key_dtype, device=key_device)
            sin_emb = emb.sin().to(dtype=key_dtype, device=key_device)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE embeddings.

        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].

        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.

        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.device == query.device
        assert key.dtype == query.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."

        query_float = query.float()
        key_float = key.float()

        self._compute_sin_cos_embeddings(
            key_len, key_device=key_float.device, key_dtype=key_float.dtype
        )
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.type_as(query), key_float.type_as(key)


class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of multi-head self-attention.

        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        batch_size, seq_length, d_model = hidden_states.size()

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(1, 2)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(
                keys, values, self.layer_idx, cache_kwargs
            )

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            values = values.repeat_interleave(self.num_groups, dim=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OpenELMModel(OpenELMPreTrainedModel):
    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(
            embedding_dim=config.model_dim,
            num_embeddings=config.vocab_size,
        )

        self.layers = nn.ModuleList(
            OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_transformer_layers)
        )
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                in_features=config.model_dim,
                out_features=config.vocab_size,
                bias=False,
            )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_context_length`.
        causal_mask = torch.full(
            (config.max_context_length, config.max_context_length),
            fill_value=True,
            dtype=torch.bool,
        )
        self.register_buffer(
            "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
        )

        # Initialize weights and apply final processing
        self.post_init()
        self.reset_parameters(config=config)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.token_embeddings = new_embeddings

    def reset_parameters(self, config: OpenELMConfig) -> None:
        """Initialize the layers in Language Model

        The initialization scheme is followed, following `OPT <https://arxiv.org/pdf/2205.01068.pdf>`_.

        Args:
            use_megatron_std: Use standard deviation as described in Megatron-LM.

        Returns:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = module.in_features**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                std = module.embedding_dim**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, OpenELMRMSNorm):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        model_dim = config.model_dim
        n_layers = config.num_transformer_layers
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight") or param_name.endswith(
                "ffn.proj_2.weight"
            ):
                torch.nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]),
                fill_value=1,
            )
            self.register_buffer(
                "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
            )

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min
        causal_mask = (
            self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype)
            * min_dtype
        )

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            # For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = causal_mask.mul(
                    ~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)
                ).to(dtype)

        return causal_mask


class OpenELMForCausalLM(OpenELMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.lm_head is None:
            # shared
            logits = F.linear(
                hidden_states, weight=self.transformer.token_embeddings.weight
            )
        else:
            logits = self.lm_head(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if self.generation_config.cache_implementation == "static":
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(
            past_length,
            past_length + position_ids.shape[-1],
            device=position_ids.device,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # We could use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
