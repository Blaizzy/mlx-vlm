import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Constants for image processing (from internvl_chat.py)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
# chat_template = get_conv_template("internvl2_5")
chat_template = "{% for message in messages %}{{message['role'].capitalize() + ': '}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['content'] }}{% endfor %}{{'\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}"

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_transform(input_size):
    """
    Builds a transformation pipeline for images.

    Args:
        input_size (int): The target size for the image (height and width).

    Returns:
        function: A function that takes a PIL image and returns a normalized mx.array.
    """
    mean = mx.array(IMAGENET_MEAN)
    std = mx.array(IMAGENET_STD)

    def transform(img: Image.Image) -> mx.array:
        # Ensure image is RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize using PIL - BICUBIC interpolation is default in Pillow >= 9.1.0 for resize
        # For older versions, you might need Pillow-SIMD or explicitly set
        # resampling=Image.BICUBIC if available.
        img = img.resize((input_size, input_size), resample=Image.Resampling.BICUBIC)

        # Convert PIL image to NumPy array (H, W, C) and scale to [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0

        # Convert to MLX array and transpose to (C, H, W)
        img_mx = mx.array(img_np).transpose(2, 0, 1)

        # Normalize
        img_mx = (img_mx - mean[:, None, None]) / std[:, None, None]

        return img_mx

    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio from a list of targets."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Prioritize ratios closer to the original image area if diffs are equal
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if abs(area - target_area) < abs(
                area - image_size * image_size * best_ratio[0] * best_ratio[1]
            ):
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """
    Preprocesses the image by splitting it into blocks based on the closest aspect ratio.

    Args:
        image (PIL.Image.Image): Input image.
        min_num (int): Minimum number of blocks.
        max_num (int): Maximum number of blocks.
        image_size (int): Target size for each block.
        use_thumbnail (bool): Whether to include a thumbnail of the original image.

    Returns:
        list[PIL.Image.Image]: A list of processed image blocks (as PIL images).
    """
    orig_width, orig_height = image.size
    if orig_width == 0 or orig_height == 0:
        # Handle potential zero dimensions
        return []
    aspect_ratio = orig_width / orig_height

    # Calculate the possible target aspect ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest target aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target dimensions for resizing
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image to fit the target block structure
    # Using BICUBIC resampling
    resized_img = image.resize(
        (target_width, target_height), resample=Image.Resampling.BICUBIC
    )

    processed_images = []
    # Crop the resized image into blocks
    for i in range(blocks):
        # Calculate crop box for the i-th block
        row_idx = i // target_aspect_ratio[0]
        col_idx = i % target_aspect_ratio[0]
        left = col_idx * image_size
        top = row_idx * image_size
        right = (col_idx + 1) * image_size
        bottom = (row_idx + 1) * image_size
        box = (left, top, right, bottom)

        # Crop and add the block
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert (
        len(processed_images) == blocks
    ), f"Expected {blocks} blocks, but got {len(processed_images)}"

    # Add a thumbnail if requested and if the image was split
    if use_thumbnail and blocks > 1:
        thumbnail_img = image.resize(
            (image_size, image_size), resample=Image.Resampling.BICUBIC
        )
        processed_images.append(thumbnail_img)

    return processed_images


class InternVLImageProcessor(ImageFeatureExtractionMixin):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: int = 448,  # Default image size from dynamic_preprocess
        resample=Image.Resampling.BICUBIC,
        do_center_crop: bool = False,  # Not used in original, but standard HF param
        crop_size=None,
        do_rescale: bool = True,  # Original code scales by 1/255.0
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean=IMAGENET_MEAN.tolist(),
        image_std=IMAGENET_STD.tolist(),
        do_dynamic_preprocess: bool = True,
        dynamic_min_num: int = 1,
        dynamic_max_num: int = 12,
        dynamic_use_thumbnail: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.do_resize = (
            do_resize  # Although dynamic_preprocess handles resizing internally
        )
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        # Custom dynamic processing params
        self.do_dynamic_preprocess = do_dynamic_preprocess
        self.dynamic_min_num = dynamic_min_num
        self.dynamic_max_num = dynamic_max_num
        self.dynamic_use_thumbnail = dynamic_use_thumbnail

    def preprocess(
        self,
        images: List[Image.Image],
        do_dynamic_preprocess: Optional[bool] = None,
        size: Optional[int] = None,
        # ... other params matching __init__ ...
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> List[mx.array]:

        do_dynamic_preprocess = (
            do_dynamic_preprocess
            if do_dynamic_preprocess is not None
            else self.do_dynamic_preprocess
        )
        size = size if size is not None else self.size
        # ... handle other overrides ...

        if not isinstance(images, list):
            images = [images]

        if not all(isinstance(image, Image.Image) for image in images):
            raise ValueError("Input must be a list of PIL Images.")

        processed_images_batch = []
        for image in images:
            # Apply dynamic preprocessing
            if do_dynamic_preprocess:
                processed_images = dynamic_preprocess(
                    image,
                    min_num=self.dynamic_min_num,
                    max_num=self.dynamic_max_num,
                    image_size=size,
                    use_thumbnail=self.dynamic_use_thumbnail,
                )
            else:
                # Fallback or alternative simpler preprocessing if needed
                # e.g., simple resize + normalize
                processed_images = [image.resize((size, size), resample=self.resample)]

            # Create transform function
            transform = build_transform(input_size=size)

            # Apply transform to each image block and collect arrays
            pixel_values_list = [transform(img) for img in processed_images]

            # Stack the arrays along a new dimension (batch dimension)
            pixel_values = mx.stack(pixel_values_list, axis=0)

            processed_images_batch.append(pixel_values)

        # At this point, processed_images_batch contains a list of mx arrays,
        # each array corresponding to an input image with stacked blocks.

        data = {"pixel_values": mx.array(processed_images_batch)}
        return BatchFeature(data=data, tensor_type=None)


class InternVLChatProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "InternVLImageProcessor"
    tokenizer_class = (
        "AutoTokenizer",
        "Qwen2TokenizerFast",
    )  # Specify possible classes

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=chat_template,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = InternVLImageProcessor(**kwargs)
        if isinstance(tokenizer, str):
            # Defaulting to the likely repo ID found earlier
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True, **kwargs
            )

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.num_image_token = int((448 // 14) ** 2 * (0.5**2))

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: List[Image.Image] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",  # Default to PyTorch tensors
        **kwargs,
    ):
        processed_inputs = {}
        if images is not None:
            image_features = self.image_processor.preprocess(
                images, return_tensors=return_tensors, **kwargs
            )
            processed_inputs.update(image_features)  # Should contain 'pixel_values'

        if text is not None:
            queries = []

            if isinstance(text, str):
                text = [text]

            for idx in range(len(images)):
                question = text[idx]

                if images is not None and "<image>" not in question:
                    question = "<image>\n" + question

                num_patches = image_features["pixel_values"][idx].shape[0]
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + IMG_END_TOKEN
                )
                question = question.replace("<image>", image_tokens, 1)
                queries.append(question)

            self.tokenizer.padding_side = "left"
            text_inputs = self.tokenizer(
                queries,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
            processed_inputs.update(text_inputs)  # 'input_ids', 'attention_mask'

        return processed_inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        pass

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        image_processor = InternVLImageProcessor(**kwargs)
        return InternVLChatProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )

    # Need save_pretrained and from_pretrained
    # save_pretrained should save both tokenizer and image_processor configs/files
    # from_pretrained should load both

    # Example:
    # def save_pretrained(self, save_directory, **kwargs):
    #     self.tokenizer.save_pretrained(save_directory, **kwargs)
    #     self.image_processor.save_pretrained(save_directory, **kwargs)

    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    #     image_processor = InternVLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    #     return cls(image_processor=image_processor, tokenizer=tokenizer)


# Registration
MODEL_TYPE = "internvl_chat"  # Verify this from the model's config.json

AutoImageProcessor.register(
    MODEL_TYPE, slow_image_processor_class=InternVLImageProcessor
)
AutoProcessor.register(MODEL_TYPE, InternVLChatProcessor)

logger.info(f"Registered custom processor classes for model type '{MODEL_TYPE}'.")
