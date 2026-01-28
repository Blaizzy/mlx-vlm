"""
MLX-based Phi3V Processor.

This module provides an MLX-native processor for Phi-3.5-Vision models that:
1. Uses HuggingFace tokenizer (no custom dependencies)
2. Provides an MLX-based image processor (no torch/torchvision dependency)
3. Handles dynamic resolution with HD image processing
"""

import json
import math
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import transformers.processing_utils as processing_utils
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType


def _validate_images_text_input_order(images, text):
    """
    Validate and potentially swap the order of images and text arguments.
    """
    if images is not None and text is not None:
        images_is_text = isinstance(images, str) or (
            isinstance(images, (list, tuple))
            and len(images) > 0
            and isinstance(images[0], str)
        )
        text_is_image = not isinstance(text, str) and not (
            isinstance(text, (list, tuple))
            and len(text) > 0
            and isinstance(text[0], str)
        )

        if images_is_text and text_is_image:
            warnings.warn(
                "You passed text as the first argument and images as the second. "
                "This is deprecated and will be removed in a future version. "
                "Please pass images first and text second.",
                FutureWarning,
            )
            return text, images

    return images, text


# Add the function to transformers.processing_utils if it doesn't exist
if not hasattr(processing_utils, "_validate_images_text_input_order"):
    processing_utils._validate_images_text_input_order = (
        _validate_images_text_input_order
    )

# Also add Unpack if it doesn't exist (for older Python versions)
if not hasattr(processing_utils, "Unpack"):
    try:
        from typing import Unpack

        processing_utils.Unpack = Unpack
    except ImportError:
        from typing_extensions import Unpack

        processing_utils.Unpack = Unpack


# CLIP-style normalization constants (same as OpenAI CLIP)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _calc_padded_size(width: int, height: int, padding_unit: int = 336):
    """Calculate the padded size to be divisible by padding_unit."""
    target_height = math.ceil(height / padding_unit) * padding_unit
    target_width = math.ceil(width / padding_unit) * padding_unit
    return target_width, target_height


def _calc_hd_transform_size(width: int, height: int, hd_num: int = 16):
    """
    Calculate the HD transform size for dynamic resolution.
    Phi-3.5 uses a 336x336 base size and supports up to hd_num tiles.
    """
    transposed = False
    if width < height:
        width, height = height, width
        transposed = True

    ratio = width / height
    scale = 1
    while scale * math.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1

    new_width = int(scale * 336)
    new_height = int(new_width / ratio)

    # Make dimensions divisible by 336
    padded_width, padded_height = _calc_padded_size(new_width, new_height, 336)

    if transposed:
        padded_width, padded_height = padded_height, padded_width

    return padded_width, padded_height


def _hd_transform(img: Image.Image, hd_num: int = 16) -> Image.Image:
    """
    Apply HD transform to resize image for dynamic resolution.
    """
    width, height = img.size
    target_width, target_height = _calc_hd_transform_size(width, height, hd_num)
    return img.resize((target_width, target_height), Image.Resampling.BICUBIC)


def _pad_to_336(img: Image.Image) -> Image.Image:
    """
    Pad image dimensions to be divisible by 336.
    """
    width, height = img.size
    target_width = math.ceil(width / 336) * 336
    target_height = math.ceil(height / 336) * 336

    if target_width == width and target_height == height:
        return img

    # Create new image with black background
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img


class Phi3VImageProcessor(BaseImageProcessor):
    """
    Image processor for Phi-3.5-Vision models.

    Processes images using HD dynamic resolution with 336x336 tiles,
    similar to the official Phi-3.5-Vision implementation.
    """

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
        image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
        num_crops: int = 4,
        num_img_tokens: int = 144,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_mean = image_mean
        self.image_std = image_std
        self.num_crops = num_crops
        self.num_img_tokens = num_img_tokens
        self.img_size = 336

    def calc_num_image_tokens(self, image: Image.Image) -> int:
        """
        Calculate the number of image tokens for a given image.
        """
        width, height = image.size
        hd_width, hd_height = _calc_hd_transform_size(width, height, self.num_crops)
        num_h_tiles = hd_height // self.img_size
        num_w_tiles = hd_width // self.img_size
        # Global image tokens + sub-image tokens + separators
        num_tokens = (
            (num_h_tiles * num_w_tiles + 1) * self.num_img_tokens
            + 1
            + (num_h_tiles + 1) * 12
        )
        return num_tokens

    def _process_single_image(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Process a single image with HD transform and normalize.

        Returns:
            pixel_values: numpy array of shape (num_tiles + 1, C, H, W)
            image_size: (height, width) of the HD transformed image
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply HD transform
        hd_image = _hd_transform(image, self.num_crops)
        hd_image = _pad_to_336(hd_image)
        hd_width, hd_height = hd_image.size

        # Create global image (resized to 336x336)
        global_image = hd_image.resize(
            (self.img_size, self.img_size), Image.Resampling.BICUBIC
        )

        # Split HD image into 336x336 tiles
        num_h_tiles = hd_height // self.img_size
        num_w_tiles = hd_width // self.img_size

        tiles = []
        for h in range(num_h_tiles):
            for w in range(num_w_tiles):
                left = w * self.img_size
                top = h * self.img_size
                right = left + self.img_size
                bottom = top + self.img_size
                tile = hd_image.crop((left, top, right, bottom))
                tiles.append(tile)

        # Global image first, then tiles
        all_images = [global_image] + tiles

        # Convert to numpy arrays and normalize
        processed = []
        for img in all_images:
            arr = np.array(img, dtype=np.float32) / 255.0
            # Normalize
            arr = (arr - np.array(self.image_mean)) / np.array(self.image_std)
            # HWC to CHW
            arr = arr.transpose(2, 0, 1)
            processed.append(arr)

        pixel_values = np.stack(processed, axis=0)  # (num_tiles + 1, C, H, W)
        image_size = (hd_height, hd_width)

        return pixel_values, image_size

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """Process images and return BatchFeature."""
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image or similar."
            )

        all_pixel_values = []
        all_image_sizes = []

        for image in images:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            pixel_values, image_size = self._process_single_image(image)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_size)

        # Stack with padding to handle variable number of tiles
        max_tiles = max(pv.shape[0] for pv in all_pixel_values)
        batch_size = len(all_pixel_values)

        # Pad to same number of tiles
        padded_pixel_values = []
        for pv in all_pixel_values:
            if pv.shape[0] < max_tiles:
                padding = np.zeros(
                    (max_tiles - pv.shape[0], *pv.shape[1:]), dtype=pv.dtype
                )
                pv = np.concatenate([pv, padding], axis=0)
            padded_pixel_values.append(pv)

        pixel_values = np.stack(padded_pixel_values, axis=0)  # (B, T, C, H, W)
        image_sizes = np.array(all_image_sizes)  # (B, 2)

        data = {
            "pixel_values": mx.array(pixel_values),
            "image_sizes": mx.array(image_sizes),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """Make the image processor callable."""
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


class Phi3VProcessor(ProcessorMixin):
    """
    MLX-based processor for Phi-3.5-Vision that doesn't require torch/torchvision.

    Constructs a Phi3V processor which wraps a Phi3V image processor and a tokenizer
    into a single processor.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Phi3VImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = Phi3VImageProcessor()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _convert_images_texts_to_inputs(
        self,
        images: List[Image.Image],
        texts: str,
        padding: bool = False,
        truncation: bool = None,
        max_length: int = None,
    ) -> BatchFeature:
        """
        Convert images and text to model inputs, replacing image tokens with negative IDs.

        The Phi3V model expects image tokens to be represented as negative values in input_ids.
        For example, <|image_1|> becomes a sequence of -1 values, <|image_2|> becomes -2 values.
        """
        # Pattern to match image tokens like <|image_1|>, <|image_2|>, etc.
        pattern = r"<\|image_\d+\|>"

        # Process images first to get their sizes and calculate token counts
        if images:
            images = make_list_of_images(images)
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)

            # Calculate number of tokens for each image
            num_img_tokens = [
                self.image_processor.calc_num_image_tokens(img) for img in pil_images
            ]

            # Process images through image processor
            image_inputs = self.image_processor(pil_images)
        else:
            pil_images = []
            num_img_tokens = []
            image_inputs = {}

        # Find image tags and extract their IDs
        image_tags = re.findall(pattern, texts)

        if image_tags:
            # Extract image IDs from tags (e.g., <|image_1|> -> 1)
            image_ids = [int(tag.split("|")[1].split("_")[-1]) for tag in image_tags]

            # Validate: unique image IDs should be sequential starting from 1
            unique_ids = sorted(set(image_ids))
            if unique_ids != list(range(1, len(unique_ids) + 1)):
                raise ValueError(
                    f"Image IDs must be sequential starting from 1. Got: {unique_ids}"
                )

            # Validate: number of unique image IDs should match number of images
            if len(unique_ids) != len(pil_images):
                raise ValueError(
                    f"Number of image tags ({len(unique_ids)}) doesn't match "
                    f"number of images ({len(pil_images)})"
                )

            # Create padded negative IDs for each image tag
            # Each <|image_N|> is replaced with num_img_tokens[N-1] copies of -N
            image_ids_pad = [[-iid] * num_img_tokens[iid - 1] for iid in image_ids]

            # Split text by image pattern and tokenize each chunk
            text_chunks = texts.split("<|image_")

            # Reconstruct the split to handle the pattern properly
            prompt_chunks = []
            for i, chunk in enumerate(re.split(pattern, texts)):
                tokens = self.tokenizer.encode(chunk, add_special_tokens=(i == 0))
                prompt_chunks.append(tokens)

            # Interleave text chunks with image token sequences
            input_ids = []
            img_idx = 0
            for i, chunk in enumerate(prompt_chunks):
                # Add text tokens (skip BOS if not first chunk)
                offset = 0 if i == 0 else 1  # Skip BOS token for subsequent chunks
                if i > 0 and len(chunk) > 0 and chunk[0] == self.tokenizer.bos_token_id:
                    offset = 1
                input_ids.extend(chunk[offset:])

                # Add image tokens if there's a corresponding image
                if img_idx < len(image_ids_pad):
                    input_ids.extend(image_ids_pad[img_idx])
                    img_idx += 1
        else:
            # No image tokens, just tokenize normally
            input_ids = self.tokenizer.encode(texts)

        # Create attention mask (all tokens including negative IDs are attended to)
        attention_mask = [1] * len(input_ids)

        text_inputs = {
            "input_ids": mx.array([input_ids]),
            "attention_mask": mx.array([attention_mask]),
        }

        return BatchFeature(data={**text_inputs, **image_inputs})

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            images: The image or batch of images to be prepared.
            text: The sequence or batch of sequences to be encoded.
            return_tensors: If set, will return tensors of a particular framework.

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, and image_sizes.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # Check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        # Extract return_tensors from kwargs (unused, we always return MLX arrays)
        kwargs.pop("return_tensors", None)
        padding = kwargs.pop("padding", False)
        truncation = kwargs.pop("truncation", None)
        max_length = kwargs.pop("max_length", None)

        # Convert to list if single text
        if isinstance(text, str):
            texts = [text]
        elif text is not None:
            texts = list(text)
        else:
            texts = None

        # Convert images to list if needed
        if images is not None:
            if not isinstance(images, list):
                images = [images]
        else:
            images = []

        # Process images and text together (handles image token replacement)
        if texts is not None:
            # For now, handle single text input (batching can be added later)
            if len(texts) == 1:
                return self._convert_images_texts_to_inputs(
                    images=images,
                    texts=texts[0],
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                )
            else:
                # Batch processing: process each text separately and combine
                all_input_ids = []
                all_attention_masks = []
                all_pixel_values = []
                all_image_sizes = []

                for txt in texts:
                    result = self._convert_images_texts_to_inputs(
                        images=images,
                        texts=txt,
                        padding=padding,
                        truncation=truncation,
                        max_length=max_length,
                    )
                    all_input_ids.append(result["input_ids"][0].tolist())
                    all_attention_masks.append(result["attention_mask"][0].tolist())
                    if "pixel_values" in result:
                        all_pixel_values.append(result["pixel_values"])
                    if "image_sizes" in result:
                        all_image_sizes.append(result["image_sizes"])

                # Pad input_ids and attention_masks to same length
                max_len = max(len(ids) for ids in all_input_ids)
                pad_token_id = self.tokenizer.pad_token_id or 0

                padded_input_ids = []
                padded_attention_masks = []
                for ids, mask in zip(all_input_ids, all_attention_masks):
                    padding_length = max_len - len(ids)
                    padded_input_ids.append(ids + [pad_token_id] * padding_length)
                    padded_attention_masks.append(mask + [0] * padding_length)

                data = {
                    "input_ids": mx.array(padded_input_ids),
                    "attention_mask": mx.array(padded_attention_masks),
                }

                if all_pixel_values:
                    data["pixel_values"] = all_pixel_values[
                        0
                    ]  # Same images for all texts
                if all_image_sizes:
                    data["image_sizes"] = all_image_sizes[0]

                return BatchFeature(data=data)

        # Text-only case
        if images:
            image_inputs = self.image_processor(images)
        else:
            image_inputs = {}

        return BatchFeature(data=image_inputs)

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        """Apply chat template to the conversation."""
        if chat_template is None:
            chat_template = self.chat_template
        if chat_template is None:
            chat_template = getattr(self.tokenizer, "chat_template", None)

        if chat_template is None:
            raise ValueError(
                "No chat template found. Please provide a chat_template argument "
                "or ensure the tokenizer has a chat_template attribute."
            )

        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("jinja2 is required for apply_chat_template")

        template = Template(chat_template)
        rendered = template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.tokenizer.encode(rendered)
        return rendered

    @property
    def model_input_names(self):
        """Get the model input names from tokenizer and image processor."""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the processor from a pretrained model path."""
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Load image processor config
        image_processor_config = {}
        try:
            if is_local:
                config_path = model_path / "preprocessor_config.json"
            else:
                config_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "preprocessor_config.json"
                    )
                )
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    preprocessor_config = json.load(f)
                if "num_crops" in preprocessor_config:
                    image_processor_config["num_crops"] = preprocessor_config[
                        "num_crops"
                    ]
                if "num_img_tokens" in preprocessor_config:
                    image_processor_config["num_img_tokens"] = preprocessor_config[
                        "num_img_tokens"
                    ]
                if "image_mean" in preprocessor_config:
                    image_processor_config["image_mean"] = tuple(
                        preprocessor_config["image_mean"]
                    )
                if "image_std" in preprocessor_config:
                    image_processor_config["image_std"] = tuple(
                        preprocessor_config["image_std"]
                    )
        except Exception:
            pass

        image_processor = Phi3VImageProcessor(**image_processor_config)

        # Load chat template from jinja file if not already set on tokenizer
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            try:
                if is_local:
                    jinja_path = model_path / "chat_template.jinja"
                else:
                    jinja_path = Path(
                        hf_hub_download(
                            pretrained_model_name_or_path, "chat_template.jinja"
                        )
                    )
                if jinja_path.exists():
                    chat_template = jinja_path.read_text(encoding="utf-8")
                    tokenizer.chat_template = chat_template
            except Exception:
                pass

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


# Register the processor with AutoProcessor
from transformers import AutoProcessor

_original_auto_processor_from_pretrained = AutoProcessor.from_pretrained


@classmethod
def _patched_auto_processor_from_pretrained(
    cls, pretrained_model_name_or_path, **kwargs
):
    """Patched from_pretrained that returns Phi3VProcessor for phi3_v models."""
    from huggingface_hub import hf_hub_download

    model_path = Path(pretrained_model_name_or_path)
    is_local = model_path.exists() and model_path.is_dir()

    # Check if this is a phi3_v model
    is_phi3_v = False
    try:
        if is_local:
            config_path = model_path / "config.json"
        else:
            config_path = Path(
                hf_hub_download(pretrained_model_name_or_path, "config.json")
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        is_phi3_v = model_type in ("phi3_v", "phi3-v", "phi3v")
    except Exception:
        pass

    if is_phi3_v:
        return Phi3VProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

    return _original_auto_processor_from_pretrained.__func__(
        cls, pretrained_model_name_or_path, **kwargs
    )


AutoProcessor.from_pretrained = _patched_auto_processor_from_pretrained
