"""
MLX-based KimiVL Processor.

This module provides an MLX-native processor for KimiVL models that:
1. Uses a pre-converted fast tokenizer (no tiktoken dependency)
2. Provides an MLX-based image processor (no torch/torchvision dependency)
3. Patches missing functions for transformers 5.0 compatibility
"""

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import transformers.processing_utils as processing_utils
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from .config import ModelConfig


def _validate_images_text_input_order(images, text):
    """
    Validate and potentially swap the order of images and text arguments.

    This function checks if the arguments are in the correct order (images first, text second)
    for backward compatibility. If text is passed as the first argument and images as the second,
    it swaps them and issues a deprecation warning.

    Args:
        images: The images argument (should be image-like objects or None)
        text: The text argument (should be strings or None)

    Returns:
        Tuple of (images, text) in the correct order
    """
    # Check if arguments are swapped (text passed as images, images passed as text)
    if images is not None and text is not None:
        # If 'images' looks like text and 'text' looks like images, swap them
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


# CLIP-style normalization constants
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class KimiVLImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(
        self,
        patch_size: int = 14,
        pad_input: bool = False,
        image_mean: Tuple[float, float, float] = OPENAI_DATASET_MEAN,
        image_std: Tuple[float, float, float] = OPENAI_DATASET_STD,
        in_token_limit: int = 4096,
        merge_kernel_size: List[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.pad_input = pad_input
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = (
            merge_kernel_size if merge_kernel_size is not None else [2, 2]
        )

    def rescale(
        self, image: Image.Image, merge_kernel_size: List[int] = None
    ) -> Image.Image:
        """Rescale image to fit within token limits and pad/crop to patch boundaries."""
        if merge_kernel_size is None:
            merge_kernel_size = self.merge_kernel_size

        w, h = image.size
        patch_size = self.patch_size

        # Rescale if exceeds token limit
        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(
                self.in_token_limit / ((w // patch_size) * (h // patch_size))
            )
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        if self.pad_input:
            new_w, new_h = image.size
            pad_size_h = merge_kernel_size[0] * patch_size
            pad_size_w = merge_kernel_size[1] * patch_size

            pad_h = (pad_size_h - new_h % pad_size_h) % pad_size_h
            pad_w = (pad_size_w - new_w % pad_size_w) % pad_size_w

            if pad_h > 0 or pad_w > 0:
                # Pad image (bottom and right padding)
                new_image = Image.new(
                    image.mode, (new_w + pad_w, new_h + pad_h), (0, 0, 0)
                )
                new_image.paste(image, (0, 0))
                image = new_image
        else:
            new_w, new_h = image.size
            # Ensure dimensions are divisible by merge_kernel_size * patch_size
            # so that the grid dimensions are divisible by merge_kernel_size
            crop_size_w = merge_kernel_size[1] * patch_size
            crop_size_h = merge_kernel_size[0] * patch_size
            new_w = new_w - new_w % crop_size_w
            new_h = new_h - new_h % crop_size_h
            # Center crop
            left = (image.size[0] - new_w) // 2
            top = (image.size[1] - new_h) // 2
            image = image.crop((left, top, left + new_w, top + new_h))

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_mlx(self, image: Image.Image) -> mx.array:
        """Convert PIL image to MLX array in CHW format, normalized to [0, 1]."""
        image = image.convert("RGB")
        w, h = image.size
        # Convert PIL image to MLX array directly via bytes
        arr = mx.array(list(image.getdata()), dtype=mx.float32).reshape(h, w, 3) / 255.0
        # Convert from HWC to CHW format
        arr = arr.transpose(2, 0, 1)
        return arr

    def normalize(self, image: mx.array) -> mx.array:
        """Normalize image with CLIP-style mean and std."""
        mean = mx.array(self.image_mean, dtype=mx.float32).reshape(3, 1, 1)
        std = mx.array(self.image_std, dtype=mx.float32).reshape(3, 1, 1)
        return (image - mean) / std

    def patchify(self, image: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """Convert image to patches."""
        patch_size = self.patch_size
        C, H, W = image.shape

        # Reshape to (C, H//p, p, W//p, p) then to (num_patches, C, p, p)
        patches = image.reshape(
            C, H // patch_size, patch_size, W // patch_size, patch_size
        )
        # Permute to (H//p, W//p, C, p, p)
        patches = patches.transpose(1, 3, 0, 2, 4)
        # Flatten to (num_patches, C, p, p)
        patches = patches.reshape(-1, C, patch_size, patch_size)

        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: ImageInput) -> Tuple[mx.array, Tuple[int, int]]:
        """
        Preprocess image and patchify it.

        Args:
            image: Image to preprocess.

        Returns:
            patches: mx.array
            grid_hw: Tuple[int, int]
        """
        image = self.rescale(image, self.merge_kernel_size)
        image = self.to_mlx(image)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

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
                "Invalid image type. Must be of type PIL.Image.Image or mx.array."
            )

        pixel_values_list = []
        image_grid_hws = []

        for image in images:
            # Convert MLX arrays to PIL Images if needed
            if isinstance(image, mx.array):
                # Ensure we're working with the array values
                arr = image
                if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                    # CHW format, convert to HWC
                    arr = arr.transpose(1, 2, 0)
                # Convert to uint8 for PIL
                if arr.dtype in [mx.float32, mx.float16, mx.bfloat16]:
                    arr = (arr * 255).astype(mx.uint8)
                # Convert to PIL via list (MLX -> list -> PIL)
                h, w, _ = arr.shape
                flat_data = arr.reshape(-1).tolist()
                image = Image.frombytes("RGB", (w, h), bytes(flat_data))

            patches, image_grid_hw = self._preprocess(image)
            pixel_values_list.append(patches)
            image_grid_hws.append(image_grid_hw)

        pixel_values = mx.concatenate(pixel_values_list, axis=0)
        image_grid_hws = mx.array(image_grid_hws)

        # Return MLX arrays directly
        data = {
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
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


class KimiVLProcessor(ProcessorMixin):
    """
    MLX-based processor for KimiVL that doesn't require torch/torchvision.

    Constructs a KimiVL processor which wraps a KimiVL image processor and a tokenizer
    into a single processor.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "KimiVLImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|media_pad|>"
        if image_processor is None:
            image_processor = KimiVLImageProcessor()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

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
            BatchFeature with input_ids, attention_mask, and pixel_values.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # Check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        # Extract return_tensors from kwargs (unused, we always return MLX arrays)
        kwargs.pop("return_tensors", None)

        # Process images
        if images is not None:
            image_inputs = self.image_processor(images)
            image_grid_hws = image_inputs["image_grid_hws"]
        else:
            image_inputs = {}
            image_grid_hws = None

        # Process text
        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # Replace image tokens with the correct number of placeholder tokens
        if image_grid_hws is not None and text is not None:
            merge_length = (
                self.image_processor.merge_kernel_size[0]
                * self.image_processor.merge_kernel_size[1]
            )
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    # Use mx.prod for MLX arrays
                    grid_hw = image_grid_hws[index]
                    num_placeholders = int(mx.prod(grid_hw).item()) // merge_length
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_placeholders,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Tokenize text
        # Note: The TikToken tokenizer doesn't work properly with transformers' standard
        # __call__ method due to issues with the pad function. We use encode() directly.
        if text is not None:
            # Encode each text and build the result manually
            all_input_ids = []
            for t in text:
                ids = self.tokenizer.encode(t)
                all_input_ids.append(ids)

            # Pad sequences to the same length if needed
            max_len = max(len(ids) for ids in all_input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0

            padded_input_ids = []
            attention_masks = []
            for ids in all_input_ids:
                padding_length = max_len - len(ids)
                padded_ids = ids + [pad_token_id] * padding_length
                mask = [1] * len(ids) + [0] * padding_length
                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)

            # Convert to MLX arrays
            text_inputs = {
                "input_ids": mx.array(padded_input_ids),
                "attention_mask": mx.array(attention_masks),
            }
        else:
            text_inputs = {}

        return BatchFeature(data={**text_inputs, **image_inputs})

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
        # Use provided template, processor's template, or tokenizer's template
        if chat_template is None:
            chat_template = self.chat_template
        if chat_template is None:
            chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template is None:
            raise ValueError(
                "No chat template found. Please provide a chat_template argument "
                "or ensure the tokenizer has a chat_template attribute."
            )

        # Use jinja2 to render the template
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

        # Load image processor config and create our processor
        image_processor_config = {}
        try:
            if is_local:
                config_path = model_path / "config.json"
            else:
                config_path = Path(
                    hf_hub_download(pretrained_model_name_or_path, "config.json")
                )
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)
            if hasattr(config, "vision_config"):
                vision_config = config.vision_config
                if hasattr(vision_config, "patch_size"):
                    image_processor_config["patch_size"] = vision_config.patch_size
                if hasattr(vision_config, "in_token_limit"):
                    image_processor_config["in_token_limit"] = (
                        vision_config.in_token_limit
                    )
                if hasattr(vision_config, "merge_kernel_size"):
                    image_processor_config["merge_kernel_size"] = (
                        vision_config.merge_kernel_size
                    )
        except Exception:
            pass

        image_processor = KimiVLImageProcessor(**image_processor_config)

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
                    # Set chat_template on tokenizer so apply_chat_template works
                    tokenizer.chat_template = chat_template
            except Exception:
                pass

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


from transformers import AutoProcessor

_original_auto_processor_from_pretrained = AutoProcessor.from_pretrained


@classmethod
def _patched_auto_processor_from_pretrained(
    cls, pretrained_model_name_or_path, **kwargs
):
    """Patched from_pretrained that returns KimiVLProcessor for kimi_vl models."""
    from huggingface_hub import hf_hub_download

    model_path = Path(pretrained_model_name_or_path)
    is_local = model_path.exists() and model_path.is_dir()

    # Check if this is a kimi_vl model
    is_kimi_vl = False
    try:
        if is_local:
            config_path = model_path / "config.json"
        else:
            config_path = Path(
                hf_hub_download(pretrained_model_name_or_path, "config.json")
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        is_kimi_vl = config.get("model_type", "").lower() == "kimi_vl"
    except Exception:
        pass

    if is_kimi_vl:
        return KimiVLProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

    return _original_auto_processor_from_pretrained.__func__(
        cls, pretrained_model_name_or_path, **kwargs
    )


AutoProcessor.from_pretrained = _patched_auto_processor_from_pretrained
