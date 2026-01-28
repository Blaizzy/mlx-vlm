"""Image processor and Processor for ERNIE 4.5 VL MoE."""

import math
import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import sentencepiece as spm
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import (
    BaseImageProcessor as HFBaseImageProcessor,
)
from transformers.image_transforms import (
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    is_valid_image,
    to_numpy_array,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class Ernie4_5_VLTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for ERNIE 4.5 VL model using SentencePiece.

    Matches the original Baidu implementation: a thin wrapper around SentencePiece
    with no custom pre-tokenization or added_tokens handling.
    """

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<|begin_of_sentence|>",
        eos_token="</s>",
        mask_token="<mask:1>",
        pad_token="<unk>",
        sep_token="<|end_of_sentence|>",
        unk_token="<unk>",
        additional_special_tokens=None,
        chat_template=None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]

        # Load chat_template from tokenizer_config.json if not provided
        if chat_template is None:
            import json

            config_file = os.path.join(
                os.path.dirname(vocab_file), "tokenizer_config.json"
            )
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                    chat_template = config.get("chat_template")

        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            chat_template=chat_template,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.vocab_size()

    @property
    def space_token_id(self):
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token_id(self):
        return self.sp_model.piece_to_id("<mask:7>")

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        return self.sp_model.id_to_piece(id)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Do not add CLS/SEP tokens - the chat template handles BOS."""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return None
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.vocab_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args,
            **kwargs,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )


def _validate_images_text_input_order(images, text):
    """Validate and possibly swap images and text if they were passed in wrong order."""
    # If images is a string and text is None, treat images as text
    if isinstance(images, str) and text is None:
        return None, images
    # If both provided and images is actually text
    if images is not None and text is not None:
        if isinstance(images, str) and not isinstance(text, str):
            # images is actually text, text is actually images
            return text, images
    return images, text


def round_by_factor(number: int, factor: int) -> int:
    """Round number to nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Round up number to nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Round down number to nearest multiple of factor."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> Tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels within ['min_pixels', 'max_pixels']
    3. Aspect ratio maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        factor: Factor to make dimensions divisible by (patch_size * merge_size)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (resized_height, resized_width)
    """
    # Clamp extreme aspect ratios
    MAX_RATIO = 200
    if height / width > MAX_RATIO:
        width = height // MAX_RATIO
    elif width / height > MAX_RATIO:
        height = width // MAX_RATIO

    # Round to nearest factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # Scale down if exceeding max_pixels
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    # Scale up if below min_pixels
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    # Ensure minimum size
    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)

    return h_bar, w_bar


class ImageProcessor(HFBaseImageProcessor):
    """
    Image processor for ERNIE 4.5 VL MoE model.

    Handles variable resolution images by:
    1. Smart resizing to dimensions divisible by (patch_size * merge_size)
    2. Extracting patches in the format expected by the vision encoder
    3. Computing grid_thw (temporal, height, width in patches)

    Inherits from HuggingFace's BaseImageProcessor for ProcessorMixin compatibility.
    Does NOT inherit from local BaseImageProcessor to avoid wrong code path in prepare_inputs.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        image_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        image_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        size: Tuple[int, int] = (224, 224),
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        config=None,
        **kwargs,
    ):
        # Extract values from config if provided (can be dict or object)
        if config is not None:
            # Handle both dict (from load_image_processor) and object configs
            if isinstance(config, dict):
                # Get vision_config from the main config dict
                vision_config = config.get("vision_config", {})

                # Extract image processing params from root or vision_config
                image_mean = config.get("image_mean", image_mean)
                image_std = config.get("image_std", image_std)
                min_pixels = config.get("min_pixels", min_pixels)
                max_pixels = config.get("max_pixels", max_pixels)

                # Extract vision params
                patch_size = vision_config.get(
                    "patch_size", config.get("patch_size", patch_size)
                )
                merge_size = vision_config.get(
                    "spatial_merge_size", config.get("spatial_merge_size", merge_size)
                )
                temporal_patch_size = vision_config.get(
                    "temporal_patch_size",
                    config.get("temporal_patch_size", temporal_patch_size),
                )
            else:
                # Object config (VisionConfig or similar)
                patch_size = getattr(config, "patch_size", patch_size)
                merge_size = getattr(
                    config,
                    "spatial_merge_size",
                    getattr(config, "merge_size", merge_size),
                )
                temporal_patch_size = getattr(
                    config, "temporal_patch_size", temporal_patch_size
                )

        # Initialize HFBaseImageProcessor (for ProcessorMixin compatibility)
        HFBaseImageProcessor.__init__(self, **kwargs)

        # Store our custom attributes
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.factor = patch_size * merge_size

    def get_smart_resize(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Compute smart resize dimensions and grid dimensions.

        Args:
            height: Original image height
            width: Original image width
            min_pixels: Override minimum pixels
            max_pixels: Override maximum pixels

        Returns:
            Tuple of ((resized_h, resized_w), (grid_h, grid_w))
        """
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )

        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        return (resized_height, resized_width), (grid_h, grid_w)

    def _extract_patches(
        self,
        image: np.ndarray,
        grid_h: int,
        grid_w: int,
    ) -> np.ndarray:
        """
        Extract patches from image in the format expected by the vision encoder.

        Args:
            image: Image array of shape [C, H, W]
            grid_h: Number of patches in height
            grid_w: Number of patches in width

        Returns:
            Patches of shape [grid_h * grid_w, C * patch_size * patch_size]
        """
        C, H, W = image.shape

        # Reshape to patches with merge_size aggregation
        # [C, H, W] -> [C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
        patches = image.reshape(
            C,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Transpose to group spatial patches together
        # -> [grid_h/merge, grid_w/merge, merge, merge, C, patch, patch]
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)

        # Flatten to [num_patches, C * patch_size * patch_size]
        num_patches = (
            (grid_h // self.merge_size)
            * (grid_w // self.merge_size)
            * (self.merge_size**2)
        )
        patches = patches.reshape(num_patches, C * self.patch_size * self.patch_size)

        return patches

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_grid_thw: bool = True,
    ) -> Union[np.ndarray, Dict]:
        """
        Preprocess images for ERNIE 4.5 VL.

        Args:
            images: Single image or list of images
            return_grid_thw: If True, return dict with pixel_values and image_grid_thw

        Returns:
            If return_grid_thw is True: Dict with 'pixel_values' and 'image_grid_thw'
            Otherwise: numpy array of processed images
        """
        if isinstance(images, Image.Image):
            images = [images]

        all_patches = []
        all_grid_thw = []

        for image in images:
            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get smart resize dimensions
            (resized_h, resized_w), (grid_h, grid_w) = self.get_smart_resize(
                image.height, image.width
            )

            # Convert to numpy
            img_array = to_numpy_array(image)

            # Resize
            img_array = resize(
                img_array,
                size=(resized_h, resized_w),
                resample=self.resample,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Rescale
            img_array = rescale(
                img_array,
                scale=self.rescale_factor,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Normalize
            img_array = normalize(
                img_array,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Convert to channel first [H, W, C] -> [C, H, W]
            img_array = to_channel_dimension_format(
                img_array,
                channel_dim=ChannelDimension.FIRST,
                input_channel_dim=ChannelDimension.LAST,
            )

            # Extract patches
            patches = self._extract_patches(img_array, grid_h, grid_w)
            all_patches.append(patches)

            # Store grid_thw (temporal=1 for images)
            all_grid_thw.append([1, grid_h, grid_w])

        # Concatenate all patches
        pixel_values = np.concatenate(all_patches, axis=0)

        if return_grid_thw:
            return {
                "pixel_values": pixel_values,
                "image_grid_thw": np.array(all_grid_thw, dtype=np.int64),
            }

        return pixel_values

    def preprocess_video(
        self,
        frames: List[Image.Image],
        return_grid_thw: bool = True,
    ) -> Union[np.ndarray, Dict]:
        """
        Preprocess video frames for ERNIE 4.5 VL.

        Args:
            frames: List of video frames as PIL Images
            return_grid_thw: If True, return dict with pixel_values and video_grid_thw

        Returns:
            If return_grid_thw is True: Dict with 'pixel_values' and 'video_grid_thw'
            Otherwise: numpy array of processed frames
        """
        if not frames:
            raise ValueError("frames list cannot be empty")

        # Get dimensions from first frame
        first_frame = frames[0]
        if first_frame.mode != "RGB":
            first_frame = first_frame.convert("RGB")

        (resized_h, resized_w), (grid_h, grid_w) = self.get_smart_resize(
            first_frame.height, first_frame.width
        )

        all_patches = []

        for frame in frames:
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            # Convert to numpy
            img_array = to_numpy_array(frame)

            # Resize
            img_array = resize(
                img_array,
                size=(resized_h, resized_w),
                resample=self.resample,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Rescale
            img_array = rescale(
                img_array,
                scale=self.rescale_factor,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Normalize
            img_array = normalize(
                img_array,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Convert to channel first
            img_array = to_channel_dimension_format(
                img_array,
                channel_dim=ChannelDimension.FIRST,
                input_channel_dim=ChannelDimension.LAST,
            )

            # Extract patches
            patches = self._extract_patches(img_array, grid_h, grid_w)
            all_patches.append(patches)

        # Stack all frame patches
        pixel_values = np.concatenate(all_patches, axis=0)

        # Compute temporal grid
        num_frames = len(frames)
        grid_t = num_frames

        if return_grid_thw:
            return {
                "pixel_values": pixel_values,
                "video_grid_thw": np.array([[grid_t, grid_h, grid_w]], dtype=np.int64),
            }

        return pixel_values

    def __call__(
        self,
        images: ImageInput,
        **kwargs,
    ) -> BatchFeature:
        """Make the image processor callable."""
        return self.preprocess(images, **kwargs)


class Ernie4_5_VLProcessor(ProcessorMixin):
    """
    MLX-based processor for ERNIE 4.5 VL that doesn't require decord.

    Constructs an ERNIE 4.5 VL processor which wraps an image processor and a tokenizer
    into a single processor for image-only inference.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "spatial_conv_size", "temporal_conv_size"]
    image_processor_class = "ImageProcessor"
    tokenizer_class = "Ernie4_5_VLTokenizer"

    # Special tokens
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"
    IMAGE_PLACEHOLDER = "<|IMAGE_PLACEHOLDER|>"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = ImageProcessor()
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @property
    def pad_token(self):
        """Forward pad_token to tokenizer."""
        return self.tokenizer.pad_token if self.tokenizer else None

    @property
    def pad_token_id(self):
        """Forward pad_token_id to tokenizer."""
        return self.tokenizer.pad_token_id if self.tokenizer else None

    @property
    def eos_token(self):
        """Forward eos_token to tokenizer."""
        return self.tokenizer.eos_token if self.tokenizer else None

    @property
    def eos_token_id(self):
        """Forward eos_token_id to tokenizer."""
        return self.tokenizer.eos_token_id if self.tokenizer else None

    @property
    def bos_token(self):
        """Forward bos_token to tokenizer."""
        return self.tokenizer.bos_token if self.tokenizer else None

    @property
    def bos_token_id(self):
        """Forward bos_token_id to tokenizer."""
        return self.tokenizer.bos_token_id if self.tokenizer else None

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

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, and image_grid_thw.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # Check if images and text inputs are reversed
        images, text = _validate_images_text_input_order(images, text)

        # Extract return_tensors from kwargs
        kwargs.pop("return_tensors", None)

        # Process images
        if images is not None:
            # Handle single image
            if is_valid_image(images):
                images = [images]

            image_inputs = self.image_processor(images)
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # Process text
        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # Replace image placeholders with the correct number of placeholder tokens
        if image_grid_thw is not None and text is not None:
            merge_length = self.spatial_conv_size * self.spatial_conv_size
            index = 0
            for i in range(len(text)):
                # Handle <|image@placeholder|> format used in chat templates
                placeholder = f"{self.IMG_START}<|image@placeholder|>{self.IMG_END}"
                while placeholder in text[i]:
                    if index < len(image_grid_thw):
                        grid_thw = image_grid_thw[index]
                        # grid_thw is [t, h, w], compute number of tokens
                        num_patches = int(np.prod(grid_thw))
                        num_placeholders = num_patches // merge_length
                        replacement = (
                            f"{self.IMG_START}"
                            f"{self.IMAGE_PLACEHOLDER * num_placeholders}"
                            f"{self.IMG_END}"
                        )
                        text[i] = text[i].replace(placeholder, replacement, 1)
                        index += 1
                    else:
                        break

        # Tokenize text
        if text is not None:
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

            # For text-only processing (no images), return lists for compatibility
            # with prepare_inputs which does list concatenation
            if images is None:
                # Return as flat list if single sequence, else as list of lists
                if len(padded_input_ids) == 1:
                    text_inputs = {
                        "input_ids": padded_input_ids[0],
                        "attention_mask": attention_masks[0],
                    }
                else:
                    text_inputs = {
                        "input_ids": padded_input_ids,
                        "attention_mask": attention_masks,
                    }
            else:
                # When images are present, return MLX arrays
                text_inputs = {
                    "input_ids": mx.array(padded_input_ids),
                    "attention_mask": mx.array(attention_masks),
                }
        else:
            text_inputs = {}

        # Convert image inputs to MLX arrays
        if image_inputs:
            image_inputs = {
                "pixel_values": mx.array(image_inputs["pixel_values"]),
                "image_grid_thw": mx.array(image_inputs["image_grid_thw"]),
            }

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

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model path."""
        from pathlib import Path

        # Get the actual path if it's a HuggingFace model ID
        if not Path(pretrained_model_name_or_path).exists():
            from huggingface_hub import snapshot_download

            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path,
                allow_patterns=["*.json", "*.model", "*.txt"],
            )

        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(pretrained_model_name_or_path)
        image_processor = ImageProcessor()

        return Ernie4_5_VLProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )


MODEL_TYPE = "ernie4_5_moe_vl"

try:
    AutoImageProcessor.register(MODEL_TYPE, slow_image_processor_class=ImageProcessor)
    AutoProcessor.register(MODEL_TYPE, Ernie4_5_VLProcessor)
except Exception as e:
    raise Exception(f"Error registering {MODEL_TYPE} processor: {e}")
