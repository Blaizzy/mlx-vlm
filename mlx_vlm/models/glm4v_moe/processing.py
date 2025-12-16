"""Processor for GLM-4V-MoE model.

Handles image/video token expansion based on grid dimensions and merge size.
Based on the HuggingFace transformers GLM-4.6V processor implementation.
"""

from typing import List, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


class Glm46VMoEProcessor(ProcessorMixin):
    """
    Processor for GLM-4V-MoE that wraps an image processor and tokenizer.

    Handles:
    - Image preprocessing via image_processor
    - Token replacement for image/video placeholders based on grid dimensions
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # Get image/video tokens from tokenizer or use defaults
        self.image_token = "<|image|>"
        self.video_token = "<|video|>"

        if tokenizer is not None:
            self.image_token = getattr(tokenizer, "image_token", "<|image|>")
            self.video_token = getattr(tokenizer, "video_token", "<|video|>")

            # Get token IDs
            self.image_token_id = getattr(tokenizer, "image_token_id", None)
            if self.image_token_id is None:
                self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

            self.video_token_id = getattr(tokenizer, "video_token_id", None)
            if self.video_token_id is None:
                self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        else:
            self.image_token_id = None
            self.video_token_id = None

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process images/videos and text for the model.

        Args:
            images: Single image or list of images (PIL.Image, np.ndarray, etc.)
            text: Single text or list of texts
            videos: Video inputs (optional)
            **kwargs: Additional arguments passed to image_processor and tokenizer

        Returns:
            BatchFeature with:
                - input_ids: Token IDs with image/video placeholders expanded
                - attention_mask: Attention mask
                - pixel_values: Processed image/video patches
                - image_grid_thw: Grid dimensions for each image
                - video_grid_thw: Grid dimensions for each video (if videos provided)
        """
        image_inputs = {}
        video_inputs = {}
        image_grid_thw = None
        video_grid_thw = None

        # Pop tokenizer-specific kwargs that shouldn't go to image processor
        padding = kwargs.pop("padding", False)
        return_token_type_ids = kwargs.pop("return_token_type_ids", False)
        return_tensors = kwargs.pop("return_tensors", None)

        # Process images
        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs.get("image_grid_thw")

        # Process videos
        if videos is not None:
            if hasattr(self, "video_processor") and self.video_processor is not None:
                video_inputs = self.video_processor(videos=videos, **kwargs)
                video_grid_thw = video_inputs.get("video_grid_thw")

        # Handle text input
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        # Make a copy to avoid modifying original
        text = [t for t in text]

        # Get merge_size from image_processor
        merge_size = getattr(self.image_processor, "merge_size", 2)
        if hasattr(self.image_processor, "spatial_merge_size"):
            merge_size = self.image_processor.spatial_merge_size
        merge_length = merge_size**2

        # Expand image tokens based on grid dimensions
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    # Calculate number of image tokens: prod(grid_thw) / merge_size^2
                    grid = image_grid_thw[index]
                    if hasattr(grid, "tolist"):
                        grid = grid.tolist()
                    num_image_tokens = int(np.prod(grid) // merge_length)

                    # Replace single image token with correct number of placeholder tokens
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                # Replace placeholders back to image tokens
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Expand video tokens based on grid dimensions
        if video_grid_thw is not None:
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    grid = video_grid_thw[video_index]
                    if hasattr(grid, "tolist"):
                        grid = grid.tolist()

                    num_frames = grid[0]
                    # Calculate tokens per frame
                    num_tokens_per_frame = int(
                        np.prod(grid) // merge_length // num_frames
                    )

                    # Build video structure with frame tokens
                    video_structure = ""
                    for frame_idx in range(num_frames):
                        # Add image tokens for this frame
                        frame_structure = self.image_token * num_tokens_per_frame
                        video_structure += frame_structure

                    text[i] = text[i].replace(self.video_token, video_structure, 1)
                    video_index += 1

        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            padding=padding,
            return_token_type_ids=return_token_type_ids,
            **kwargs,
        )

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        """Apply chat template using the tokenizer."""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        """Return combined input names from tokenizer and image processor."""
        tokenizer_input_names = (
            self.tokenizer.model_input_names if self.tokenizer else []
        )
        image_processor_input_names = (
            self.image_processor.model_input_names
            if hasattr(self.image_processor, "model_input_names")
            else []
        )
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model path."""
        from transformers import AutoTokenizer, Glm4vImageProcessor

        trust_remote_code = kwargs.pop("trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        image_processor = Glm4vImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)


__all__ = ["Glm46VMoEProcessor"]

# Alias for backwards compatibility
Glm4VMoEProcessor = Glm46VMoEProcessor
