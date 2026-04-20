"""
Processor class for Granite Vision.
"""

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import install_auto_processor_patch, load_chat_template, to_mlx


class GraniteVisionProcessor(ProcessorMixin):
    r"""
    Constructs a Granite Vision processor which wraps an image processor and a tokenizer.

    Args:
        image_processor: The image processor.
        tokenizer: The tokenizer.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1).
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoImageProcessor, AutoTokenizer

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, use_fast=False, **kwargs
            )
        except ValueError:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

        # Read processor_config.json for patch_size, vision_feature_select_strategy
        proc_kwargs = {}
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in (
                "patch_size",
                "vision_feature_select_strategy",
                "image_token",
                "num_additional_image_tokens",
            ):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: (
            TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
        ) = None,
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        return_tensors = kwargs.pop("return_tensors", None)
        kwargs.pop("padding", None)
        do_pad = kwargs.pop("do_pad", True)

        if images is not None:
            image_inputs = self.image_processor(images, do_pad=do_pad, **kwargs)
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        prompt_strings = text
        if image_inputs:
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0])
            )
            prompt_strings = []
            for sample in text:
                while self.image_token in sample:
                    image_size = next(image_sizes)
                    if not isinstance(image_size, (list, tuple)):
                        image_size = image_size.tolist()
                    orig_height, orig_width = image_size
                    num_image_tokens = self._get_number_of_features(
                        orig_height, orig_width, height, width
                    )
                    if self.vision_feature_select_strategy == "default":
                        num_image_tokens -= 1
                    sample = sample.replace(
                        self.image_token,
                        "<placeholder>" * num_image_tokens,
                        1,
                    )
                prompt_strings.append(sample)
            prompt_strings = [
                sample.replace("<placeholder>", self.image_token)
                for sample in prompt_strings
            ]

        text_inputs = self.tokenizer(prompt_strings, **kwargs)

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def _get_number_of_features(
        self, orig_height: int, orig_width: int, height: int, width: int
    ) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = (
            height_best_resolution // height,
            width_best_resolution // width,
        )

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height,
            orig_width,
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )
        # The base patch covers the entire image
        base_features = (
            patches_height * patches_width + self.num_additional_image_tokens
        )
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    def _get_unpadded_features(
        self,
        height,
        width,
        patches_height,
        patches_width,
        scale_height,
        scale_width,
    ):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        return (unpadded_features, newline_features)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["GraniteVisionProcessor"]

install_auto_processor_patch(["granite_vision", "llava_next"], GraniteVisionProcessor)
