"""
Processor class for Llava.
"""

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


class LlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    Args:
        image_processor: The image processor.
        tokenizer: The tokenizer.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config.
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
        self.image_token_id = tokenizer.encode(
            self.image_token, add_special_tokens=False
        )[0]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: (
            TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
        ) = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            images (`ImageInput`, *optional*):
                The image or batch of images to be prepared.
            text (`str`, `list[str]`, *optional*):
                The sequence or batch of sequences to be encoded.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # Pop common kwargs
        return_tensors = kwargs.pop("return_tensors", None)

        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # Try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            pixel_values = image_inputs["pixel_values"]
            height, width = get_image_size(to_numpy_array(pixel_values[0]))
            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size
            ) + self.num_additional_image_tokens
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(
                    self.image_token, self.image_token * num_image_tokens
                )
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **kwargs, return_tensors=None)

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

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

        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
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
            ip_cfg = proc_cfg.get("image_processor", {})
            if "patch_size" in ip_cfg:
                ip_overrides["patch_size"] = ip_cfg["patch_size"]
            if "size" in ip_cfg:
                ip_overrides["size"] = ip_cfg["size"]

        try:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=False,
                **ip_overrides,
                **kwargs,
            )
        except ValueError:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                **ip_overrides,
                **kwargs,
            )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )


__all__ = ["LlavaProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("llava", LlavaProcessor)
