"""
Processor class for PaliGemma.
"""

import inspect
import logging

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)

from ..base import load_chat_template, to_mlx

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [
    f"<seg{i:0>3}>" for i in range(128)
]


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def _split_image_processor_kwargs(image_processor, kwargs):
    image_kwargs = {}
    valid_kwargs = set()

    typed_valid_kwargs = getattr(image_processor, "valid_kwargs", None)
    annotations = getattr(typed_valid_kwargs, "__annotations__", None)
    if annotations:
        valid_kwargs.update(annotations.keys())

    try:
        parameters = inspect.signature(image_processor.__call__).parameters
    except (TypeError, ValueError):
        parameters = {}

    valid_kwargs.update(
        name
        for name, parameter in parameters.items()
        if name not in {"self", "images"}
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )

    for key in list(kwargs.keys()):
        if key in valid_kwargs:
            image_kwargs[key] = kwargs.pop(key)

    return image_kwargs


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.

    For example, for the call:
    build_string_from_input(
        prompt="Prefix str",
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
        num_images=1,
    )
    The output will be:
    "<im><im><im><s>Prefix str"

    Args:
        prompt (`str`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


class PaliGemmaProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer
    into a single processor.

    Args:
        image_processor: The image processor.
        tokenizer: The tokenizer.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError(
                "Image processor is missing an `image_seq_length` attribute."
            )

        self.image_seq_length = image_processor.image_seq_length

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            self.image_token = IMAGE_TOKEN
        else:
            self.image_token_id = tokenizer.image_token_id
            self.image_token = tokenizer.image_token

        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

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

        # Read processor_config.json for correct init kwargs
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("image_seq_length", "image_token"):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]
            ip_cfg = proc_cfg.get("image_processor", {})
            if "patch_size" in ip_cfg:
                ip_overrides["patch_size"] = ip_cfg["patch_size"]
            if "size" in ip_cfg:
                ip_overrides["size"] = ip_cfg["size"]

        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=False,
            **ip_overrides,
            **kwargs,
        )
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
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            images (`ImageInput`, *optional*):
                The image or batch of images to be prepared.
            text (`str`, `list[str]`, *optional*):
                The sequence or batch of sequences to be encoded.
            suffix (`str`, `list[str]`, *optional*):
                The suffixes or batch of suffixes to be encoded. Only necessary for finetuning.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model.
            - **labels** -- Labels compatible with training if `suffix` is not None.
        """
        suffix = kwargs.pop("suffix", None)
        return_tensors = kwargs.pop("return_tensors", None)

        return_token_type_ids = True

        if images is None:
            raise ValueError(
                "`images` are expected as arguments to a `PaliGemmaProcessor` instance."
            )
        if text is None:
            logger.warning(
                "You are using PaliGemma without a text prefix. "
                "It will perform as a picture-captioning model."
            )
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                logger.warning(
                    "You are passing both `text` and `images` to `PaliGemmaProcessor`. "
                    "The processor expects special image tokens in the text, as many tokens "
                    "as there are images per each text. It is recommended to add `<image>` "
                    "tokens in the very beginning of your text. For this call, we will infer "
                    "how many images each text has and add special tokens."
                )

                if isinstance(text, list) and isinstance(images, list):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. "
                            "Each prompt should be associated with an image or list of images."
                        )

                # Make a nested list of lists to be able to iterate over the images and text
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (
                    isinstance(images, (list, tuple))
                    and isinstance(images[0], (list, tuple))
                    and is_valid_image(images[0][0])
                ):
                    raise ValueError(
                        "images must be an image, list of images or list of list of images"
                    )

                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=(
                            len(image_list) if isinstance(image_list, list) else 1
                        ),
                    )
                    for prompt, image_list in zip(text, images)
                ]
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(
                        IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length
                    )
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = (
                        bos_rfind_index + len(IMAGE_TOKEN)
                        if bos_rfind_index != -1
                        else 0
                    )
                    expanded_sample = (
                        expanded_sample[:bos_index]
                        + self.tokenizer.bos_token
                        + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]

        if suffix is not None and _is_str_or_image(suffix):
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

        image_kwargs = _split_image_processor_kwargs(self.image_processor, kwargs)
        pixel_values = self.image_processor(images, **image_kwargs)["pixel_values"]

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **kwargs,
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        # Generate labels: mask prefix tokens (token_type_ids == 0) with -100
        if return_token_type_ids:
            labels = np.array(inputs["input_ids"])
            labels[np.array(inputs["token_type_ids"]) == 0] = -100
            return_data.update({"labels": labels.tolist()})

        return BatchFeature(data=to_mlx(return_data))

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + [
            "token_type_ids",
            "labels",
        ]
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["PaliGemmaProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("paligemma", PaliGemmaProcessor)
