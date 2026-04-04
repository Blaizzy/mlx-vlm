"""
Processor class for SmolVLM.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/processing_smolvlm.py
"""

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


def _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_image_token,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len, fake_token_around_image, image_token, global_image_token
):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_image_token,
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_image_token=global_image_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_image_token,
    )


class SmolVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_len"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_len: int = 169,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.fake_image_token = getattr(
            tokenizer, "fake_image_token", "<fake_token_around_image>"
        )
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.end_of_utterance_token = getattr(
            tokenizer, "end_of_utterance_token", "<end_of_utterance>"
        )
        self.global_image_token = getattr(
            tokenizer, "global_image_token", "<global-img>"
        )
        self.image_seq_len = image_seq_len
        self.video_token = getattr(tokenizer, "video_token", "<video>")

        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
        )

    def expand_text_with_image_tokens(self, text, image_rows, image_cols):
        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    self.image_seq_len,
                    image_token=self.image_token,
                    fake_token_around_image=self.fake_image_token,
                    global_image_token=self.global_image_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(self.image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        return prompt_strings

    def __call__(
        self,
        images: Optional[
            Union[ImageInput, List[ImageInput], List[List[ImageInput]]]
        ] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None and videos is None:
            raise ValueError("You must provide one of `text`, `images` or `videos`.")

        # Treat video frames as images
        if videos is not None and images is None:
            images = videos

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )

        inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_nested_list_of_images(images)

            # Separate image kwargs
            images_kwargs = {}
            for k in list(kwargs.keys()):
                if k in ("return_row_col_info",):
                    images_kwargs[k] = kwargs.pop(k)

            vision_inputs = self.image_processor(images, **images_kwargs)

            image_rows = vision_inputs.pop("rows", None)
            image_cols = vision_inputs.pop("cols", None)
            inputs.update(vision_inputs)

            if text is not None:
                n_images_in_text = [sample.count(self.image_token) for sample in text]
                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )
                if image_rows is None:
                    image_rows = [[0] * n_images for n_images in n_images_in_text]
                if image_cols is None:
                    image_cols = [[0] * n_images for n_images in n_images_in_text]
                text = self.expand_text_with_image_tokens(
                    text, image_rows=image_rows, image_cols=image_cols
                )

        return_tensors = kwargs.pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            inputs.update(text_inputs)

        return BatchFeature(data=to_mlx(inputs))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
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

        # Read processor_config.json for correct init kwargs
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("image_seq_len",):
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
        if "chat_template" not in proc_kwargs:
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template is not None:
                proc_kwargs["chat_template"] = chat_template

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )


__all__ = ["SmolVLMProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("smolvlm", SmolVLMProcessor)
