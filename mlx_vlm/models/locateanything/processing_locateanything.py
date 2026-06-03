import json
import re
import warnings
from pathlib import Path
from typing import List, Union

import mlx.core as mx
import transformers.processing_utils as processing_utils
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .image_processing_locateanything import LocateAnythingImageProcessor


def _validate_images_text_input_order(images, text):
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


if not hasattr(processing_utils, "_validate_images_text_input_order"):
    processing_utils._validate_images_text_input_order = (
        _validate_images_text_input_order
    )


class LocateAnythingProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "LocateAnythingImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<IMG_CONTEXT>"
        self.image_start_token = "<img>"
        self.image_end_token = "</img>"
        if image_processor is None:
            image_processor = LocateAnythingImageProcessor()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        images, text = _validate_images_text_input_order(images, text)
        kwargs.pop("return_tensors", None)

        if images is not None:
            image_inputs = self.image_processor(images)
            image_grid_hws = image_inputs["image_grid_hws"]
        else:
            image_inputs = {}
            image_grid_hws = None

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        if image_grid_hws is not None and text is not None:
            merge_length = (
                self.image_processor.merge_kernel_size[0]
                * self.image_processor.merge_kernel_size[1]
            )
            num_images = int(image_grid_hws.shape[0])
            counter = {"index": 0}
            pattern = re.compile(r"<image-\d+>")

            def _expand(_match):
                idx = counter["index"]
                if idx >= num_images:
                    raise ValueError(
                        f"Found more <image-N> placeholders than images "
                        f"({num_images} provided)."
                    )
                num_placeholders = (
                    int(mx.prod(image_grid_hws[idx]).item()) // merge_length
                )
                counter["index"] += 1
                return (
                    self.image_start_token
                    + self.image_token * num_placeholders
                    + self.image_end_token
                )

            text = [pattern.sub(_expand, t) for t in text]
            if counter["index"] != num_images:
                raise ValueError(
                    f"Number of <image-N> placeholders ({counter['index']}) does "
                    f"not match the number of images ({num_images})."
                )

        if text is not None:
            if (
                self.tokenizer.pad_token_id is None
                and self.tokenizer.eos_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            text_inputs = self.tokenizer(
                text,
                return_tensors=None,
                padding=True,
            )
            text_inputs = {
                "input_ids": mx.array(text_inputs["input_ids"]),
                "attention_mask": mx.array(text_inputs["attention_mask"]),
            }
        else:
            text_inputs = {}

        data = {**text_inputs, **image_inputs}
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if image_token_id is not None and image_token_id != unk_id:
            data["image_token_id"] = int(image_token_id)
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
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

        rendered = Template(chat_template).render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.tokenizer.encode(rendered)
        return rendered

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        chat_template = self.chat_template or getattr(
            self.tokenizer, "chat_template", None
        )
        if chat_template is not None and hasattr(self.tokenizer, "chat_template"):
            self.tokenizer.chat_template = chat_template

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            tokenizer_files = self.tokenizer.save_pretrained(str(save_dir), **kwargs)
            if tokenizer_files is not None:
                saved_files.extend(tokenizer_files)

        auto_map = {
            "AutoImageProcessor": (
                "image_processing_locateanything.LocateAnythingImageProcessor"
            ),
            "AutoProcessor": "processing_locateanything.LocateAnythingProcessor",
        }
        image_config = {
            "auto_map": auto_map,
            "image_processor_type": type(self.image_processor).__name__,
            "image_mean": [float(x) for x in self.image_processor.image_mean],
            "image_std": [float(x) for x in self.image_processor.image_std],
            "in_token_limit": int(self.image_processor.in_token_limit),
            "merge_kernel_size": list(self.image_processor.merge_kernel_size),
            "patch_size": int(self.image_processor.patch_size),
            "processor_class": type(self).__name__,
        }
        processor_config = {
            "auto_map": auto_map,
            "image_end_token": self.image_end_token,
            "image_placeholder": "image",
            "image_start_token": self.image_start_token,
            "image_token": self.image_token,
            "in_token_limit": image_config["in_token_limit"],
            "merge_kernel_size": image_config["merge_kernel_size"],
            "patch_size": image_config["patch_size"],
            "processor_class": type(self).__name__,
            "video_placeholder": "video",
            "video_token": self.image_token,
        }
        if chat_template is not None:
            processor_config["chat_template"] = chat_template

        for filename, config in (
            ("processor_config.json", processor_config),
            ("preprocessor_config.json", image_config),
        ):
            path = save_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            saved_files.append(str(path))

        if chat_template is not None:
            path = save_dir / "chat_template.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"chat_template": chat_template}, f, indent=2)
            saved_files.append(str(path))

        return tuple(saved_files)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        def _load_json(filename):
            try:
                if is_local:
                    path = model_path / filename
                    if not path.exists():
                        return {}
                else:
                    path = Path(
                        hf_hub_download(pretrained_model_name_or_path, filename)
                    )
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

        proc_cfg = _load_json("processor_config.json")
        preproc_cfg = _load_json("preprocessor_config.json")

        image_processor_config = {}
        for key in ("patch_size", "merge_kernel_size", "in_token_limit"):
            if key in preproc_cfg:
                image_processor_config[key] = preproc_cfg[key]
            elif key in proc_cfg:
                image_processor_config[key] = proc_cfg[key]

        image_processor = LocateAnythingImageProcessor(**image_processor_config)

        chat_template = None
        chat_tpl_cfg = _load_json("chat_template.json")
        if "chat_template" in chat_tpl_cfg:
            chat_template = chat_tpl_cfg["chat_template"]
        elif "chat_template" in proc_cfg:
            chat_template = proc_cfg["chat_template"]
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is not None:
            tokenizer.chat_template = chat_template

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


from ..base import install_auto_processor_patch

install_auto_processor_patch("locateanything", LocateAnythingProcessor)
