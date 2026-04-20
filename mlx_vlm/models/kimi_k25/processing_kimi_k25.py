"""
Processor class for Kimi K25.

Adapted from custom HuggingFace Transformers implementation:
https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/kimi_k25_processor.py
https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/kimi_k25_vision_processing.py
"""

import json
import math
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import mlx.core as mx
import transformers.processing_utils as processing_utils
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .config import ModelConfig


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
                "This is deprecated. Please pass images first and text second.",
                FutureWarning,
            )
            return text, images
    return images, text


# Compatibility shims for transformers versions
if not hasattr(processing_utils, "_validate_images_text_input_order"):
    processing_utils._validate_images_text_input_order = (
        _validate_images_text_input_order
    )

if not hasattr(processing_utils, "Unpack"):
    try:
        from typing import Unpack

        processing_utils.Unpack = Unpack
    except ImportError:
        from typing_extensions import Unpack

        processing_utils.Unpack = Unpack


def _ensure_gpt2_bytes_to_unicode():
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    except Exception:
        return
    if hasattr(gpt2_tokenization, "bytes_to_unicode"):
        return

    def bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    gpt2_tokenization.bytes_to_unicode = bytes_to_unicode


class KimiK25ImageProcessor(BaseImageProcessor):
    """Image processor for Kimi K2.5 using the navit resize/pad/patchify pipeline."""

    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        in_token_limit: int = 16384,
        merge_kernel_size: List[int] = None,
        patch_limit_on_one_side: int = 512,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # If a full model config dict is provided (from load_image_processor),
        # extract vision params from it as fallbacks.
        if config is not None:
            vc = config.get("vision_config", {}) if isinstance(config, dict) else {}
            patch_size = vc.get("patch_size", patch_size)
            mks = vc.get("merge_kernel_size", None)
            if mks is not None:
                if isinstance(mks, int):
                    mks = [mks, mks]
                merge_kernel_size = merge_kernel_size or mks

        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.in_token_limit = in_token_limit
        self.merge_kernel_size = (
            merge_kernel_size if merge_kernel_size is not None else [2, 2]
        )
        self.patch_limit_on_one_side = patch_limit_on_one_side

    def rescale(self, image: Image.Image) -> Image.Image:
        """Rescale image using navit logic: scale to fit patch limits, then pad to boundaries."""
        w, h = image.size
        patch_size = self.patch_size
        merge_h, merge_w = self.merge_kernel_size

        # Apply patch limits (navit_resize_image logic)
        s1 = math.sqrt(
            self.in_token_limit
            / (max(1.0, w // patch_size) * max(1.0, h // patch_size))
        )
        s2 = self.patch_limit_on_one_side * patch_size / w
        s3 = self.patch_limit_on_one_side * patch_size / h
        scale = min(1.0, s1, s2, s3)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        new_w = min(new_w, self.patch_limit_on_one_side * patch_size)
        new_h = min(new_h, self.patch_limit_on_one_side * patch_size)

        # Pad to make dimensions divisible by merge_kernel_size * patch_size
        factor_w = merge_w * patch_size
        factor_h = merge_h * patch_size
        pad_w = (factor_w - new_w % factor_w) % factor_w
        pad_h = (factor_h - new_h % factor_h) % factor_h

        image = image.convert("RGB")
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        if pad_w > 0 or pad_h > 0:
            padded = Image.new("RGB", (new_w + pad_w, new_h + pad_h), (0, 0, 0))
            padded.paste(image, (0, 0))
            image = padded

        return image

    def to_mlx(self, image: Image.Image) -> mx.array:
        """Convert PIL image to MLX array in CHW format, normalized to [0, 1]."""
        w, h = image.size
        arr = (
            mx.array(list(image.getdata()), dtype=mx.float32).reshape(h, w, 3) / 255.0
        )
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr

    def normalize(self, image: mx.array) -> mx.array:
        """Normalize image with configured mean and std."""
        mean = mx.array(self.image_mean, dtype=mx.float32).reshape(3, 1, 1)
        std = mx.array(self.image_std, dtype=mx.float32).reshape(3, 1, 1)
        return (image - mean) / std

    def patchify(self, image: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """Convert image to patches. Returns (patches, grid_hw)."""
        C, H, W = image.shape
        patch_size = self.patch_size

        patches = image.reshape(
            C, H // patch_size, patch_size, W // patch_size, patch_size
        )
        patches = patches.transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, C, patch_size, patch_size)

        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: Image.Image) -> Tuple[mx.array, Tuple[int, int]]:
        """Full preprocessing pipeline for a single image."""
        image = self.rescale(image)
        image = self.to_mlx(image)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

    def preprocess(
        self,
        images: ImageInput,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        """Process images and return BatchFeature."""
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type.")

        pixel_values_list = []
        image_grid_hws = []

        for image in images:
            if isinstance(image, mx.array):
                arr = image
                if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                    arr = arr.transpose(1, 2, 0)
                if arr.dtype in [mx.float32, mx.float16, mx.bfloat16]:
                    arr = (arr * 255).astype(mx.uint8)
                h, w, _ = arr.shape
                flat_data = arr.reshape(-1).tolist()
                image = Image.frombytes("RGB", (w, h), bytes(flat_data))

            patches, grid_hw = self._preprocess(image)
            pixel_values_list.append(patches)
            image_grid_hws.append(grid_hw)

        pixel_values = mx.concatenate(pixel_values_list, axis=0)
        image_grid_hws = mx.array(image_grid_hws)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_hws": image_grid_hws},
            tensor_type=return_tensors,
        )

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


class KimiK25Processor(ProcessorMixin):
    """MLX-based processor for Kimi K2.5."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "KimiK25ImageProcessor"
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
            image_processor = KimiK25ImageProcessor()
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
            raise ValueError("Invalid input text. Please provide a string or list.")

        # Compute per-image placeholder counts
        placeholder_counts = []
        if image_grid_hws is not None:
            merge_length = (
                self.image_processor.merge_kernel_size[0]
                * self.image_processor.merge_kernel_size[1]
            )
            for grid_hw in image_grid_hws:
                placeholder_counts.append(
                    int(mx.prod(grid_hw).item()) // merge_length
                )

        # Tokenize text, then expand image placeholders at the token-ID level.
        # This avoids a TikToken bug where very long runs of the same special
        # token are partially byte-pair-encoded instead of matched as specials.
        if text is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            all_input_ids = []
            img_idx = 0
            for t in text:
                ids = self.tokenizer.encode(t)
                if placeholder_counts:
                    expanded = []
                    for tok in ids:
                        if tok == image_token_id and img_idx < len(placeholder_counts):
                            expanded.extend(
                                [image_token_id] * placeholder_counts[img_idx]
                            )
                            img_idx += 1
                        else:
                            expanded.append(tok)
                    ids = expanded
                all_input_ids.append(ids)

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

            text_inputs = {
                "input_ids": mx.array(padded_input_ids),
                "attention_mask": mx.array(attention_masks),
            }
        else:
            text_inputs = {}

        data = {**text_inputs, **image_inputs}
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id is not None:
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
            from jinja2 import Environment
            from jinja2.ext import loopcontrols
        except ImportError:
            raise ImportError("jinja2 is required for apply_chat_template")

        env = Environment(extensions=[loopcontrols])
        template = env.from_string(chat_template)
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
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)
        _ensure_gpt2_bytes_to_unicode()

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Read processor_config.json for chat template
        proc_kwargs = {}
        proc_cfg_path = model_path / "processor_config.json" if is_local else None
        if proc_cfg_path and proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("chat_template",):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]

        # Read image processor config from preprocessor_config.json and config.json
        image_processor_config = {}
        try:
            # Read preprocessor_config.json (has media_proc_cfg for K2.5)
            if is_local:
                preproc_path = model_path / "preprocessor_config.json"
            else:
                preproc_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "preprocessor_config.json"
                    )
                )
            if preproc_path.exists():
                with open(preproc_path, "r", encoding="utf-8") as f:
                    preproc_cfg = json.load(f)
                media_cfg = preproc_cfg.get("media_proc_cfg", {})
                if "patch_size" in media_cfg:
                    image_processor_config["patch_size"] = media_cfg["patch_size"]
                if "in_patch_limit" in media_cfg:
                    image_processor_config["in_token_limit"] = media_cfg[
                        "in_patch_limit"
                    ]
                if "image_mean" in media_cfg:
                    image_processor_config["image_mean"] = tuple(
                        media_cfg["image_mean"]
                    )
                if "image_std" in media_cfg:
                    image_processor_config["image_std"] = tuple(media_cfg["image_std"])
                if "merge_kernel_size" in media_cfg:
                    mks = media_cfg["merge_kernel_size"]
                    if isinstance(mks, int):
                        mks = [mks, mks]
                    image_processor_config["merge_kernel_size"] = mks
                if "patch_limit_on_one_side" in media_cfg:
                    image_processor_config["patch_limit_on_one_side"] = media_cfg[
                        "patch_limit_on_one_side"
                    ]
        except Exception:
            pass

        # Fallback: read vision_config from config.json
        if not image_processor_config:
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
                    vc = config.vision_config
                    if hasattr(vc, "patch_size"):
                        image_processor_config["patch_size"] = vc.patch_size
                    if hasattr(vc, "merge_kernel_size"):
                        image_processor_config["merge_kernel_size"] = (
                            vc.merge_kernel_size
                        )
            except Exception:
                pass

        image_processor = KimiK25ImageProcessor(**image_processor_config)

        # Load chat template
        chat_template = proc_kwargs.pop("chat_template", None)
        if chat_template is None:
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
            **proc_kwargs,
        )


from ..base import install_auto_processor_patch

install_auto_processor_patch("kimi_k25", KimiK25Processor)
