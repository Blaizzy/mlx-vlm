import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch
from .config import ModelConfig

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

DEFAULT_CHAT_TEMPLATE = (
    "{{- '<｜begin▁of▁sentence｜>' -}}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{{- messages[0]['content'] -}}"
    "{%- set start = 1 -%}"
    "{%- else -%}"
    "{%- set start = 0 -%}"
    "{%- endif -%}"
    "{%- for m in messages[start:] -%}"
    "{%- if m['role'] == 'user' -%}"
    "{{- '<｜User｜>' + m['content'] -}}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{{- '<｜Assistant｜>' + m['content'] + '<｜end▁of▁sentence｜>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<｜Assistant｜></think>' -}}"
    "{%- endif -%}"
)


@dataclass
class ImageInput:
    """One image's ViT patches plus its LLM span layout."""

    start: int
    patches: mx.array
    n_vit_h: int
    n_vit_w: int
    types: List[int]


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> Tuple[int, int]:
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> Tuple[int, int]:
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> Tuple[int, int, int, int]:
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(
    width: int, height: int, config: ModelConfig
) -> Tuple[int, int, int, int]:
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = config.vision_patch_size
    if (
        config.vision_max_wh_ratio is not None
        and width > height * config.vision_max_wh_ratio
    ):
        width = int(height * config.vision_max_wh_ratio)
    if 0 < width * height < config.vision_min_pixels:
        ratio = (config.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        config.vision_downsample_ratio,
        config.vision_max_image_tokens,
    )


def load_image_bytes(record) -> bytes:
    """Load image bytes from raw/base64 data, a source dict, URL, or path."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def load_image(record, config: ModelConfig) -> Tuple[mx.array, int, int, int, int]:
    """Load and transform one image record into ViT patches."""
    p = config.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, config
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if (
        config.vision_max_wh_ratio is not None
        and image.width >= config.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(n_vit_h * n_vit_w, 3, p, p)
    return mx.array(patches), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> List[int]:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return types


def prepare_vl_inputs(
    prompt_tokens: List[int],
    images,
    config: ModelConfig,
) -> Tuple[List[int], List[int], List[ImageInput]]:
    """Expand each image placeholder token into its image span.

    Returns (tokens, token_types, image_inputs). Image-span positions carry
    `image_token_id` in tokens and are distinguished only by token types
    (TEXT elsewhere). `image_inputs` is empty when the prompt has no images.
    """
    image_inputs = []
    if sum(token == config.image_token_id for token in prompt_tokens) != len(images):
        raise ValueError("Image placeholder count does not match the image count")
    if prompt_tokens.count(config.image_token_id) and not config.vision_num_layers:
        raise ValueError(
            "The model config has no vision tower but the prompt contains images"
        )

    tokens, token_types = [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != config.image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
            next(image_iter), config
        )
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [config.image_token_id] * len(types)
        token_types += types
    return tokens, token_types, image_inputs


def load_deepseek_v41_chat_template(model_path, **kwargs) -> Optional[str]:
    local_path = Path(model_path)
    if local_path.exists():
        template_path = local_path / "chat_template.jinja"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return None

    try:
        from huggingface_hub import hf_hub_download

        download_kwargs = {
            key: kwargs[key]
            for key in ("revision", "token", "local_files_only")
            if key in kwargs
        }
        template_path = hf_hub_download(
            repo_id=str(model_path),
            filename="chat_template.jinja",
            **download_kwargs,
        )
        return Path(template_path).read_text(encoding="utf-8")
    except Exception:
        return None


class DeepseekV41Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        chat_template = (
            chat_template
            or getattr(tokenizer, "chat_template", None)
            or DEFAULT_CHAT_TEMPLATE
        )
        self.tokenizer.chat_template = chat_template
        super().__init__(tokenizer, chat_template=chat_template, **kwargs)

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("tokenize", False)
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        chat_template = kwargs.pop("chat_template", None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        except (AttributeError, ValueError):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        if chat_template is None:
            chat_template = load_deepseek_v41_chat_template(
                pretrained_model_name_or_path,
                **kwargs,
            )
        return cls(tokenizer=tokenizer, chat_template=chat_template)


install_auto_processor_patch("deepseek_v41", DeepseekV41Processor)

__all__ = [
    "DEFAULT_CHAT_TEMPLATE",
    "IMAGE_PLACEHOLDER",
    "DeepseekV41Processor",
    "ImageInput",
    "image_token_types",
    "load_deepseek_v41_chat_template",
    "load_image",
    "num_image_tokens",
    "plan_image_grid",
    "prepare_vl_inputs",
]
