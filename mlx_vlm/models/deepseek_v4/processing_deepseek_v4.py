import base64
import io
import math
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
COMPRESS_PAD_TO = 4

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


def grid_tokens(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    n_llm_h = math.ceil((height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    aspect_ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / aspect_ratio + 0.25) - 0.5
    max_h_float = max_w_float * aspect_ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("DeepSeek-V4 image token budget is too small")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size

    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            budget,
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[np.ndarray, np.ndarray]:
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = np.array(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=np.int32,
    )
    order = (
        np.arange(rows * row_len)
        .reshape(rows // 2, 2, row_len)
        .transpose(0, 2, 1)
        .reshape(-1)
    )
    image_idx = np.full((rows * row_len,), -1, dtype=np.int32)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w, dtype=np.int32
    ).reshape(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = np.concatenate(
        [
            np.full((compress_pad,), IMAGE_PAD, dtype=np.int32),
            np.array([IMAGE_START], dtype=np.int32),
            types[order],
            np.full((pad_last,), IMAGE_PAD, dtype=np.int32),
            np.array([IMAGE_END], dtype=np.int32),
        ]
    )
    return types, perm


def _image_from_input(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, dict):
        url = image.get("url") or image.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        data = image.get("data")
        if data is None and isinstance(image.get("source"), dict):
            source = image["source"]
            data = source.get("data")
            url = url or source.get("url")
        if data is not None:
            raw = data if isinstance(data, bytes) else base64.b64decode(data)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        image = url
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        if image.startswith("data:"):
            header, _, payload = image.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        if image.startswith(("http://", "https://")):
            with urlopen(image, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(image).convert("RGB")
    raise ValueError(f"Unsupported DeepSeek-V4 image input: {type(image).__name__}")


def preprocess_image(
    image,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: Optional[int],
) -> tuple[np.ndarray, int, int, int, int]:
    image = _image_from_input(image)
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
        max_n_token,
    )
    n_vit_h = best_height // patch_size
    n_vit_w = best_width // patch_size
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def load_deepseek_v4_chat_template(model_path, **kwargs) -> Optional[str]:
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


class DeepseekV4Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer,
        chat_template: Optional[str] = None,
        vocab_size: int = 129280,
        vision_patch_size: int = 14,
        vision_downsample_ratio: int = 3,
        vision_max_n_token: int = 384,
        vision_min_pixels: int = 147456,
        vision_max_wh_ratio: Optional[int] = 8,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.vision_patch_size = vision_patch_size
        self.vision_downsample_ratio = vision_downsample_ratio
        self.vision_max_n_token = vision_max_n_token
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_wh_ratio = vision_max_wh_ratio
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

    @staticmethod
    def _content_to_text(content):
        if not isinstance(content, list):
            return content
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type in ("text", "input_text"):
                parts.append(block.get("text", block.get("content", "")))
            elif block_type in ("image", "image_url", "input_image"):
                parts.append(IMAGE_PLACEHOLDER)
        return "".join(parts)

    def apply_chat_template(self, conversation, *args, **kwargs):
        if isinstance(conversation, dict):
            conversation = [conversation]
        normalized = []
        for message in conversation:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            message = dict(message)
            message["content"] = self._content_to_text(message.get("content", ""))
            normalized.append(message)
        return self.tokenizer.apply_chat_template(normalized, *args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @staticmethod
    def _as_samples(text, images):
        texts = [text] if isinstance(text, str) else list(text)
        if images is None:
            return texts, [[] for _ in texts]
        images = [images] if not isinstance(images, (list, tuple)) else list(images)
        if (
            len(texts) > 1
            and len(images) == len(texts)
            and all(isinstance(image, (list, tuple)) for image in images)
        ):
            return texts, [
                list(image) if isinstance(image, (list, tuple)) else [image]
                for image in images
            ]
        if len(texts) == 1:
            return texts, [images]
        return texts, [images]

    def __call__(
        self,
        text,
        images=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        return_tensors="mlx",
        **kwargs,
    ):
        if images is None:
            return self.tokenizer(
                text,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs,
            )

        texts, image_samples = self._as_samples(text, images)
        if len(image_samples) != len(texts):
            flat_images = (
                image_samples[0]
                if len(image_samples) == 1
                and isinstance(image_samples[0], (list, tuple))
                else image_samples
            )
            image_samples = []
            image_offset = 0
            image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
            for prompt in texts:
                count = self.tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens
                ).count(image_token_id)
                image_samples.append(flat_images[image_offset : image_offset + count])
                image_offset += count
            if image_offset != len(flat_images):
                raise ValueError(
                    f"Found {image_offset} image placeholders but got "
                    f"{len(flat_images)} images"
                )

        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if image_token_id is None or image_token_id == getattr(
            self.tokenizer, "unk_token_id", None
        ):
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")

        token_rows = []
        sample_layouts = []
        all_patches = []
        image_grid_hw = []
        image_sample_indices = []
        image_offsets = []
        image_types = []
        image_type_offsets = [0]
        image_permutations = []

        for sample_idx, (prompt, sample_images) in enumerate(zip(texts, image_samples)):
            prompt_tokens = self.tokenizer.encode(
                prompt, add_special_tokens=add_special_tokens
            )
            placeholders = sum(token == image_token_id for token in prompt_tokens)
            if placeholders != len(sample_images):
                raise ValueError(
                    f"Found {placeholders} image placeholders but got "
                    f"{len(sample_images)} images"
                )

            tokens = []
            layouts = []
            image_iter = iter(sample_images)
            for token in prompt_tokens:
                if token != image_token_id:
                    tokens.append(token)
                    continue
                patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(
                    next(image_iter),
                    self.vision_patch_size,
                    self.vision_downsample_ratio,
                    self.vision_max_n_token,
                    self.vision_min_pixels,
                    self.vision_max_wh_ratio,
                )
                types, perm = build_image_block(n_llm_h, n_llm_w, len(tokens))
                layouts.append((len(tokens), types))
                tokens.extend((self.vocab_size + types).tolist())
                all_patches.append(patches)
                image_grid_hw.append((n_vit_h, n_vit_w))
                image_sample_indices.append(sample_idx)
                image_types.append(types)
                image_type_offsets.append(image_type_offsets[-1] + len(types))
                image_permutations.append(perm)
            token_rows.append(tokens)
            sample_layouts.append(layouts)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        max_length = max(len(row) for row in token_rows)
        padded_rows = []
        attention_mask = []
        for row, layouts in zip(token_rows, sample_layouts):
            pad_length = max_length - len(row) if padding else 0
            left_pad = pad_length if padding_side == "left" else 0
            right_pad = pad_length - left_pad
            padded_rows.append(
                [pad_token_id] * left_pad + row + [pad_token_id] * right_pad
            )
            attention_mask.append([0] * left_pad + [1] * len(row) + [0] * right_pad)
            image_offsets.extend(left_pad + start for start, _ in layouts)

        output = {
            "input_ids": np.asarray(padded_rows, dtype=np.int32),
            "attention_mask": np.asarray(attention_mask, dtype=np.int32),
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_hw": np.asarray(image_grid_hw, dtype=np.int32),
            "image_sample_indices": np.asarray(image_sample_indices, dtype=np.int32),
            "image_offsets": np.asarray(image_offsets, dtype=np.int32),
            "image_types": np.concatenate(image_types),
            "image_type_offsets": np.asarray(image_type_offsets, dtype=np.int32),
            "image_permutations": np.concatenate(image_permutations),
        }
        if return_tensors == "mlx":
            output = {key: mx.array(value) for key, value in output.items()}
        return output

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
            chat_template = load_deepseek_v4_chat_template(
                pretrained_model_name_or_path,
                **kwargs,
            )
        config_values = {}
        local_config = Path(pretrained_model_name_or_path) / "config.json"
        try:
            if local_config.exists():
                import json

                config_values = json.loads(local_config.read_text(encoding="utf-8"))
            else:
                from huggingface_hub import hf_hub_download

                download_kwargs = {
                    key: kwargs[key]
                    for key in ("revision", "token", "local_files_only")
                    if key in kwargs
                }
                config_path = hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename="config.json",
                    **download_kwargs,
                )
                import json

                config_values = json.loads(
                    Path(config_path).read_text(encoding="utf-8")
                )
        except Exception:
            config_values = {}
        processor_fields = (
            "vocab_size",
            "vision_patch_size",
            "vision_downsample_ratio",
            "vision_max_n_token",
            "vision_min_pixels",
            "vision_max_wh_ratio",
        )
        return cls(
            tokenizer=tokenizer,
            chat_template=chat_template,
            **{
                key: config_values[key]
                for key in processor_fields
                if key in config_values
            },
        )


install_auto_processor_patch("deepseek_v4", DeepseekV4Processor)

__all__ = [
    "DEFAULT_CHAT_TEMPLATE",
    "IMAGE_PLACEHOLDER",
    "DeepseekV4Processor",
    "build_image_block",
    "grid_tokens",
    "load_deepseek_v4_chat_template",
    "preprocess_image",
    "safe_resize",
]
