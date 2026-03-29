import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch

_IMAGE_MEAN = [0.5, 0.5, 0.5]
_IMAGE_STD = [0.5, 0.5, 0.5]

CATEGORY_PROMPTS = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}


def make_ocr_prompt(category: str = "plain") -> str:
    instruction = CATEGORY_PROMPTS.get(
        category.strip().lower(), CATEGORY_PROMPTS["plain"]
    )
    return f"<|image|>{instruction}\n<|OCR_PLAIN|>"


def _resize_if_necessary(
    image: Image.Image,
    shortest: int = 64,
    longest: int = 1024,
) -> Image.Image:
    w, h = image.size
    ar = w / h
    if shortest <= w <= longest and shortest <= h <= longest:
        return image
    is_vert = w < h
    if w < shortest or h < shortest:
        if is_vert:
            new_w, new_h = shortest, int(shortest / ar)
        else:
            new_h, new_w = shortest, int(shortest * ar)
    else:
        if is_vert:
            new_w = longest
            new_h = int(new_w / ar)
        else:
            new_h = longest
            new_w = int(new_h * ar)
    if new_w > longest:
        new_w = longest
        new_h = int(new_w / ar)
    if new_h > longest:
        new_h = longest
        new_w = int(new_h * ar)
    return image.resize((new_w, new_h), Image.BICUBIC)


def _smart_resize(
    image: Image.Image,
    factor: int,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> Image.Image:
    w, h = image.size

    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor

    if (w_bar, h_bar) != (w, h):
        image = image.resize((w_bar, h_bar), Image.BICUBIC)

    return image


def preprocess_image(
    pil_image: Image.Image,
    spatial_patch_size: int = 16,
    min_image_size: int = 64,
    max_image_size: int = 1024,
) -> tuple:
    pil_image = _resize_if_necessary(pil_image, min_image_size, max_image_size)
    pil_image = pil_image.convert("RGB")
    pil_image = _smart_resize(pil_image, factor=spatial_patch_size)

    img_np = np.array(pil_image).astype(np.float32) / 255.0

    mean = np.array(_IMAGE_MEAN, dtype=np.float32)
    std = np.array(_IMAGE_STD, dtype=np.float32)
    img_np = (img_np - mean) / std

    h, w, _ = img_np.shape
    grid_h = h // spatial_patch_size
    grid_w = w // spatial_patch_size
    return img_np, grid_h, grid_w


class FalconOCRProcessor:
    def __init__(self, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self._config = config

        self.spatial_patch_size = config.get("spatial_patch_size", 16)
        self.img_id = config.get("img_id", 227)
        self.img_end_id = config.get("img_end_id", 230)
        self.image_cls_token_id = config.get("image_cls_token_id", 244)
        self.image_reg_1_token_id = config.get("image_reg_1_token_id", 245)
        self.image_reg_2_token_id = config.get("image_reg_2_token_id", 246)
        self.image_reg_3_token_id = config.get("image_reg_3_token_id", 247)
        self.image_reg_4_token_id = config.get("image_reg_4_token_id", 248)

        self._image_prefix_ids = [
            self.image_cls_token_id,
            self.image_reg_1_token_id,
            self.image_reg_2_token_id,
            self.image_reg_3_token_id,
            self.image_reg_4_token_id,
        ]

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("use_fast", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        if is_local:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            config_file = model_path / "config.json"
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
            try:
                from huggingface_hub import hf_hub_download

                cfg_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
                config = json.loads(Path(cfg_path).read_text())
            except Exception:
                config = {}

        return cls(tokenizer, config)

    def apply_chat_template(self, messages=None, *args, **kwargs):
        if messages is not None:
            wrapped = []
            for m in messages:
                if isinstance(m, str):
                    wrapped.append({"role": "user", "content": make_ocr_prompt(m)})
                elif isinstance(m, dict) and m.get("role") == "user":
                    wrapped.append(
                        {**m, "content": make_ocr_prompt(m.get("content", ""))}
                    )
                else:
                    wrapped.append(m)
            messages = wrapped
        return self.tokenizer.apply_chat_template(messages, *args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __call__(
        self,
        text=None,
        images=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        if images is None or (isinstance(images, list) and len(images) == 0):
            text_inputs = self.tokenizer(text, padding=padding, **kwargs)
            return BatchFeature(data=dict(text_inputs), tensor_type=return_tensors)

        if not isinstance(images, list):
            images = [images]

        text = [make_ocr_prompt(t) for t in text]

        pixel_list = []
        grid_hws = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert("RGB")
            pv, gh, gw = preprocess_image(img, self.spatial_patch_size)
            pixel_list.append(pv)
            grid_hws.append([gh, gw])

        all_ids = []
        img_offset = 0
        for t in text:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            n_img = sum(1 for tid in token_ids if tid == self.img_id)
            sample_grids = grid_hws[img_offset : img_offset + n_img]
            all_ids.append(self._expand_image_tokens(token_ids, sample_grids))
            img_offset += n_img

        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        attention_masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids) if padding else 0
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        data = {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
            "pixel_values": np.stack(pixel_list),
            "image_grid_hw": np.array(grid_hws, dtype=np.int32),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def process(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: bool = True,
        return_tensors: str = "mlx",
        **kwargs,
    ) -> Dict[str, Any]:
        result = self(
            text=text,
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs,
        )
        return dict(result)

    def _expand_image_tokens(
        self, token_ids: List[int], grid_hws: List[List[int]]
    ) -> List[int]:
        expanded = []
        img_idx = 0

        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.img_id and img_idx < len(grid_hws):
                gh, gw = grid_hws[img_idx]
                n_patches = gh * gw
                expanded.extend(self._image_prefix_ids)
                expanded.extend([self.img_id] * n_patches)
                expanded.append(self.img_end_id)
                img_idx += 1
            else:
                expanded.append(token_ids[i])
            i += 1

        return expanded


def _load_image(image):
    from mlx_vlm.utils import load_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return load_image(image)


def generate_with_layout(
    model,
    processor,
    *,
    image,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    layout_threshold: float = 0.3,
    containment_threshold: float = 0.8,
    layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    use_tqdm: bool = True,
) -> List[dict]:
    """Layout detection + per-region OCR using the standard generate flow.

    Each detected region is OCR'd via ``generate(model, processor, ...)``.
    """
    from mlx_vlm import generate as mlx_generate

    from .layout import LAYOUT_TO_OCR_CATEGORY, LayoutDetector, crop_region

    pil_img = _load_image(image)

    detector = LayoutDetector(layout_model)
    detections = detector.detect(
        [pil_img],
        threshold=layout_threshold,
        batch_size=1,
        containment_threshold=containment_threshold,
    )[0]
    detector.unload()

    _only_image = (
        len(detections) == 1
        and LAYOUT_TO_OCR_CATEGORY.get(detections[0]["category"].strip().lower())
        is None
    )
    if not detections or _only_image:
        output = mlx_generate(
            model,
            processor,
            "plain",
            image=pil_img,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        w, h = pil_img.size
        return [
            {
                "category": "plain",
                "bbox": [0, 0, w, h],
                "score": 1.0,
                "text": output.text.strip(),
            }
        ]

    crop_queue = []
    for det_idx, det in enumerate(detections):
        ocr_cat = LAYOUT_TO_OCR_CATEGORY.get(det["category"].strip().lower())
        if ocr_cat is None:
            continue
        crop = crop_region(pil_img, det["bbox"])
        if crop is None:
            continue
        crop_queue.append({"crop": crop, "det_idx": det_idx, "ocr_category": ocr_cat})

    iterator = crop_queue
    if use_tqdm:
        try:
            from tqdm import tqdm

            iterator = tqdm(crop_queue, desc="OCR", unit="crop")
        except ImportError:
            pass

    results: List[dict] = []
    for seq in iterator:
        output = mlx_generate(
            model,
            processor,
            seq["ocr_category"],
            image=seq["crop"],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        det = detections[seq["det_idx"]]
        results.append(
            {
                "category": det["category"],
                "bbox": det["bbox"],
                "score": det["score"],
                "text": output.text.strip(),
            }
        )

    return results


install_auto_processor_patch("falcon_ocr", FalconOCRProcessor)

__all__ = [
    "FalconOCRProcessor",
    "CATEGORY_PROMPTS",
    "make_ocr_prompt",
    "generate_with_layout",
]
