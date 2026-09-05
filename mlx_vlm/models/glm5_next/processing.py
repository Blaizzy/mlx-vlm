"""GLM-5.3-Flash image processor: token-budget smart_resize + pad-after-scale.

`min_pixels`/`max_pixels` arguments to `smart_resize` are **token counts**
(processor `min_image_tokens`/`max_image_tokens`), multiplied internally by
`temporal_factor * factor**2`. This is not the glm_ocr/Qwen pixel-budget helper.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template
from ..qwen3_vl.processing_qwen3_vl import _resize_video_frames, _to_numpy_image


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> Tuple[int, int]:
    """Port of transformers `modular_glm5_next.smart_resize`.

    `min_pixels`/`max_pixels` are token budgets. They are converted to a pixel
    budget with `pixels_per_token = temporal_factor * factor**2`.
    """
    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, f):
        return math.ceil(value / f) * f

    def fit_within_budget(aligned_frames):
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. "
                f"At least {minimum_pixels} pixels are required for one aligned patch."
            )
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)
            pixel_budget = aligned_frames * candidate_height * candidate_width
            if pixel_budget <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)

    return aligned_height, aligned_width


def llm_image_tokens(
    height: int,
    width: int,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
    patch_expand_factor: int = 1,
) -> int:
    factor = patch_size * merge_size * patch_expand_factor
    resized_h, resized_w = smart_resize(
        num_frames=temporal_patch_size,
        height=height,
        width=width,
        temporal_factor=temporal_patch_size,
        factor=factor,
        min_pixels=min_image_tokens,
        max_pixels=max_image_tokens,
    )
    grid_h, grid_w = resized_h // patch_size, resized_w // patch_size
    return (grid_h * grid_w) // (merge_size**2)


def _resize_chw(chw: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bicubic resize of one ``(C, H, W)`` image via the shared qwen3_vl helper.

    Float inputs are resized in uint8 space and restored to ``[0, 1]`` floats,
    which is what ``_process_one`` relies on when a caller hands in floats.
    """
    if chw.shape[-2:] == (height, width):
        return chw
    if chw.dtype in (np.float32, np.float64):
        as_u8 = (chw * 255.0).clip(0, 255).astype(np.uint8)
        out = _resize_video_frames(as_u8[None], height, width)[0]
        return out.astype(np.float32) / 255.0
    return _resize_video_frames(chw[None], height, width)[0]


class Glm5NextImageProcessor(ImageProcessingMixin):
    """Numpy GLM-5.3-Flash image processor: smart_resize + pad-after-scale + patchify."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb

    def resize_hw(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Return (target_h, target_w, content_h, content_w) matching HF pad-after-scale."""
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        target_h, target_w = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=self.min_image_tokens,
            max_pixels=self.max_image_tokens,
        )
        pixels_per_token = self.temporal_patch_size * factor**2
        scale = min(target_h / height, target_w / width)
        if self.temporal_patch_size * height * width >= (
            pixels_per_token * self.min_image_tokens
        ):
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(height * scale)))
        content_w = max(1, min(target_w, math.floor(width * scale)))
        return target_h, target_w, content_h, content_w

    def _process_one(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        C, H, W = image.shape
        target_h, target_w, content_h, content_w = self.resize_hw(H, W)
        frame = image.astype(np.float32)
        if self.do_rescale and image.dtype == np.uint8:
            frame = frame * self.rescale_factor
        if (content_h, content_w) != (H, W):
            # Resize in pixel space: undo rescale for PIL, then re-apply.
            src = image if image.dtype == np.uint8 else np.clip(image, 0, 255)
            if src.dtype != np.uint8:
                src = (
                    (src * (255.0 if src.max() <= 1.5 else 1.0))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            resized = _resize_chw(src, content_h, content_w)
            frame = resized.astype(np.float32)
            if self.do_rescale:
                frame = frame * self.rescale_factor
        pad_h = target_h - content_h
        pad_w = target_w - content_w
        if pad_h or pad_w:
            frame = np.pad(frame, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0)

        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            frame = (frame - mean) / std

        patches = np.repeat(frame[None, None, ...], self.temporal_patch_size, axis=1)
        grid_t = 1
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size
        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size
        patches = patches.reshape(
            1,
            grid_t,
            tps,
            C,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(1, grid_t * grid_h * grid_w, C * tps * ps * ps)
        return flatten[0], [grid_t, grid_h, grid_w]

    def __call__(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]
        all_patches = []
        all_thw = []
        for image in images:
            if not (
                isinstance(image, np.ndarray)
                and image.ndim == 3
                and image.shape[0] in (1, 3, 4)
            ):
                image = _to_numpy_image(image)
            patches, thw = self._process_one(image)
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.array(all_thw, dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)
        factor = patch_size * merge_size * self.patch_expand_factor
        resized_h, resized_w = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        return (resized_h // patch_size) * (resized_w // patch_size)


def _load_json(pretrained_model_name_or_path, relative_name: str):
    import json

    local = Path(pretrained_model_name_or_path) / relative_name
    if local.exists():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download

        fetched = Path(hf_hub_download(pretrained_model_name_or_path, relative_name))
        return json.loads(fetched.read_text())
    except Exception:
        return None


def _image_processor_kwargs(pretrained_model_name_or_path):
    proc_cfg = _load_json(pretrained_model_name_or_path, "processor_config.json")
    raw = _load_json(pretrained_model_name_or_path, "preprocessor_config.json") or {}
    if proc_cfg:
        raw.update(proc_cfg.get("image_processor", {}) or {})
    out = {}
    for key in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "patch_expand_factor",
        "min_image_tokens",
        "max_image_tokens",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if key in raw:
            out[key] = raw[key]
    return out


class Glm5NextProcessor(ProcessorMixin):
    """Tokenizer + image processor. Expands `<|image|>` to grid-sized placeholders.

    The checkpoint's own `chat_template.jinja` (upstream revision 690b7052 or
    later) emits `<|begin_of_image|><|image|><|end_of_image|>` for image parts;
    this processor expands each `<|image|>` slot to `prod(image_grid_thw) /
    merge_size**2` tokens. Expansion splits on the original slots first, so an
    already-expanded run is never re-expanded (a `while token in text` loop
    would reconsume it and ask for extra `image_grid_thw` rows).
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    image_token = "<|image|>"
    video_token = "<|video|>"
    image_start_token = "<|begin_of_image|>"
    image_end_token = "<|end_of_image|>"
    video_start_token = "<|begin_of_video|>"
    video_end_token = "<|end_of_video|>"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.image_token = getattr(tokenizer, "image_token", self.image_token)
            self.video_token = getattr(tokenizer, "video_token", self.video_token)
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

    @staticmethod
    def format_image_slot() -> str:
        return "<|begin_of_image|><|image|><|end_of_image|>"

    @staticmethod
    def format_video_slot() -> str:
        return "<|begin_of_video|><|video|><|end_of_video|>"

    def replace_image_token(self, image_grid_thw, image_idx: int) -> str:
        merge_length = getattr(self.image_processor, "merge_size", 2) ** 2
        grid = image_grid_thw[image_idx]
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        num_image_tokens = int(np.prod(grid) // merge_length)
        return self.image_token * num_image_tokens

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        video_inputs = {}
        padding = kwargs.pop("padding", False)
        return_token_type_ids = kwargs.pop("return_token_type_ids", False)
        return_tensors = kwargs.pop("return_tensors", None)

        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs.get("image_grid_thw")
        else:
            image_grid_thw = None

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]
        text = list(text)

        if image_grid_thw is not None:
            index = 0
            n_images = int(len(image_grid_thw))
            for i in range(len(text)):
                # Replacement is `image_token * N`. Split on the original slots
                # before inserting the run; a `while token in text` loop would
                # consume the expanded placeholders and ask for extra grids.
                parts = text[i].split(self.image_token)
                n_slots = len(parts) - 1
                if n_slots == 0:
                    continue
                pieces = [parts[0]]
                for slot in range(n_slots):
                    if index >= n_images:
                        raise ValueError(
                            f"prompt has more {self.image_token!r} slots than "
                            f"image_grid_thw rows ({n_images})"
                        )
                    pieces.append(self.replace_image_token(image_grid_thw, index))
                    pieces.append(parts[slot + 1])
                    index += 1
                text[i] = "".join(pieces)
            if index != n_images:
                raise ValueError(
                    f"prompt expanded {index} image slots but image_grid_thw "
                    f"has {n_images} rows"
                )

        if videos is not None:
            raise NotImplementedError(
                "Video preprocessing is prepared at the token/mask layer only; "
                "pass image frames or wait for the video processor port."
            )

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
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
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
        from transformers import AutoTokenizer

        trust_remote_code = kwargs.pop("trust_remote_code", True)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        proc_cfg = _load_json(pretrained_model_name_or_path, "processor_config.json")
        proc_kwargs = dict(proc_cfg or {})
        proc_kwargs.pop("image_processor", None)
        proc_kwargs.pop("video_processor", None)
        proc_kwargs.pop("processor_class", None)
        image_processor = Glm5NextImageProcessor(
            **_image_processor_kwargs(pretrained_model_name_or_path)
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
            **kwargs,
        )


__all__ = [
    "Glm5NextProcessor",
    "Glm5NextImageProcessor",
    "smart_resize",
    "llm_image_tokens",
]

install_auto_processor_patch("glm5_next", Glm5NextProcessor)
