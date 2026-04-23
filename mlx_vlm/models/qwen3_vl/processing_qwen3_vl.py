"""
Processor class for Qwen3VL.

Adapted from HuggingFace Transformers:
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_processing_utils import BaseVideoProcessor

from ..base import load_chat_template, to_mlx


def _smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> Tuple[int, int]:
    """Compute target (height, width) to fit a video into the token budget.

    Mirrors HF's ``smart_resize`` exactly (no torch).
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize_video_frames(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bicubic resize each frame of a ``(T, C, H, W)`` video."""
    from PIL import Image

    T, C, H, W = video.shape
    if target_h == H and target_w == W:
        return video
    out = np.empty((T, C, target_h, target_w), dtype=video.dtype)
    for i, frame in enumerate(video):
        arr = np.transpose(frame, (1, 2, 0))
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((target_w, target_h), resample=Image.BICUBIC)
        out[i] = np.transpose(np.array(pil), (2, 0, 1))
    return out


def _smart_resize_image(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Image variant of ``smart_resize`` — ports HF's qwen2_vl ``smart_resize``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_numpy_image(img) -> np.ndarray:
    """Coerce a PIL.Image / path / numpy to a ``(C, H, W)`` uint8 array."""
    from PIL import Image

    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
        arr = np.array(img)  # (H, W, C)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] in (1, 3, 4) and arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    if arr.shape[0] == 4:
        arr = arr[:3]
    return arr


class Qwen3VLImageProcessor(ImageProcessingMixin):
    """Numpy port of Qwen2/3-VL image processor (torch-free).

    Produces:
      - ``pixel_values``: ``(sum_i grid_t*grid_h*grid_w, C * tps * ps * ps)``
        where images have ``grid_t=1`` (duplicated along the temporal axis to
        match the model's ``temporal_patch_size``)
      - ``image_grid_thw``: ``(num_images, 3)``
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
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
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def fetch_images(self, images):
        if not isinstance(images, list):
            images = [images]
        return [_to_numpy_image(img) for img in images]

    def _process_one(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        C, H, W = image.shape
        resized_h, resized_w = _smart_resize_image(
            H,
            W,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        # Bicubic resize via PIL (same pattern as the video path).
        frame = _resize_video_frames(image[None, ...], resized_h, resized_w)[0]

        img = frame.astype(np.float32)
        if self.do_rescale and image.dtype == np.uint8:
            img = img * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            img = (img - mean) / std

        # Duplicate along T so grid_t * tps frames match the model's expectation.
        patches = np.repeat(img[None, None, ...], self.temporal_patch_size, axis=1)

        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size
        grid_t = 1
        grid_h = resized_h // ps
        grid_w = resized_w // ps

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
        imgs = [
            (
                img
                if (isinstance(img, np.ndarray) and img.ndim == 3)
                else _to_numpy_image(img)
            )
            for img in images
        ]
        all_patches = []
        all_thw = []
        for v in imgs:
            patches, thw = self._process_one(v)
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.array(all_thw, dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class Qwen3VLVideoProcessor(BaseVideoProcessor):
    """Numpy port of ``transformers.Qwen3VLVideoProcessor``.

    Produces:
      - ``pixel_values_videos``: shape
        ``(sum_i grid_t_i * grid_h_i * grid_w_i, C * tps * ps * ps)``
      - ``video_grid_thw``: ``(num_videos, 3)`` of ``(grid_t, grid_h, grid_w)``

    The upstream implementation hard-requires torch/torchvision via
    ``BaseVideoProcessor``; this port reproduces the same outputs with
    numpy + PIL only.
    """

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 128 * 32 * 32,
        max_pixels: int = 32 * 32 * 768,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 768,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def _process_one(self, video: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if video.ndim != 4:
            raise ValueError(
                f"Expected video as (T, C, H, W), got shape {video.shape}."
            )
        T, C, H, W = video.shape
        if C == 1 and self.do_convert_rgb:
            video = np.repeat(video, 3, axis=1)
            C = 3

        resized_h, resized_w = _smart_resize_video(
            num_frames=T,
            height=H,
            width=W,
            temporal_factor=self.temporal_patch_size,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        video = _resize_video_frames(video, resized_h, resized_w)

        video_f = video.astype(np.float32)
        if self.do_rescale and video.dtype == np.uint8:
            video_f = video_f * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[None, :, None, None]
            std = np.array(self.image_std, dtype=np.float32)[None, :, None, None]
            video_f = (video_f - mean) / std

        pad = (-video_f.shape[0]) % self.temporal_patch_size
        if pad:
            video_f = np.concatenate(
                [video_f, np.repeat(video_f[-1:], pad, axis=0)], axis=0
            )

        T_padded = video_f.shape[0]
        grid_t = T_padded // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size

        patches = video_f[None, ...]  # (1, T_padded, C, H, W)
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

    def __call__(self, videos, **kwargs):
        if not isinstance(videos, list):
            videos = [videos]
        all_patches = []
        all_thw = []
        for v in videos:
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            patches, thw = self._process_one(v)
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values_videos": np.concatenate(all_patches, axis=0),
            "video_grid_thw": np.array(all_thw, dtype=np.int64),
        }


def _load_qwen_vl_json(pretrained_model_name_or_path, relative_name: str):
    """Load ``<checkpoint>/<relative_name>`` from disk or the Hub, or None."""
    import json
    from pathlib import Path

    local = Path(pretrained_model_name_or_path) / relative_name
    if local.exists():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download

        fetched = Path(hf_hub_download(pretrained_model_name_or_path, relative_name))
        return json.loads(fetched.read_text())
    except Exception:
        return None


def _qwen_vl_image_kwargs(pretrained_model_name_or_path, default_patch_size: int = 16):
    """Read Qwen-VL image processor kwargs out of a checkpoint."""
    proc_cfg = (
        _load_qwen_vl_json(pretrained_model_name_or_path, "processor_config.json") or {}
    )
    raw = (
        _load_qwen_vl_json(pretrained_model_name_or_path, "preprocessor_config.json")
        or {}
    )
    raw.update(proc_cfg.get("image_processor", {}) or {})
    out = {"patch_size": default_patch_size}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if k in raw:
            out[k] = raw[k]
    size = raw.get("size", {})
    if "shortest_edge" in size:
        out["min_pixels"] = size["shortest_edge"]
    if "longest_edge" in size:
        out["max_pixels"] = size["longest_edge"]
    # legacy flat-key forms (some Qwen2 checkpoints)
    if "min_pixels" in raw:
        out["min_pixels"] = raw["min_pixels"]
    if "max_pixels" in raw:
        out["max_pixels"] = raw["max_pixels"]
    return out


def _qwen_vl_video_kwargs(pretrained_model_name_or_path, default_patch_size: int = 16):
    """Read Qwen-VL video processor kwargs out of a checkpoint."""
    raw = _load_qwen_vl_json(
        pretrained_model_name_or_path, "video_preprocessor_config.json"
    )
    if raw is None:
        # Older checkpoints (e.g. qwen2_vl) keep video settings inside
        # preprocessor_config.json alongside the image settings.
        raw = (
            _load_qwen_vl_json(
                pretrained_model_name_or_path, "preprocessor_config.json"
            )
            or {}
        )
    out = {"patch_size": default_patch_size}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "fps",
        "min_frames",
        "max_frames",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if k in raw:
            out[k] = raw[k]
    size = raw.get("size", {})
    if "shortest_edge" in size:
        out["min_pixels"] = size["shortest_edge"]
    if "longest_edge" in size:
        out["max_pixels"] = size["longest_edge"]
    if "min_pixels" in raw:
        out["min_pixels"] = raw["min_pixels"]
    if "max_pixels" in raw:
        out["max_pixels"] = raw["max_pixels"]
    return out


class Qwen3VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    # HF's ProcessorMixin resolves expected base classes at runtime; in torch-
    # free environments it picks up dummy classes from
    # ``transformers.utils.dummy_torchvision_objects``, so our (real) numpy
    # subclasses fail ``isinstance``. Skip that validation — our processors
    # are duck-typed to the interfaces the call sites use.
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

        self.vision_start_token = (
            "<|vision_start|>"
            if not hasattr(tokenizer, "vision_start_token")
            else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>"
            if not hasattr(tokenizer, "vision_end_token")
            else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
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
        image_inputs = {}
        videos_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_grid_thw = None

        if videos is not None:
            _video_proc = self.video_processor or self.image_processor
            videos_inputs = _video_proc(videos=videos)
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            _video_proc = self.video_processor or self.image_processor
            merge_length = _video_proc.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * num_video_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **kwargs)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **videos_inputs})
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + image_processor_input_names
                + ["mm_token_type_ids"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        image_processor = Qwen3VLImageProcessor(
            **_qwen_vl_image_kwargs(
                pretrained_model_name_or_path, default_patch_size=16
            )
        )
        video_processor = Qwen3VLVideoProcessor(
            **_qwen_vl_video_kwargs(
                pretrained_model_name_or_path, default_patch_size=16
            )
        )

        proc_cfg = (
            _load_qwen_vl_json(pretrained_model_name_or_path, "processor_config.json")
            or {}
        )
        chat_template = proc_cfg.get(
            "chat_template", getattr(tokenizer, "chat_template", None)
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )


__all__ = ["Qwen3VLProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("qwen3_vl", Qwen3VLProcessor)
