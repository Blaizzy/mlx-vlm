import math
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import _resize_video_frames, _to_numpy_image


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
) -> Tuple[int, int]:
    """Torch-free port of Transformers' GLM-4V smart resize helper."""
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Glm46VImageProcessor(ImageProcessingMixin):
    """Numpy/PIL GLM-4.6V image processor matching Transformers' patch layout."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 112 * 112,
        max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        use_transformers_backend: bool = True,
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
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb
        self.use_transformers_backend = use_transformers_backend
        self._transformers_image_processor = None

    def _get_transformers_image_processor(self):
        if not self.use_transformers_backend:
            return None
        if self._transformers_image_processor is not None:
            return self._transformers_image_processor

        try:
            from transformers.models.glm46v.image_processing_glm46v import (
                Glm46VImageProcessor as HFGlm46VImageProcessor,
            )

            self._transformers_image_processor = HFGlm46VImageProcessor(
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
                size={
                    "shortest_edge": self.min_pixels,
                    "longest_edge": self.max_pixels,
                },
                do_rescale=self.do_rescale,
                rescale_factor=self.rescale_factor,
                do_normalize=self.do_normalize,
                image_mean=self.image_mean,
                image_std=self.image_std,
                do_convert_rgb=self.do_convert_rgb,
            )
        except Exception:
            self._transformers_image_processor = None
        return self._transformers_image_processor

    def _process_with_transformers_backend(self, images):
        image_processor = self._get_transformers_image_processor()
        if image_processor is None:
            return None

        try:
            import torch

            outputs = image_processor(images=images)
            return {
                name: (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                for name, value in outputs.items()
            }
        except Exception:
            return None

    def _process_one(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        C, H, W = image.shape
        factor = self.patch_size * self.merge_size
        resized_h, resized_w = smart_resize(
            num_frames=self.temporal_patch_size,
            height=H,
            width=W,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        frame = _resize_video_frames(image[None, ...], resized_h, resized_w)[0]
        patches = frame.astype(np.float32)
        if self.do_rescale and frame.dtype == np.uint8:
            patches = patches * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            patches = (patches - mean) / std

        patches = np.repeat(patches[None, None, ...], self.temporal_patch_size, axis=1)

        grid_t = 1
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
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
        transformers_outputs = self._process_with_transformers_backend(images)
        if transformers_outputs is not None:
            return transformers_outputs

        if not isinstance(images, list):
            images = [images]
        all_patches = []
        all_thw = []
        for image in images:
            if not (isinstance(image, np.ndarray) and image.ndim == 3):
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


def _load_glm_ocr_json(pretrained_model_name_or_path, relative_name: str):
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


def _glm_ocr_image_kwargs(pretrained_model_name_or_path):
    proc_cfg = _load_glm_ocr_json(
        pretrained_model_name_or_path, "processor_config.json"
    )
    raw = _load_glm_ocr_json(pretrained_model_name_or_path, "preprocessor_config.json")
    raw = raw or {}
    if proc_cfg:
        raw.update(proc_cfg.get("image_processor", {}) or {})

    out = {}
    for key in (
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
        if key in raw:
            out[key] = raw[key]

    size = raw.get("size") or {}
    if size.get("shortest_edge") is not None:
        out["min_pixels"] = size["shortest_edge"]
    if size.get("longest_edge") is not None:
        out["max_pixels"] = size["longest_edge"]
    return out


class GlmOcrProcessor(ProcessorMixin):
    """
    Processor for GLM-OCR that wraps an image processor and tokenizer.

    Handles:
    - Image preprocessing via image_processor
    - Token replacement for image/video placeholders based on grid dimensions
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

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

        self.image_token = "<|image|>"
        self.video_token = "<|video|>"

        if tokenizer is not None:
            self.image_token = getattr(tokenizer, "image_token", "<|image|>")
            self.video_token = getattr(tokenizer, "video_token", "<|video|>")

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

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process images/videos and text for the model.

        Args:
            images: Single image or list of images (PIL.Image, np.ndarray, etc.)
            text: Single text or list of texts
            videos: Video inputs (optional)
            **kwargs: Additional arguments passed to image_processor and tokenizer

        Returns:
            BatchFeature with:
                - input_ids: Token IDs with image/video placeholders expanded
                - attention_mask: Attention mask
                - pixel_values: Processed image/video patches
                - image_grid_thw: Grid dimensions for each image
                - video_grid_thw: Grid dimensions for each video (if videos provided)
        """
        image_inputs = {}
        video_inputs = {}
        image_grid_thw = None
        video_grid_thw = None

        padding = kwargs.pop("padding", False)
        return_token_type_ids = kwargs.pop("return_token_type_ids", False)
        return_tensors = kwargs.pop("return_tensors", None)

        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs.get("image_grid_thw")

        if videos is not None:
            if hasattr(self, "video_processor") and self.video_processor is not None:
                video_inputs = self.video_processor(videos=videos, **kwargs)
                video_grid_thw = video_inputs.get("video_grid_thw")

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        text = [t for t in text]

        merge_size = getattr(self.image_processor, "merge_size", 2)
        if hasattr(self.image_processor, "spatial_merge_size"):
            merge_size = self.image_processor.spatial_merge_size
        merge_length = merge_size**2

        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid = image_grid_thw[index]
                    if hasattr(grid, "tolist"):
                        grid = grid.tolist()
                    num_image_tokens = int(np.prod(grid) // merge_length)

                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    grid = video_grid_thw[video_index]
                    if hasattr(grid, "tolist"):
                        grid = grid.tolist()

                    num_frames = grid[0]
                    num_tokens_per_frame = int(
                        np.prod(grid) // merge_length // num_frames
                    )

                    video_structure = ""
                    for frame_idx in range(num_frames):
                        frame_structure = self.image_token * num_tokens_per_frame
                        video_structure += frame_structure

                    text[i] = text[i].replace(self.video_token, video_structure, 1)
                    video_index += 1

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
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        """Apply chat template using the tokenizer."""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        """Return combined input names from tokenizer and image processor."""
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
        """Load processor from pretrained model path."""
        from transformers import AutoTokenizer

        trust_remote_code = kwargs.pop("trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        from ..base import load_chat_template

        load_chat_template(tokenizer, pretrained_model_name_or_path)

        proc_cfg = _load_glm_ocr_json(
            pretrained_model_name_or_path, "processor_config.json"
        )
        proc_kwargs = proc_cfg or {}
        proc_kwargs.pop("image_processor", None)
        proc_kwargs.pop("processor_class", None)
        image_processor = Glm46VImageProcessor(
            **_glm_ocr_image_kwargs(pretrained_model_name_or_path)
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
            **kwargs,
        )


__all__ = ["GlmOcrProcessor", "Glm46VImageProcessor", "smart_resize"]

# Register the processor with AutoProcessor for the glm_ocr model type
install_auto_processor_patch("glm_ocr", GlmOcrProcessor)
