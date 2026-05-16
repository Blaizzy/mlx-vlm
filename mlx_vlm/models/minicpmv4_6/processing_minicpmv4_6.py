import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from transformers import AddedToken, AutoTokenizer, Qwen2TokenizerFast
from transformers.image_processing_utils import (
    BaseImageProcessor as HFBaseImageProcessor,
)
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_token_config_value(value):
    if isinstance(value, dict):
        content = value.get("content")
        if content is None:
            return None
        return AddedToken(
            content=content,
            lstrip=bool(value.get("lstrip", False)),
            rstrip=bool(value.get("rstrip", False)),
            normalized=bool(value.get("normalized", False)),
            single_word=bool(value.get("single_word", False)),
        )
    if isinstance(value, list):
        out = []
        for item in value:
            normalized = _normalize_token_config_value(item)
            if normalized is not None:
                out.append(normalized)
        return out
    return value


def _extract_hf_from_pretrained_kwargs(kwargs: Dict) -> Dict:
    allowed = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "revision",
        "subfolder",
        "token",
        "use_auth_token",
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def _load_local_qwen2_tokenizer(
    pretrained_model_name_or_path: Union[str, Path],
    *,
    use_fast: bool = True,
    hf_kwargs: Optional[Dict] = None,
):
    hf_kwargs = dict(hf_kwargs or {})
    path = Path(pretrained_model_name_or_path)

    token_kwargs: Dict = {}
    if path.exists() and path.is_dir():
        tokenizer_cfg_path = path / "tokenizer_config.json"
        if tokenizer_cfg_path.exists():
            tokenizer_cfg = _load_json(tokenizer_cfg_path)
            # transformers 4.57 expects extra_special_tokens to be a mapping,
            # while this checkpoint stores it as a legacy list.
            token_kwargs["extra_special_tokens"] = {}
            for key in (
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "additional_special_tokens",
            ):
                if key in tokenizer_cfg:
                    token_kwargs[key] = _normalize_token_config_value(
                        tokenizer_cfg[key]
                    )

            for key in (
                "model_max_length",
                "clean_up_tokenization_spaces",
                "split_special_tokens",
                "add_prefix_space",
            ):
                if key in tokenizer_cfg:
                    token_kwargs[key] = tokenizer_cfg[key]

        special_tokens_map_path = path / "special_tokens_map.json"
        if special_tokens_map_path.exists():
            special_tokens_map = _load_json(special_tokens_map_path)
            for key, value in special_tokens_map.items():
                token_kwargs[key] = _normalize_token_config_value(value)

    return Qwen2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        use_fast=use_fast,
        **hf_kwargs,
        **token_kwargs,
    )


class MiniCPMVImageProcessor(HFBaseImageProcessor):
    """Minimal image processor for MiniCPM-V image inference."""

    model_input_names = ["pixel_values", "image_sizes", "tgt_sizes"]

    def __init__(
        self,
        max_slice_nums: int = 9,
        patch_size: int = 14,
        scale_resolution: int = 448,
        image_feature_size: int = 64,
        im_start: str = "<image>",
        im_end: str = "</image>",
        slice_start: str = "<slice>",
        slice_end: str = "</slice>",
        unk: str = "<unk>",
        im_id_start: str = "<image_id>",
        im_id_end: str = "</image_id>",
        slice_mode: bool = True,
        use_image_id: bool = True,
        downsample_mode: str = "16x",
        pack_image_by_patch: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_slice_nums = int(max_slice_nums)
        self.patch_size = int(patch_size)
        self.scale_resolution = int(scale_resolution)
        self.image_feature_size = int(image_feature_size)
        self.im_start = im_start
        self.im_end = im_end
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.unk = unk
        self.im_id_start = im_id_start
        self.im_id_end = im_id_end
        self.slice_mode = bool(slice_mode)
        self.use_image_id = bool(use_image_id)
        self.downsample_mode = str(downsample_mode)
        # The official MiniCPM-V 4.6 processor uses a patch-packed image layout.
        self.pack_image_by_patch = bool(pack_image_by_patch)

        mean = norm_mean if norm_mean is not None else [0.5, 0.5, 0.5]
        std = norm_std if norm_std is not None else [0.5, 0.5, 0.5]
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    @property
    def image_placeholder(self) -> str:
        return f"{self.im_start}{self.unk * self.image_feature_size}{self.im_end}"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs):
        config_path = Path(pretrained_model_name_or_path) / "preprocessor_config.json"
        cfg = _load_json(config_path) if config_path.exists() else {}
        return cls(**cfg)

    @staticmethod
    def _ensure_divide(length: int, patch_size: int) -> int:
        return max(round(length / patch_size) * patch_size, patch_size)

    @classmethod
    def _find_best_resize(
        cls,
        image_size: Tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> Tuple[int, int]:
        width, height = image_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            ratio = width / max(height, 1)
            height = int(scale_resolution / math.sqrt(max(ratio, 1e-6)))
            width = int(height * ratio)

        merge_factor = patch_size * 4
        best_width = cls._ensure_divide(width, merge_factor)
        best_height = cls._ensure_divide(height, merge_factor)
        return best_width, best_height

    def _get_refine_size(
        self,
        image_size: Tuple[int, int],
        grid: Tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> Tuple[int, int]:
        width, height = image_size
        grid_x, grid_y = grid
        refine_width = self._ensure_divide(width, grid_x)
        refine_height = self._ensure_divide(height, grid_y)
        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y
        best_grid_size = self._find_best_resize(
            (grid_width, grid_height),
            scale_resolution,
            patch_size,
            allow_upscale=allow_upscale,
        )
        return best_grid_size[0] * grid_x, best_grid_size[1] * grid_y

    @staticmethod
    def _split_to_patches(
        image: Image.Image, grid: Tuple[int, int]
    ) -> List[List[Image.Image]]:
        patches: List[List[Image.Image]] = []
        width, height = image.size
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])
        for top in range(0, height, grid_y):
            row = []
            for left in range(0, width, grid_x):
                row.append(image.crop((left, top, left + grid_x, top + grid_y)))
            patches.append(row)
        return patches

    def get_sliced_grid(
        self,
        image_size: Tuple[int, int],
        max_slice_nums: Optional[int] = None,
        never_split: bool = False,
    ) -> Optional[Tuple[int, int]]:
        original_width, original_height = image_size
        ratio = (
            original_width
            * original_height
            / float(self.scale_resolution * self.scale_resolution)
        )
        max_slice_nums = (
            self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        )
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or never_split:
            return None

        candidate_grid_nums = []
        for grid_num in (multiple - 1, multiple, multiple + 1):
            if grid_num == 1 or grid_num > max_slice_nums:
                continue
            candidate_grid_nums.append(grid_num)

        candidate_grids = []
        for grid_num in candidate_grid_nums:
            factor = 1
            while factor <= grid_num:
                if grid_num % factor == 0:
                    candidate_grids.append((factor, grid_num // factor))
                factor += 1

        log_ratio = math.log(original_width / max(original_height, 1))
        best_grid = (1, 1)
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error
        return best_grid

    def slice_image(
        self,
        image: Image.Image,
        max_slice_nums: Optional[int] = None,
        never_split: bool = False,
    ) -> Tuple[Image.Image, List[List[Image.Image]], Optional[Tuple[int, int]]]:
        max_slice_nums = (
            self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        )
        original_size = image.size
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)
        patches: List[List[Image.Image]] = []

        if best_grid is None:
            best_size = self._find_best_resize(
                original_size,
                self.scale_resolution,
                self.patch_size,
                allow_upscale=True,
            )
            source_image = image.resize(best_size, resample=Image.Resampling.BICUBIC)
        else:
            best_resize = self._find_best_resize(
                original_size,
                self.scale_resolution,
                self.patch_size,
                allow_upscale=False,
            )
            source_image = image.resize(best_resize, resample=Image.Resampling.BICUBIC)
            refine_size = self._get_refine_size(
                original_size,
                best_grid,
                self.scale_resolution,
                self.patch_size,
                allow_upscale=True,
            )
            refine_image = image.resize(refine_size, resample=Image.Resampling.BICUBIC)
            patches = self._split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_sliced_images(
        self,
        image: Union[Image.Image, np.ndarray],
        max_slice_nums: Optional[int] = None,
    ) -> List[Image.Image]:
        image = self._to_pil_image(image)
        if not self.slice_mode:
            best_size = self._find_best_resize(
                image.size,
                self.scale_resolution,
                self.patch_size,
                allow_upscale=True,
            )
            return [image.resize(best_size, resample=Image.Resampling.BICUBIC)]

        source_image, patches, _ = self.slice_image(image, max_slice_nums)
        slice_images = [source_image]
        for row in patches:
            slice_images.extend(row)
        return slice_images

    @staticmethod
    def _to_pil_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    def _preprocess_resized(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        image = self._to_pil_image(image)
        image_size = image.size

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # CHW

        tgt_size = np.array(
            [image_size[1] // self.patch_size, image_size[0] // self.patch_size],
            dtype=np.int32,
        )
        if self.pack_image_by_patch:
            arr = self._reshape_by_patch(arr)
        return arr, tgt_size, image_size

    def _reshape_by_patch(self, image: np.ndarray) -> np.ndarray:
        channels, height, width = image.shape
        patch_size = self.patch_size
        valid_height = (height // patch_size) * patch_size
        valid_width = (width // patch_size) * patch_size
        if valid_height <= 0 or valid_width <= 0:
            raise ValueError(
                "MiniCPM-V image is too small for patch packing: "
                f"got {(height, width)} with patch_size={patch_size}."
            )
        if valid_height != height or valid_width != width:
            image = image[:, :valid_height, :valid_width]
            height, width = valid_height, valid_width
        patches = image.reshape(
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        patches = patches.transpose(0, 2, 1, 3, 4)
        return patches.reshape(channels, patch_size, -1).astype(np.float32)

    def _preprocess_single(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        image = self._to_pil_image(image)
        best_size = self._find_best_resize(
            image.size,
            self.scale_resolution,
            self.patch_size,
            allow_upscale=True,
        )
        image = image.resize(best_size, Image.Resampling.BICUBIC)
        return self._preprocess_resized(image)

    def preprocess(
        self,
        images: List[List[Union[Image.Image, np.ndarray]]],
    ) -> Dict[str, List]:
        pixel_values: List[List[np.ndarray]] = []
        tgt_sizes: List[np.ndarray] = []
        image_sizes: List[List[Tuple[int, int]]] = []
        grids: List[List[Tuple[int, int]]] = []
        num_patches_per_image: List[List[int]] = []

        for sample_images in images:
            sample_pixels: List[np.ndarray] = []
            sample_tgts: List[np.ndarray] = []
            sample_sizes: List[Tuple[int, int]] = []
            sample_grids: List[Tuple[int, int]] = []
            sample_patch_counts: List[int] = []

            for image in sample_images:
                pil_image = self._to_pil_image(image)
                sample_sizes.append(pil_image.size)
                if self.slice_mode:
                    source_image, patches, best_grid = self.slice_image(
                        pil_image,
                        self.max_slice_nums,
                    )
                    sliced_images = [source_image]
                    for row in patches:
                        sliced_images.extend(row)
                else:
                    best_size = self._find_best_resize(
                        pil_image.size,
                        self.scale_resolution,
                        self.patch_size,
                        allow_upscale=True,
                    )
                    sliced_images = [
                        pil_image.resize(best_size, resample=Image.Resampling.BICUBIC)
                    ]
                    best_grid = None

                sample_grids.append(best_grid if best_grid is not None else (0, 0))
                sample_patch_counts.append(len(sliced_images))

                for sliced_image in sliced_images:
                    pixels, tgt, _ = self._preprocess_resized(sliced_image)
                    sample_pixels.append(pixels)
                    sample_tgts.append(tgt)

            pixel_values.append(sample_pixels)
            if sample_tgts:
                tgt_sizes.append(np.stack(sample_tgts, axis=0).astype(np.int32))
            else:
                tgt_sizes.append(np.zeros((0, 2), dtype=np.int32))
            image_sizes.append(sample_sizes)
            grids.append(sample_grids)
            num_patches_per_image.append(sample_patch_counts)

        return {
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_sizes": image_sizes,
            "grids": grids,
            "num_patches_per_image": num_patches_per_image,
        }

    def __call__(self, images: List[List[Union[Image.Image, np.ndarray]]], **kwargs):
        return self.preprocess(images)


class MiniCPMVVideoProcessor(MiniCPMVImageProcessor):
    """Minimal video processor for MiniCPM-V video inference."""

    model_input_names = ["pixel_values_videos", "tgt_sizes_videos"]

    def __init__(
        self,
        max_num_frames: int = 128,
        stack_frames: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_num_frames = int(max_num_frames)
        self.stack_frames = int(stack_frames)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs):
        model_dir = Path(pretrained_model_name_or_path)
        cfg = {}

        processor_cfg_path = model_dir / "processor_config.json"
        if processor_cfg_path.exists():
            processor_cfg = _load_json(processor_cfg_path)
            cfg.update(processor_cfg.get("video_processor") or {})
            if not cfg:
                cfg.update(processor_cfg.get("image_processor") or {})

        if not cfg:
            video_cfg_path = model_dir / "video_preprocessor_config.json"
            image_cfg_path = model_dir / "preprocessor_config.json"
            if video_cfg_path.exists():
                cfg.update(_load_json(video_cfg_path))
            elif image_cfg_path.exists():
                cfg.update(_load_json(image_cfg_path))

        cfg.update(kwargs)
        for key in ("processor_class", "image_processor_type", "video_processor_type"):
            cfg.pop(key, None)
        return cls(**cfg)

    @staticmethod
    def _is_frame(value) -> bool:
        if isinstance(value, Image.Image):
            return True
        if isinstance(value, np.ndarray):
            return value.ndim in (2, 3)
        return False

    def _to_video_frames(self, video) -> List[Union[Image.Image, np.ndarray]]:
        if isinstance(video, (list, tuple)):
            if len(video) == 0:
                return []
            if all(self._is_frame(frame) for frame in video):
                return list(video)
            video = np.asarray(video)

        if isinstance(video, Image.Image):
            return [video]

        arr = np.asarray(video)
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3) or arr.shape[-1] in (1, 3):
                return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim in (2, 3):
            return [arr]
        raise ValueError(
            "MiniCPM-V video inputs must be a frame sequence or a numpy array "
            f"with shape (T, C, H, W) or (T, H, W, C), got {arr.shape}."
        )

    @staticmethod
    def _select_frames(
        frames: List[Union[Image.Image, np.ndarray]], max_num_frames: int
    ) -> List[Union[Image.Image, np.ndarray]]:
        if max_num_frames <= 0 or len(frames) <= max_num_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, max_num_frames).round().astype(int)
        return [frames[int(i)] for i in indices]

    def _concat_frames_as_image(
        self, frames: List[Union[Image.Image, np.ndarray]]
    ) -> Image.Image:
        pil_frames = [self._to_pil_image(frame) for frame in frames]
        if len(pil_frames) == 1:
            return pil_frames[0]

        line_width = 6
        width, height = pil_frames[0].size
        count = len(pil_frames)

        if count == 4:
            rows, cols = 2, 2
        elif count == 3:
            rows, cols = (1, 3) if width >= height else (3, 1)
        elif count == 2:
            rows, cols = (1, 2) if width >= height else (2, 1)
        else:
            rows, cols = 1, count

        canvas_width = cols * width + (cols - 1) * line_width
        canvas_height = rows * height + (rows - 1) * line_width
        canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))

        for idx, frame in enumerate(pil_frames):
            row = idx // cols
            col = idx % cols
            canvas.paste(
                frame.resize((width, height)),
                (col * (width + line_width), row * (height + line_width)),
            )

        return canvas

    def _stack_frame_units(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        stack_frames: int,
    ) -> List[Union[Image.Image, np.ndarray]]:
        if stack_frames <= 1:
            return frames
        units: List[Union[Image.Image, np.ndarray]] = []
        for start in range(0, len(frames), stack_frames):
            chunk = frames[start : start + stack_frames]
            units.append(self._concat_frames_as_image(chunk))
        return units

    def preprocess(
        self,
        videos: List[List],
        max_num_frames: Optional[int] = None,
        stack_frames: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, List]:
        del kwargs
        max_num_frames = (
            self.max_num_frames if max_num_frames is None else int(max_num_frames)
        )
        stack_frames = self.stack_frames if stack_frames is None else int(stack_frames)

        pixel_values: List[List[np.ndarray]] = []
        tgt_sizes: List[np.ndarray] = []
        video_sizes: List[List[Tuple[int, int]]] = []
        grids_videos: List[List[Tuple[int, int]]] = []
        num_patches_per_frame: List[List[int]] = []
        num_frames_per_video: List[List[int]] = []

        for sample_videos in videos:
            sample_pixels: List[np.ndarray] = []
            sample_tgts: List[np.ndarray] = []
            sample_sizes: List[Tuple[int, int]] = []
            sample_grids: List[Tuple[int, int]] = []
            sample_patch_counts: List[int] = []
            sample_frame_counts: List[int] = []

            for video in sample_videos:
                frames = self._to_video_frames(video)
                frames = self._select_frames(frames, max_num_frames)
                units = self._stack_frame_units(frames, stack_frames)
                sample_frame_counts.append(len(units))

                for frame in units:
                    pil_frame = self._to_pil_image(frame)
                    sample_sizes.append(pil_frame.size)
                    if self.slice_mode:
                        source_image, patches, best_grid = self.slice_image(
                            pil_frame,
                            self.max_slice_nums,
                        )
                        sliced_images = [source_image]
                        for row in patches:
                            sliced_images.extend(row)
                    else:
                        best_size = self._find_best_resize(
                            pil_frame.size,
                            self.scale_resolution,
                            self.patch_size,
                            allow_upscale=True,
                        )
                        sliced_images = [
                            pil_frame.resize(
                                best_size, resample=Image.Resampling.BICUBIC
                            )
                        ]
                        best_grid = None

                    sample_grids.append(best_grid if best_grid is not None else (0, 0))
                    sample_patch_counts.append(len(sliced_images))

                    for sliced_image in sliced_images:
                        pixels, tgt, _ = self._preprocess_resized(sliced_image)
                        sample_pixels.append(pixels)
                        sample_tgts.append(tgt)

            pixel_values.append(sample_pixels)
            if sample_tgts:
                tgt_sizes.append(np.stack(sample_tgts, axis=0).astype(np.int32))
            else:
                tgt_sizes.append(np.zeros((0, 2), dtype=np.int32))
            video_sizes.append(sample_sizes)
            grids_videos.append(sample_grids)
            num_patches_per_frame.append(sample_patch_counts)
            num_frames_per_video.append(sample_frame_counts)

        return {
            "pixel_values_videos": pixel_values,
            "tgt_sizes_videos": tgt_sizes,
            "target_sizes_videos": tgt_sizes,
            "video_sizes": video_sizes,
            "grids_videos": grids_videos,
            "num_patches_per_frame": num_patches_per_frame,
            "num_frames_per_video": num_frames_per_video,
        }

    def __call__(self, videos: List[List], **kwargs):
        return self.preprocess(videos, **kwargs)


class MiniCPMVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template"]

    _IMAGE_MARKER_PATTERN = re.compile(r"<image>\./</image>|<image>|<\|image_pad\|>")
    _VIDEO_MARKER_PATTERN = re.compile(r"<video>|<\|video_pad\|>")
    _MULTIMODAL_MARKER_PATTERN = re.compile(
        r"<image>\./</image>|<image>|<\|image_pad\|>|<video>|<\|video_pad\|>"
    )

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        video_processor = kwargs.pop("video_processor", None)
        self.image_processor = image_processor or MiniCPMVImageProcessor()
        self.video_processor = video_processor or MiniCPMVVideoProcessor()
        self.tokenizer = tokenizer
        self.image_feature_size = self.image_processor.image_feature_size

        self._ensure_tokenizer_attrs()
        self.image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        self.image_token_id = getattr(
            self.tokenizer,
            "image_token_id",
            (
                self.tokenizer.convert_tokens_to_ids(self.image_token)
                if self.tokenizer is not None
                else None
            ),
        )
        self.video_token = getattr(self.tokenizer, "video_token", "<|video_pad|>")
        self.video_token_id = getattr(
            self.tokenizer,
            "video_token_id",
            (
                self.tokenizer.convert_tokens_to_ids(self.video_token)
                if self.tokenizer is not None
                else None
            ),
        )
        super().__init__(
            self.image_processor,
            self.tokenizer,
            chat_template=chat_template,
        )

    def _ensure_tokenizer_attrs(self):
        if self.tokenizer is None:
            return

        token_map = {
            "im_start": "<image>",
            "im_end": "</image>",
            "slice_start": "<slice>",
            "slice_end": "</slice>",
            "im_id_start": "<image_id>",
            "im_id_end": "</image_id>",
            "image_start_token": "<image>",
            "image_end_token": "</image>",
            "slice_start_token": "<slice>",
            "slice_end_token": "</slice>",
            "image_id_start_token": "<image_id>",
            "image_id_end_token": "</image_id>",
        }

        for attr, token in token_map.items():
            if not hasattr(self.tokenizer, attr):
                setattr(self.tokenizer, attr, token)
            id_attr = f"{attr}_id"
            if not hasattr(self.tokenizer, id_attr):
                setattr(
                    self.tokenizer, id_attr, self.tokenizer.convert_tokens_to_ids(token)
                )

        for attr, token in {
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
        }.items():
            if not hasattr(self.tokenizer, attr):
                setattr(self.tokenizer, attr, token)
            id_attr = f"{attr}_id"
            if not hasattr(self.tokenizer, id_attr):
                token_id = self.tokenizer.convert_tokens_to_ids(
                    getattr(self.tokenizer, attr)
                )
                setattr(self.tokenizer, id_attr, token_id)

        listen_token_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        # If <|listen|> is unknown, convert_tokens_to_ids returns unk_token_id.
        # In that case we must not filter it, otherwise image placeholder <unk>
        # tokens are removed and vision features cannot be injected.
        if (
            listen_token_id is None
            or listen_token_id < 0
            or (unk_token_id is not None and listen_token_id == unk_token_id)
        ):
            self.listen_token_id = None
        else:
            self.listen_token_id = int(listen_token_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _normalize_text(self, text: Union[str, List[str]]) -> List[str]:
        if text is None:
            return [""]
        if isinstance(text, str):
            return [text]
        return list(text)

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @classmethod
    def _count_image_markers(cls, text: str) -> int:
        return len(cls._IMAGE_MARKER_PATTERN.findall(text or ""))

    @classmethod
    def _count_video_markers(cls, text: str) -> int:
        return len(cls._VIDEO_MARKER_PATTERN.findall(text or ""))

    def _normalize_images(
        self,
        images,
        batch_size: int,
        texts: List[str],
    ) -> List[List[Union[Image.Image, np.ndarray]]]:
        if images is None:
            return [[] for _ in range(batch_size)]

        if isinstance(images, (Image.Image, np.ndarray)):
            out = [[] for _ in range(batch_size)]
            out[0] = [images]
            return out

        if not isinstance(images, list) or len(images) == 0:
            return [[] for _ in range(batch_size)]

        if isinstance(images[0], list):
            if len(images) == batch_size:
                return [list(sample) for sample in images]
            if batch_size == 1:
                return [list(images[0])]
            return [list(images[0])] + [[] for _ in range(batch_size - 1)]

        # Flat list of images.
        if batch_size == 1:
            return [list(images)]

        if len(images) == batch_size:
            return [[img] for img in images]

        expected = [self._count_image_markers(text) for text in texts]
        if sum(expected) == len(images):
            out = []
            idx = 0
            for count in expected:
                out.append(images[idx : idx + count])
                idx += count
            return out

        return [list(images)] + [[] for _ in range(batch_size - 1)]

    @staticmethod
    def _looks_like_frame_sequence(value) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            return False
        for item in value:
            if isinstance(item, Image.Image):
                continue
            if isinstance(item, np.ndarray) and item.ndim in (2, 3):
                continue
            return False
        return True

    def _normalize_videos(
        self,
        videos,
        batch_size: int,
        texts: List[str],
    ) -> List[List]:
        if videos is None:
            return [[] for _ in range(batch_size)]

        if isinstance(
            videos, (Image.Image, np.ndarray)
        ) or self._looks_like_frame_sequence(videos):
            out = [[] for _ in range(batch_size)]
            out[0] = [videos]
            return out

        if not isinstance(videos, list) or len(videos) == 0:
            return [[] for _ in range(batch_size)]

        if batch_size == 1:
            return [list(videos)]

        if len(videos) == batch_size:
            return [[video] for video in videos]

        expected = [self._count_video_markers(text) for text in texts]
        if sum(expected) == len(videos):
            out = []
            idx = 0
            for count in expected:
                out.append(videos[idx : idx + count])
                idx += count
            return out

        return [list(videos)] + [[] for _ in range(batch_size - 1)]

    def _inject_image_placeholders(self, text: str, num_images: int) -> str:
        if num_images <= 0:
            return text

        placeholder = self.image_processor.image_placeholder
        used = 0

        def _replace(match):
            nonlocal used
            if used < num_images:
                used += 1
                return placeholder
            return match.group(0)

        output = self._IMAGE_MARKER_PATTERN.sub(_replace, text or "")
        if used < num_images:
            output = (placeholder * (num_images - used)) + output
        return output

    @staticmethod
    def _strip_think_prefix(text: str) -> str:
        if not isinstance(text, str):
            return text
        # The no-thinking chat template emits an empty reasoning block before
        # answer generation. MiniCPM-V can get stuck extending the blank tail, so
        # drop only that exact empty block while preserving real/open think tags.
        text = text.replace("<think>\n\n</think>\n\n", "")
        return text

    def _build_feature_placeholder_ids(
        self,
        start_attr: str,
        start_token: str,
        end_attr: str,
        end_token: str,
        token_count: int,
        fill_attr: str,
        fill_token: str,
    ) -> List[int]:
        def _resolve_token_id(attr_name: str, token_text: str) -> int:
            token_id = getattr(self.tokenizer, attr_name, None)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
            if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                converted = self.tokenizer.convert_tokens_to_ids(token_text)
                if isinstance(converted, int) and converted >= 0:
                    return converted
            return -1

        start_id = _resolve_token_id(start_attr, start_token)
        end_id = _resolve_token_id(end_attr, end_token)
        fill_id = _resolve_token_id(fill_attr, fill_token)
        if start_id < 0 or end_id < 0 or fill_id < 0:
            return []
        return [start_id] + [fill_id] * int(token_count) + [end_id]

    def _build_image_placeholder_ids(
        self,
        token_count: int,
        fill_attr: str = "image_token_id",
        fill_token: Optional[str] = None,
    ) -> List[int]:
        return self._build_feature_placeholder_ids(
            "im_start_id",
            self.image_processor.im_start,
            "im_end_id",
            self.image_processor.im_end,
            token_count,
            fill_attr,
            self.image_token if fill_token is None else fill_token,
        )

    def _build_slice_placeholder_ids(
        self,
        token_count: int,
        fill_attr: str = "image_token_id",
        fill_token: Optional[str] = None,
    ) -> List[int]:
        return self._build_feature_placeholder_ids(
            "slice_start_id",
            self.image_processor.slice_start,
            "slice_end_id",
            self.image_processor.slice_end,
            token_count,
            fill_attr,
            self.image_token if fill_token is None else fill_token,
        )

    def _build_image_id_ids(self, image_idx: int) -> List[int]:
        if not getattr(self.image_processor, "use_image_id", False):
            return []
        text = (
            f"{self.image_processor.im_id_start}{image_idx}"
            f"{self.image_processor.im_id_end}"
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_placeholder_ids_for_image(
        self,
        patch_target_sizes: np.ndarray,
        grid: Tuple[int, int],
        image_idx: int,
        token_divisor: int,
        include_image_id: bool = True,
        slice_mode: Optional[bool] = None,
        fill_attr: str = "image_token_id",
        fill_token: Optional[str] = None,
    ) -> List[int]:
        if patch_target_sizes.size == 0:
            return []

        token_counts = [
            int(np.prod(target_size, dtype=np.int64) // token_divisor)
            for target_size in np.asarray(patch_target_sizes, dtype=np.int32)
        ]
        image_placeholder_ids = self._build_image_placeholder_ids(
            token_counts[0],
            fill_attr=fill_attr,
            fill_token=fill_token,
        )
        if len(image_placeholder_ids) == 0:
            return []

        ids = []
        if include_image_id:
            ids.extend(self._build_image_id_ids(image_idx))
        ids.extend(image_placeholder_ids)

        grid_x, grid_y = grid
        if slice_mode is None:
            slice_mode = getattr(self.image_processor, "slice_mode", False)
        if not slice_mode or grid_x <= 0 or grid_y <= 0:
            return ids

        per_slice_tokens = token_counts[1] if len(token_counts) > 1 else 0
        slice_placeholder_ids = self._build_slice_placeholder_ids(
            per_slice_tokens,
            fill_attr=fill_attr,
            fill_token=fill_token,
        )
        if len(slice_placeholder_ids) == 0:
            return []

        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        grid_x, grid_y = grid
        for row_idx in range(grid_y):
            for _ in range(grid_x):
                ids.extend(slice_placeholder_ids)
            if row_idx != grid_y - 1:
                ids.extend(newline_ids)
        return ids

    def _build_placeholder_ids_for_video(
        self,
        video_target_sizes: np.ndarray,
        frame_grids: Sequence[Tuple[int, int]],
        num_patches_per_frame: Sequence[int],
        video_idx: int,
        token_divisor: int,
    ) -> List[int]:
        if np.asarray(video_target_sizes).size == 0:
            return []

        ids = []
        ids.extend(self._build_image_id_ids(video_idx))

        start = 0
        for frame_idx, count in enumerate(num_patches_per_frame):
            end = start + int(count)
            frame_target_sizes = np.asarray(
                video_target_sizes[start:end], dtype=np.int32
            )
            start = end
            if frame_target_sizes.size == 0:
                continue

            grid = frame_grids[frame_idx] if frame_idx < len(frame_grids) else (0, 0)
            frame_ids = self._build_placeholder_ids_for_image(
                frame_target_sizes,
                grid,
                frame_idx,
                token_divisor,
                include_image_id=False,
                slice_mode=getattr(self.video_processor, "slice_mode", False),
                fill_attr="video_token_id",
                fill_token=self.video_token,
            )
            if len(frame_ids) == 0:
                return []
            ids.extend(frame_ids)

        return ids

    def _encode_with_image_placeholders(
        self,
        text: str,
        patch_target_sizes: np.ndarray,
        image_grids: Sequence[Tuple[int, int]],
        num_patches_per_image: Sequence[int],
        token_divisor: int,
        add_special_tokens: bool,
    ) -> List[int]:
        if text is None:
            text = ""
        num_images = len(num_patches_per_image)
        if num_images <= 0:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        if np.asarray(patch_target_sizes).size == 0:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        preview_tokens = int(
            np.prod(np.asarray(patch_target_sizes[0], dtype=np.int32), dtype=np.int64)
            // token_divisor
        )
        placeholder_ids = self._build_image_placeholder_ids(preview_tokens)
        if len(placeholder_ids) == 0:
            # Fallback to text substitution when tokenizer lacks required special token ids.
            fallback = self._inject_image_placeholders(text, num_images)
            return self.tokenizer.encode(
                fallback, add_special_tokens=add_special_tokens
            )

        per_image_target_sizes: List[np.ndarray] = []
        start = 0
        for count in num_patches_per_image:
            end = start + int(count)
            per_image_target_sizes.append(
                np.asarray(patch_target_sizes[start:end], dtype=np.int32)
            )
            start = end

        image_placeholder_blocks = [
            self._build_placeholder_ids_for_image(
                per_image_target_sizes[image_idx],
                image_grids[image_idx],
                image_idx,
                token_divisor,
            )
            for image_idx in range(num_images)
        ]
        if any(len(block) == 0 for block in image_placeholder_blocks):
            fallback = self._inject_image_placeholders(text, num_images)
            return self.tokenizer.encode(
                fallback, add_special_tokens=add_special_tokens
            )

        ids: List[int] = []
        cursor = 0
        used = 0
        for match in self._IMAGE_MARKER_PATTERN.finditer(text):
            start, end = match.span()
            if start > cursor:
                ids.extend(
                    self.tokenizer.encode(text[cursor:start], add_special_tokens=False)
                )
            if used < num_images:
                ids.extend(image_placeholder_blocks[used])
                used += 1
            else:
                ids.extend(
                    self.tokenizer.encode(text[start:end], add_special_tokens=False)
                )
            cursor = end

        if cursor < len(text):
            ids.extend(self.tokenizer.encode(text[cursor:], add_special_tokens=False))

        if used < num_images:
            leading = []
            for block in image_placeholder_blocks[used:]:
                leading.extend(block)
            ids = leading + ids

        if add_special_tokens:
            if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            else:
                # Fallback for very minimal slow tokenizers.
                # Do NOT append EOS here; for chat-formatted prompts this can cause
                # immediate empty termination at generation start.
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None and (len(ids) == 0 or ids[0] != int(bos_id)):
                    ids = [int(bos_id)] + ids
        return ids

    def _encode_with_multimodal_placeholders(
        self,
        text: str,
        image_patch_target_sizes: np.ndarray,
        image_grids: Sequence[Tuple[int, int]],
        num_patches_per_image: Sequence[int],
        video_patch_target_sizes: np.ndarray,
        video_grids: Sequence[Tuple[int, int]],
        num_patches_per_frame: Sequence[int],
        num_frames_per_video: Sequence[int],
        image_token_divisor: int,
        video_token_divisor: int,
        add_special_tokens: bool,
    ) -> Tuple[List[int], List[int]]:
        if text is None:
            text = ""

        image_patch_target_sizes = np.asarray(image_patch_target_sizes, dtype=np.int32)
        video_patch_target_sizes = np.asarray(video_patch_target_sizes, dtype=np.int32)

        image_blocks = []
        image_start = 0
        for image_idx, count in enumerate(num_patches_per_image):
            image_end = image_start + int(count)
            block = self._build_placeholder_ids_for_image(
                image_patch_target_sizes[image_start:image_end],
                image_grids[image_idx] if image_idx < len(image_grids) else (0, 0),
                image_idx,
                image_token_divisor,
            )
            if len(block) == 0:
                raise ValueError(
                    "MiniCPM-V tokenizer is missing image placeholder token ids."
                )
            image_blocks.append((block, list(range(image_start, image_end))))
            image_start = image_end

        image_patch_total = int(image_start)
        video_blocks = []
        frame_start = 0
        video_patch_start = 0
        for video_idx, frame_count in enumerate(num_frames_per_video):
            frame_end = frame_start + int(frame_count)
            frame_patch_counts = list(num_patches_per_frame[frame_start:frame_end])
            patch_count = int(sum(frame_patch_counts))
            video_patch_end = video_patch_start + patch_count
            block = self._build_placeholder_ids_for_video(
                video_patch_target_sizes[video_patch_start:video_patch_end],
                video_grids[frame_start:frame_end],
                frame_patch_counts,
                video_idx,
                video_token_divisor,
            )
            if len(block) == 0:
                raise ValueError(
                    "MiniCPM-V tokenizer is missing video placeholder token ids."
                )
            feature_indices = list(
                range(
                    image_patch_total + video_patch_start,
                    image_patch_total + video_patch_end,
                )
            )
            video_blocks.append((block, feature_indices))
            frame_start = frame_end
            video_patch_start = video_patch_end

        if not image_blocks and not video_blocks:
            return (
                self.tokenizer.encode(text, add_special_tokens=add_special_tokens),
                [],
            )

        ids: List[int] = []
        feature_order: List[int] = []
        cursor = 0
        image_used = 0
        video_used = 0

        for match in self._MULTIMODAL_MARKER_PATTERN.finditer(text):
            start, end = match.span()
            marker = match.group(0)
            if start > cursor:
                ids.extend(
                    self.tokenizer.encode(text[cursor:start], add_special_tokens=False)
                )

            if marker in (
                "<video>",
                "<|video_pad|>",
                getattr(self, "video_token", ""),
            ):
                if video_used < len(video_blocks):
                    block, indices = video_blocks[video_used]
                    ids.extend(block)
                    feature_order.extend(indices)
                    video_used += 1
                else:
                    ids.extend(
                        self.tokenizer.encode(text[start:end], add_special_tokens=False)
                    )
            else:
                if image_used < len(image_blocks):
                    block, indices = image_blocks[image_used]
                    ids.extend(block)
                    feature_order.extend(indices)
                    image_used += 1
                else:
                    ids.extend(
                        self.tokenizer.encode(text[start:end], add_special_tokens=False)
                    )
            cursor = end

        if cursor < len(text):
            ids.extend(self.tokenizer.encode(text[cursor:], add_special_tokens=False))

        leading_ids: List[int] = []
        leading_order: List[int] = []
        for block, indices in image_blocks[image_used:]:
            leading_ids.extend(block)
            leading_order.extend(indices)
        for block, indices in video_blocks[video_used:]:
            leading_ids.extend(block)
            leading_order.extend(indices)
        if leading_ids:
            ids = leading_ids + ids
            feature_order = leading_order + feature_order

        if add_special_tokens:
            if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            else:
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None and (len(ids) == 0 or ids[0] != int(bos_id)):
                    ids = [int(bos_id)] + ids
        return ids, feature_order

    def _compute_image_bounds(self, input_ids: np.ndarray) -> np.ndarray:
        start_ids = np.array(
            [self.tokenizer.im_start_id, self.tokenizer.slice_start_id], dtype=np.int32
        )
        end_ids = np.array(
            [self.tokenizer.im_end_id, self.tokenizer.slice_end_id], dtype=np.int32
        )

        start_idx = np.where(np.isin(input_ids, start_ids))[0] + 1
        end_idx = np.where(np.isin(input_ids, end_ids))[0]
        n = min(len(start_idx), len(end_idx))
        if n == 0:
            return np.zeros((0, 2), dtype=np.int32)

        return np.stack([start_idx[:n], end_idx[:n]], axis=1).astype(np.int32)

    @staticmethod
    def _left_or_right_pad(
        input_ids: List[np.ndarray],
        pad_token_id: int,
        padding_side: str = "left",
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        batch_size = len(input_ids)
        max_len = max((len(x) for x in input_ids), default=0)

        padded = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
        offsets: List[int] = []

        for i, ids in enumerate(input_ids):
            length = len(ids)
            if padding_side == "left":
                offset = max_len - length
                padded[i, offset:] = ids
                attention_mask[i, offset:] = 1
            else:
                offset = 0
                padded[i, :length] = ids
                attention_mask[i, :length] = 1
            offsets.append(offset)

        return padded, attention_mask, offsets

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images=None,
        videos=None,
        audios=None,
        audio=None,
        add_special_tokens: bool = False,
        padding: bool = True,
        padding_side: str = "left",
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        slice_mode: Optional[bool] = None,
        max_slice_nums: Optional[int] = None,
        use_image_id: Optional[bool] = None,
        pack_image_by_patch: Optional[bool] = None,
        max_num_frames: Optional[int] = None,
        stack_frames: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        del return_tensors  # This processor stays numpy/list-native for MLX.
        del kwargs
        if audios is not None or audio is not None:
            raise ValueError("MiniCPM-V processor does not support audio inputs.")

        overrides = {}
        if slice_mode is not None:
            overrides["slice_mode"] = self._coerce_bool(slice_mode)
        if max_slice_nums is not None:
            overrides["max_slice_nums"] = int(max_slice_nums)
        if use_image_id is not None:
            overrides["use_image_id"] = self._coerce_bool(use_image_id)
        if pack_image_by_patch is not None:
            overrides["pack_image_by_patch"] = self._coerce_bool(pack_image_by_patch)

        video_overrides = dict(overrides)
        if max_num_frames is not None:
            video_overrides["max_num_frames"] = int(max_num_frames)
        if stack_frames is not None:
            video_overrides["stack_frames"] = int(stack_frames)

        old_values = {
            name: getattr(self.image_processor, name)
            for name in overrides
            if hasattr(self.image_processor, name)
        }
        old_video_values = {
            name: getattr(self.video_processor, name)
            for name in video_overrides
            if hasattr(self.video_processor, name)
        }
        for name, value in overrides.items():
            setattr(self.image_processor, name, value)
        for name, value in video_overrides.items():
            setattr(self.video_processor, name, value)

        # MiniCPM chat templates already include control tokens.
        # Adding tokenizer-level special tokens here can destabilize decoding.
        encode_add_special_tokens = False

        try:
            texts = self._normalize_text(text)
            batch_size = len(texts)
            batched_images = self._normalize_images(images, batch_size, texts)
            batched_videos = self._normalize_videos(videos, batch_size, texts)
            image_inputs = self.image_processor.preprocess(batched_images)
            video_inputs = self.video_processor.preprocess(batched_videos)
            image_token_divisor = (
                4
                if getattr(self.image_processor, "downsample_mode", "16x") == "4x"
                else 16
            )
            video_token_divisor = (
                4
                if getattr(self.video_processor, "downsample_mode", "16x") == "4x"
                else 16
            )

            input_ids_list: List[np.ndarray] = []
            image_bounds_list: List[np.ndarray] = []
            pixel_values_list: List[List[np.ndarray]] = []
            tgt_sizes_list: List[np.ndarray] = []

            for i, prompt in enumerate(texts):
                prompt = self._strip_think_prefix(prompt)
                ids, feature_order = self._encode_with_multimodal_placeholders(
                    prompt,
                    image_patch_target_sizes=image_inputs["tgt_sizes"][i],
                    image_grids=image_inputs["grids"][i],
                    num_patches_per_image=image_inputs["num_patches_per_image"][i],
                    video_patch_target_sizes=video_inputs["tgt_sizes_videos"][i],
                    video_grids=video_inputs["grids_videos"][i],
                    num_patches_per_frame=video_inputs["num_patches_per_frame"][i],
                    num_frames_per_video=video_inputs["num_frames_per_video"][i],
                    image_token_divisor=image_token_divisor,
                    video_token_divisor=video_token_divisor,
                    add_special_tokens=encode_add_special_tokens,
                )
                if self.listen_token_id is not None and self.listen_token_id >= 0:
                    ids = [
                        token_id for token_id in ids if token_id != self.listen_token_id
                    ]
                if max_length is not None:
                    ids = ids[:max_length]

                ids_array = np.array(ids, dtype=np.int32)
                input_ids_list.append(ids_array)

                sample_pixels = list(image_inputs["pixel_values"][i]) + list(
                    video_inputs["pixel_values_videos"][i]
                )
                image_tgt = np.asarray(image_inputs["tgt_sizes"][i], dtype=np.int32)
                video_tgt = np.asarray(
                    video_inputs["tgt_sizes_videos"][i], dtype=np.int32
                )
                if image_tgt.size > 0 and video_tgt.size > 0:
                    sample_tgt = np.concatenate([image_tgt, video_tgt], axis=0)
                elif image_tgt.size > 0:
                    sample_tgt = image_tgt
                elif video_tgt.size > 0:
                    sample_tgt = video_tgt
                else:
                    sample_tgt = np.zeros((0, 2), dtype=np.int32)

                if feature_order:
                    sample_pixels = [sample_pixels[idx] for idx in feature_order]
                    sample_tgt = sample_tgt[np.asarray(feature_order, dtype=np.int32)]
                pixel_values_list.append(sample_pixels)
                tgt_sizes_list.append(sample_tgt.astype(np.int32))

                image_bounds = self._compute_image_bounds(ids_array)
                expected_features = len(sample_pixels)
                if expected_features != int(image_bounds.shape[0]):
                    raise ValueError(
                        "MiniCPM-V image placeholder count does not match processed "
                        f"vision inputs for sample {i}: placeholders={image_bounds.shape[0]} "
                        f"pixel_values={expected_features}."
                    )
                image_bounds_list.append(image_bounds)

            if not padding and len(input_ids_list) == 1:
                padded_input_ids = np.expand_dims(input_ids_list[0], axis=0)
                attention_mask = np.ones_like(padded_input_ids, dtype=np.int32)
                offsets = [0]
            else:
                padded_input_ids, attention_mask, offsets = self._left_or_right_pad(
                    input_ids_list,
                    pad_token_id=self.tokenizer.pad_token_id,
                    padding_side=padding_side,
                )

            for i, offset in enumerate(offsets):
                if offset > 0 and image_bounds_list[i].size > 0:
                    image_bounds_list[i] = image_bounds_list[i] + offset

            return {
                "input_ids": padded_input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values_list,
                "image_sizes": image_inputs["image_sizes"],
                "video_sizes": video_inputs["video_sizes"],
                "tgt_sizes": tgt_sizes_list,
                "grids": image_inputs["grids"],
                "grids_videos": video_inputs["grids_videos"],
                "num_patches_per_image": image_inputs["num_patches_per_image"],
                "num_patches_per_frame": video_inputs["num_patches_per_frame"],
                "num_frames_per_video": video_inputs["num_frames_per_video"],
                "image_bound": image_bounds_list,
            }
        finally:
            for name, value in old_values.items():
                setattr(self.image_processor, name, value)
            for name, value in old_video_values.items():
                setattr(self.video_processor, name, value)

    def apply_chat_template(self, *args, **kwargs):
        add_generation_prompt = bool(kwargs.get("add_generation_prompt", False))
        enable_thinking = kwargs.get("enable_thinking", None)
        tokenize = bool(kwargs.get("tokenize", False))
        if add_generation_prompt and enable_thinking is False:
            render_kwargs = dict(kwargs)
            render_kwargs["add_generation_prompt"] = False
            render_kwargs["tokenize"] = False
            rendered = self.tokenizer.apply_chat_template(*args, **render_kwargs)
            rendered = f"{rendered}<|im_start|>assistant\n"
            if tokenize:
                return self.tokenizer.encode(rendered, add_special_tokens=False)
            return rendered
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        image_input_names = getattr(self.image_processor, "model_input_names", [])
        video_input_names = getattr(self.video_processor, "model_input_names", [])
        return list(
            dict.fromkeys(
                list(tokenizer_input_names)
                + list(image_input_names)
                + list(video_input_names)
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        chat_template = kwargs.pop("chat_template", None)
        trust_remote_code = bool(kwargs.pop("trust_remote_code", False))
        use_fast = bool(kwargs.pop("use_fast", True))
        hf_kwargs = _extract_hf_from_pretrained_kwargs(kwargs)

        tokenizer = None
        local_tokenizer_error = None
        try:
            tokenizer = _load_local_qwen2_tokenizer(
                pretrained_model_name_or_path,
                use_fast=use_fast,
                hf_kwargs=hf_kwargs,
            )
        except Exception as exc:
            local_tokenizer_error = exc

        if tokenizer is None:
            if not trust_remote_code:
                raise ValueError(
                    "Failed to load MiniCPM-V tokenizer without remote code. "
                    "Set --trust-remote-code only if you explicitly want HF custom code."
                ) from local_tokenizer_error

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                use_fast=use_fast,
                **hf_kwargs,
            )

        load_chat_template(tokenizer, pretrained_model_name_or_path)

        image_processor = MiniCPMVImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **hf_kwargs
        )
        video_processor = MiniCPMVVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, **hf_kwargs
        )
        return cls(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


# Backward-compatible aliases for any existing internal imports.
MiniCPMOImageProcessor = MiniCPMVImageProcessor
MiniCPMOVideoProcessor = MiniCPMVVideoProcessor
MiniCPMOProcessor = MiniCPMVProcessor

install_auto_processor_patch(["minicpmv4_6"], MiniCPMVProcessor)
