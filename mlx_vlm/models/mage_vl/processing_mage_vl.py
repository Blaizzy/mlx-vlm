"""Torch-free processor for Mage-VL.

The checkpoint's own remote-code processor (`processing_mage_vl.MageVLProcessor` on the hub) is
torch-native — its ``__call__`` builds torch tensors, so mlx-vlm's ``prepare_inputs`` dies with
``ones_like(): argument 'input' must be Tensor, not array``. This in-repo processor follows the
``qwen3_vl`` pattern: numpy/mlx end to end, installed over AutoProcessor for ``model_type
mage_vl`` so ``mlx_vlm.load`` returns it without ``trust_remote_code``.

Video is supported through uniform frame sampling, reusing ``utils.load_video`` (OpenCV,
already a dependency) and ``Qwen3VLVideoProcessor`` (the existing numpy/PIL port). That
covers the frame-sampled path requested in #1766 without adding any dependency.

Upstream's *codec-native* path is a separate matter: it needs ``codec-video-prep``, a
native extension shipping patched FFmpeg with manylinux-only wheels, so it cannot run on
macOS at all. The tower itself is already sparsity-agnostic — ``patch_positions`` accepts
arbitrary ``(L, 3)`` coordinates and ``VisionModel.__call__`` now takes explicit
``cu_seqlens`` — so a codec-selected patch set can be supplied by a caller without further
changes here.

``patch_positions`` is deliberately NOT emitted: the model derives positions from
``image_grid_thw`` in the processor's 2x2 block order, and that derivation is pinned
row-identical to the reference processor's output by ``test_mage_vl_positions.py``.
"""

from typing import List, Optional, Union

import mlx.core as mx
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"


class MageVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None,
        video_processor=None, **kwargs
    ):
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.spatial_merge_size = getattr(image_processor, "merge_size", 2)
        self.image_token = IMAGE_PAD
        self.video_token = VIDEO_PAD
        self.video_processor = video_processor

    # The subset of preprocessor_config.json Qwen3VLImageProcessor consumes.
    _IMAGE_PROCESSOR_KEYS = (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "min_pixels",
        "max_pixels",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "do_convert_rgb",
    )

    @staticmethod
    def _load_preprocessor_config(pretrained_model_name_or_path) -> dict:
        import json
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            cfg = path / "preprocessor_config.json"
            return json.loads(cfg.read_text()) if cfg.exists() else {}
        try:
            from huggingface_hub import hf_hub_download

            return json.loads(
                Path(
                    hf_hub_download(
                        str(pretrained_model_name_or_path), "preprocessor_config.json"
                    )
                ).read_text()
            )
        except Exception:
            return {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        from ..qwen3_vl.processing_qwen3_vl import (
            Qwen3VLImageProcessor,
            Qwen3VLVideoProcessor,
        )

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        # trust_remote_code=False EXPLICITLY: the checkpoint's configs carry auto_map entries,
        # and leaving the argument unset makes transformers prompt interactively.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Construct the image processor DIRECTLY — no AutoImageProcessor. On a torch-free
        # install (the standard MLX setup) AutoImageProcessor resolves to a torchvision-backed
        # class and raises ImportError, which the AutoProcessor patch dispatcher swallows,
        # silently falling back to the checkpoint's torch remote-code processor — the exact
        # failure this class exists to prevent. Reported by @ismaelvega on PR #1745 and
        # mlx-community/Mage-VL-8bit discussion #1. Qwen3VLImageProcessor is the in-repo
        # numpy/PIL port with the same schema (Mage-VL's config: patch 16, merge 2,
        # temporal_patch_size 1, CLIP statistics).
        config = cls._load_preprocessor_config(pretrained_model_name_or_path)
        shared = {k: config[k] for k in cls._IMAGE_PROCESSOR_KEYS if k in config}
        image_processor = Qwen3VLImageProcessor(**shared)
        # Same schema, same statistics -- Mage-VL treats a still image as a degenerate
        # one-frame video, so the two processors must agree on patch/merge/normalisation.
        video_processor = Qwen3VLVideoProcessor(**shared)
        return cls(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
        )

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput]] = None,
        images=None,
        videos=None,
        audio=None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(text, str):
            text = [text]
        text = list(text or [])

        image_inputs = {}
        if images is not None:
            image_inputs = dict(
                self.image_processor(images=images, return_tensors="np")
            )
            grids = mx.array(image_inputs["image_grid_thw"]).tolist()

            # Expand each <|image_pad|> placeholder to its grid's merged-token count
            # (t*h*w / merge^2) — mirrors the reference processor's `_expand_image_pads`.
            merge = self.spatial_merge_size
            counts = [(t * h * w) // (merge * merge) for t, h, w in grids]
            text = [self._expand(s, IMAGE_PAD, counts) for s in text]

        video_inputs = {}
        if videos is not None:
            if self.video_processor is None:
                raise ValueError("this processor was built without a video_processor")
            video_inputs = dict(self.video_processor(self._as_arrays(videos)))
            grids = mx.array(video_inputs["video_grid_thw"]).tolist()
            merge = self.spatial_merge_size
            counts = [(t * h * w) // (merge * merge) for t, h, w in grids]
            text = [self._expand(s, VIDEO_PAD, counts) for s in text]

        # mlx-vlm's process_inputs passes padding in kwargs; hardcoding it too collides.
        padding = kwargs.pop("padding", True)
        text_inputs = self.tokenizer(
            text, return_tensors="np", padding=padding, **kwargs
        )
        return BatchFeature(
            data=to_mlx({**dict(text_inputs), **image_inputs, **video_inputs})
        )

    @staticmethod
    def _expand(text: str, token: str, counts: list) -> str:
        """Expand each placeholder to its grid's merged-token count.

        Mirrors the reference processor's ``_expand_image_pads``: the language side finds
        visual features by matching token ids, so the placeholder count has to equal the
        number of merged tokens the tower will emit.
        """
        idx = 0
        while token in text and idx < len(counts):
            text = text.replace(token, "<|placeholder|>" * counts[idx], 1)
            idx += 1
        return text.replace("<|placeholder|>", token)

    @staticmethod
    def _as_arrays(videos):
        """Accept file paths or already-decoded (T, C, H, W) arrays."""
        import numpy as np

        from ...utils import load_video

        if not isinstance(videos, list):
            videos = [videos]
        out = []
        for v in videos:
            out.append(load_video(v)[0] if isinstance(v, str) else np.asarray(v))
        return out

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids", "attention_mask",
            "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw",
        ]


__all__ = ["MageVLProcessor"]

install_auto_processor_patch("mage_vl", MageVLProcessor)
