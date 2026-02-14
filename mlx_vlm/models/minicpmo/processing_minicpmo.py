import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.image_processing_utils import (
    BaseImageProcessor as HFBaseImageProcessor,
)
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MiniCPMOImageProcessor(HFBaseImageProcessor):
    """Minimal image processor for MiniCPM-o image inference without torch."""

    model_input_names = ["pixel_values", "image_sizes", "tgt_sizes"]

    def __init__(
        self,
        patch_size: int = 14,
        scale_resolution: int = 448,
        image_feature_size: int = 64,
        im_start: str = "<image>",
        im_end: str = "</image>",
        unk: str = "<unk>",
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.scale_resolution = int(scale_resolution)
        self.image_feature_size = int(image_feature_size)
        self.im_start = im_start
        self.im_end = im_end
        self.unk = unk

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
        allow_upscale: bool = True,
    ) -> Tuple[int, int]:
        width, height = image_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            ratio = width / max(height, 1)
            height = int(scale_resolution / math.sqrt(max(ratio, 1e-6)))
            width = int(height * ratio)

        best_width = cls._ensure_divide(width, patch_size)
        best_height = cls._ensure_divide(height, patch_size)
        return best_width, best_height

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

    def _preprocess_single(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        image = self._to_pil_image(image)
        original_size = image.size
        resized_w, resized_h = self._find_best_resize(
            original_size,
            self.scale_resolution,
            self.patch_size,
            allow_upscale=True,
        )
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # CHW

        tgt_size = np.array(
            [resized_h // self.patch_size, resized_w // self.patch_size],
            dtype=np.int32,
        )
        return arr, tgt_size, original_size

    def preprocess(
        self,
        images: List[List[Union[Image.Image, np.ndarray]]],
    ) -> Dict[str, List]:
        pixel_values: List[List[np.ndarray]] = []
        tgt_sizes: List[np.ndarray] = []
        image_sizes: List[List[Tuple[int, int]]] = []

        for sample_images in images:
            sample_pixels: List[np.ndarray] = []
            sample_tgts: List[np.ndarray] = []
            sample_sizes: List[Tuple[int, int]] = []

            for image in sample_images:
                pixels, tgt, size = self._preprocess_single(image)
                sample_pixels.append(pixels)
                sample_tgts.append(tgt)
                sample_sizes.append(size)

            pixel_values.append(sample_pixels)
            if sample_tgts:
                tgt_sizes.append(np.stack(sample_tgts, axis=0).astype(np.int32))
            else:
                tgt_sizes.append(np.zeros((0, 2), dtype=np.int32))
            image_sizes.append(sample_sizes)

        return {
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_sizes": image_sizes,
        }

    def __call__(self, images: List[List[Union[Image.Image, np.ndarray]]], **kwargs):
        return self.preprocess(images)


class MiniCPMOProcessor(ProcessorMixin):
    attributes = ["image_processor", "audio_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template"]

    _IMAGE_MARKER_PATTERN = re.compile(r"<image>\./</image>|<image>")
    _AUDIO_MARKER_PATTERN = re.compile(r"<audio>\./</audio>|<audio>")

    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_processor = image_processor or MiniCPMOImageProcessor()
        self.audio_processor = audio_processor
        self.feature_extractor = audio_processor  # prepare_inputs compatibility
        self.tokenizer = tokenizer
        self.image_feature_size = self.image_processor.image_feature_size
        self.audio_pool_step = int(kwargs.get("audio_pool_step", 5))

        self._ensure_tokenizer_attrs()
        super().__init__(
            self.image_processor,
            self.audio_processor,
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
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "spk_start": "<|spk_bos|>",
            "spk_end": "<|spk_eos|>",
        }

        for attr, token in token_map.items():
            if not hasattr(self.tokenizer, attr):
                setattr(self.tokenizer, attr, token)
            id_attr = f"{attr}_id"
            if not hasattr(self.tokenizer, id_attr):
                setattr(
                    self.tokenizer, id_attr, self.tokenizer.convert_tokens_to_ids(token)
                )

        self.listen_token_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _normalize_text(self, text: Union[str, List[str]]) -> List[str]:
        if text is None:
            return [""]
        if isinstance(text, str):
            return [text]
        return list(text)

    @classmethod
    def _count_image_markers(cls, text: str) -> int:
        return len(cls._IMAGE_MARKER_PATTERN.findall(text or ""))

    @classmethod
    def _count_audio_markers(cls, text: str) -> int:
        return len(cls._AUDIO_MARKER_PATTERN.findall(text or ""))

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

    def _normalize_audios(
        self,
        audios,
        batch_size: int,
        texts: List[str],
    ) -> List[List[np.ndarray]]:
        if audios is None:
            return [[] for _ in range(batch_size)]

        if isinstance(audios, np.ndarray):
            out = [[] for _ in range(batch_size)]
            out[0] = [audios]
            return out

        if not isinstance(audios, list) or len(audios) == 0:
            return [[] for _ in range(batch_size)]

        if isinstance(audios[0], list):
            if len(audios) == batch_size:
                return [list(sample) for sample in audios]
            if batch_size == 1:
                return [list(audios[0])]
            return [list(audios[0])] + [[] for _ in range(batch_size - 1)]

        if batch_size == 1:
            return [list(audios)]

        if len(audios) == batch_size:
            return [[a] for a in audios]

        expected = [self._count_audio_markers(text) for text in texts]
        if sum(expected) == len(audios):
            out = []
            idx = 0
            for count in expected:
                out.append(audios[idx : idx + count])
                idx += count
            return out

        return [list(audios)] + [[] for _ in range(batch_size - 1)]

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

    def _get_audio_placeholder(self, n_samples: int) -> Tuple[str, int]:
        if self.audio_processor is None:
            return "", 0

        hop_length = int(getattr(self.audio_processor, "hop_length", 160))
        feature_lens = int(math.ceil(max(n_samples, 1) / hop_length))
        feature_lens = int((feature_lens - 1) // 2 + 1)
        output_lens = int(
            (feature_lens - self.audio_pool_step) // self.audio_pool_step + 1
        )
        output_lens = max(1, output_lens)

        placeholder = (
            self.tokenizer.audio_start
            + ("<unk>" * output_lens)
            + self.tokenizer.audio_end
        )
        return placeholder, feature_lens

    def _inject_audio_placeholders(
        self,
        text: str,
        audio_placeholders: List[str],
    ) -> str:
        if len(audio_placeholders) == 0:
            return text

        used = 0

        def _replace(match):
            nonlocal used
            if used < len(audio_placeholders):
                token = audio_placeholders[used]
                used += 1
                return token
            return match.group(0)

        output = self._AUDIO_MARKER_PATTERN.sub(_replace, text or "")
        return output

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

    def _compute_audio_bounds(self, input_ids: np.ndarray) -> np.ndarray:
        start_idx = np.where(input_ids == self.tokenizer.audio_start_id)[0] + 1
        end_idx = np.where(input_ids == self.tokenizer.audio_end_id)[0]
        n = min(len(start_idx), len(end_idx))
        if n == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.stack([start_idx[:n], end_idx[:n]], axis=1).astype(np.int32)

    def _extract_audio_inputs(
        self,
        batched_audios: List[List[np.ndarray]],
    ) -> Tuple[np.ndarray, List[List[int]], List[List[str]]]:
        if self.audio_processor is None:
            return (
                np.zeros((0, 80, 0), dtype=np.float32),
                [[] for _ in batched_audios],
                [[] for _ in batched_audios],
            )

        all_waveforms = []
        sample_counts = []
        for sample_audios in batched_audios:
            sample_counts.append(len(sample_audios))
            for waveform in sample_audios:
                arr = np.asarray(waveform, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                all_waveforms.append(arr)

        if len(all_waveforms) == 0:
            return (
                np.zeros((0, 80, 0), dtype=np.float32),
                [[] for _ in batched_audios],
                [[] for _ in batched_audios],
            )

        audio_inputs = self.audio_processor(
            all_waveforms,
            sampling_rate=self.audio_processor.sampling_rate,
            return_attention_mask=True,
            padding=True,
            return_tensors="np",
        )
        features = audio_inputs["input_features"].astype(np.float32)
        feature_lens = np.sum(audio_inputs["attention_mask"], axis=-1, dtype=np.int32)

        nested_lens: List[List[int]] = []
        nested_placeholders: List[List[str]] = []
        idx = 0
        for count in sample_counts:
            cur_lens: List[int] = []
            cur_ph: List[str] = []
            for _ in range(count):
                placeholder, cur_feature_len = self._get_audio_placeholder(
                    int(len(all_waveforms[idx]))
                )
                cur_ph.append(placeholder)
                # Use feature extractor mask length for model-side length computation.
                cur_lens.append(int(feature_lens[idx]))
                # Keep cur_feature_len computed from raw samples for placeholder stability.
                _ = cur_feature_len
                idx += 1
            nested_lens.append(cur_lens)
            nested_placeholders.append(cur_ph)

        return features, nested_lens, nested_placeholders

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
        audios=None,
        audio=None,
        add_special_tokens: bool = False,
        padding: bool = True,
        padding_side: str = "left",
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        del return_tensors  # This processor stays numpy/list-native for MLX.
        del kwargs

        raw_audios = audios if audios is not None else audio

        texts = self._normalize_text(text)
        batch_size = len(texts)
        batched_images = self._normalize_images(images, batch_size, texts)
        batched_audios = self._normalize_audios(raw_audios, batch_size, texts)
        image_inputs = self.image_processor.preprocess(batched_images)
        audio_features, audio_feature_lens, audio_placeholders = (
            self._extract_audio_inputs(batched_audios)
        )

        input_ids_list: List[np.ndarray] = []
        image_bounds_list: List[np.ndarray] = []
        audio_bounds_list: List[np.ndarray] = []

        for i, prompt in enumerate(texts):
            prompt = self._inject_image_placeholders(prompt, len(batched_images[i]))
            prompt = self._inject_audio_placeholders(prompt, audio_placeholders[i])
            ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            if self.listen_token_id is not None and self.listen_token_id >= 0:
                ids = [token_id for token_id in ids if token_id != self.listen_token_id]
            if max_length is not None:
                ids = ids[:max_length]

            ids_array = np.array(ids, dtype=np.int32)
            input_ids_list.append(ids_array)
            image_bounds_list.append(self._compute_image_bounds(ids_array))
            audio_bounds_list.append(self._compute_audio_bounds(ids_array))

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
            if offset > 0 and audio_bounds_list[i].size > 0:
                audio_bounds_list[i] = audio_bounds_list[i] + offset

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_inputs["pixel_values"],
            "image_sizes": image_inputs["image_sizes"],
            "tgt_sizes": image_inputs["tgt_sizes"],
            "image_bound": image_bounds_list,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": [np.zeros((0, 2), dtype=np.int32) for _ in range(batch_size)],
        }

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        image_input_names = getattr(self.image_processor, "model_input_names", [])
        audio_input_names = ["audio_features", "audio_feature_lens", "audio_bounds"]
        return list(
            dict.fromkeys(
                list(tokenizer_input_names)
                + list(image_input_names)
                + list(audio_input_names)
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        chat_template = kwargs.pop("chat_template", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        audio_processor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path
        )
        image_processor = MiniCPMOImageProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        return cls(
            image_processor=image_processor,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


install_auto_processor_patch("minicpmo", MiniCPMOProcessor)
