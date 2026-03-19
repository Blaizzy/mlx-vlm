"""
MLX-based processor for Phi4-Multimodal (SigLIP2 NaFlex vision + Cascades audio).

Provides:
- Phi4MMImageProcessor: NaFlex image processing (variable patches, no upscaling)
- Phi4MMAudioFeatureExtractor: Mel spectrogram feature extraction
- Phi4MMProcessor: Combined tokenizer + image processor + audio processor
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from ..base import install_auto_processor_patch

# Constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
AUDIO_TOKEN_INDEX = 200011
DEFAULT_AUDIO_TOKEN = "<|endoftext11|>"
AUDIO_PLACEHOLDER_PATTERN = "<|audio_"  # e.g., <|audio_1|>


# =============================================================================
# NaFlex Image Processing Helpers
# =============================================================================


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
) -> Tuple[int, int]:
    """Calculate target image size to fit within max_num_patches."""
    aspect_ratio = image_width / image_height

    # Calculate height in patches
    max_height_patches = int(math.sqrt(max_num_patches / aspect_ratio))
    max_width_patches = int(max_height_patches * aspect_ratio)

    # Adjust if over limit
    while max_height_patches * max_width_patches > max_num_patches:
        if max_height_patches > max_width_patches:
            max_height_patches -= 1
        else:
            max_width_patches -= 1

    max_height_patches = max(max_height_patches, 1)
    max_width_patches = max(max_width_patches, 1)

    return max_height_patches * patch_size, max_width_patches * patch_size


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert image (H, W, C) to patches (num_patches, patch_dim)."""
    H, W, C = image.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Crop to patch-aligned size
    image = image[: h_patches * patch_size, : w_patches * patch_size]

    # Reshape to extract patches
    patches = image.reshape(h_patches, patch_size, w_patches, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(h_patches * w_patches, -1)

    return patches


def pad_along_first_dim(
    array: np.ndarray, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad array along first dimension and return (padded, mask)."""
    current_length = array.shape[0]
    if current_length >= max_length:
        return array[:max_length], np.ones(max_length, dtype=np.bool_)

    pad_shape = (max_length - current_length,) + array.shape[1:]
    padding = np.zeros(pad_shape, dtype=array.dtype)
    padded = np.concatenate([array, padding], axis=0)

    mask = np.zeros(max_length, dtype=np.bool_)
    mask[:current_length] = True

    return padded, mask


# =============================================================================
# Image Processor
# =============================================================================


class Phi4MMImageProcessor(BaseImageProcessor):
    """NaFlex image processor for Phi4-Multimodal (no upscaling of small images)."""

    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(
        self,
        image_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        patch_size: int = 14,
        max_num_patches: int = 3600,
        min_num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """Process images into NaFlex patches."""
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be PIL.Image.Image, numpy.ndarray, etc."
            )

        all_pixel_values = []
        all_masks = []
        all_spatial_shapes = []

        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size
            num_patches = max(
                (height // self.patch_size) * (width // self.patch_size), 1
            )

            # Determine target size (no upscale unless below minimum)
            if num_patches < self.min_num_patches:
                target_h, target_w = get_image_size_for_max_num_patches(
                    height, width, self.patch_size, self.min_num_patches
                )
            elif num_patches > self.max_num_patches:
                target_h, target_w = get_image_size_for_max_num_patches(
                    height, width, self.patch_size, self.max_num_patches
                )
            else:
                target_h, target_w = get_image_size_for_max_num_patches(
                    height, width, self.patch_size, num_patches
                )

            # Resize
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)

            # Convert to float and normalize
            arr = np.array(image).astype(np.float32) / 255.0
            mean = np.array(self.image_mean, dtype=np.float32)
            std = np.array(self.image_std, dtype=np.float32)
            arr = (arr - mean) / std

            # Convert to patches
            patches = convert_image_to_patches(arr, self.patch_size)
            patches, mask = pad_along_first_dim(patches, self.max_num_patches)

            h_patches = target_h // self.patch_size
            w_patches = target_w // self.patch_size

            all_pixel_values.append(patches)
            all_masks.append(mask)
            all_spatial_shapes.append((h_patches, w_patches))

        data = {
            "pixel_values": mx.array(np.stack(all_pixel_values)),
            "pixel_attention_mask": mx.array(np.stack(all_masks)),
            "spatial_shapes": mx.array(np.array(all_spatial_shapes)),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


# =============================================================================
# Audio Feature Extraction
# =============================================================================


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank matching SpeechLib FbankFC."""
    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)
    khi = max(khi, klo)

    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class Phi4MMAudioFeatureExtractor:
    """Extract mel spectrogram features from audio waveforms."""

    sampling_rate = 16000

    def __init__(
        self,
        audio_compression_rate=8,
        audio_downsample_rate=1,
        audio_feat_stride=1,
    ):
        self.compression_rate = audio_compression_rate
        self.qformer_compression_rate = audio_downsample_rate
        self.feat_stride = audio_feat_stride

        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T
        self._hamming400 = np.hamming(400)
        self._hamming200 = np.hamming(200)

    def _extract_spectrogram(self, wav, fs):
        """Extract spectrogram from waveform."""
        if wav.ndim > 1:
            wav = np.squeeze(wav)
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample if needed
        if fs > 16000:
            import scipy.signal

            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            import scipy.signal

            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            import scipy.signal

            wav = scipy.signal.resample_poly(wav, 2, 1)
            fs = 16000

        preemphasis = 0.97

        if fs == 8000:
            n_fft, win_length, hop_length = 256, 200, 80
            fft_window = self._hamming200
        else:
            n_fft, win_length, hop_length = 512, 400, 160
            fft_window = self._hamming400

        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        y_frames = np.array(
            [
                wav[s : s + win_length]
                for s in range(0, hop_length * n_batch, hop_length)
            ],
            dtype=np.float32,
        )

        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)
        spec = np.abs(S).astype(np.float32)
        return spec

    def _extract_features(self, wav, fs):
        """Extract log mel-filterbank features."""
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2
        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)
        return log_fbank

    def _compute_audio_embed_size(self, audio_frames):
        """Compute the number of audio tokens after encoder compression."""
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate
        result = integer if remainder == 0 else integer + 1

        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1

        return result

    def __call__(self, audios):
        """Process a list of (audio_data, sample_rate) tuples.

        Returns dict with:
        - input_audio_embeds: mx.array (B, T, 80)
        - audio_embed_sizes: mx.array (B,)
        - audio_attention_mask: mx.array (B, T) or None
        """
        all_embeds = []
        all_sizes = []
        all_frame_counts = []

        for audio_data, sample_rate in audios:
            features = self._extract_features(audio_data, sample_rate)
            audio_frames = len(features) * self.feat_stride
            embed_size = self._compute_audio_embed_size(audio_frames)

            all_embeds.append(features)
            all_sizes.append(embed_size)
            all_frame_counts.append(audio_frames)

        # Pad to same length
        max_len = max(e.shape[0] for e in all_embeds)
        padded = []
        for e in all_embeds:
            if e.shape[0] < max_len:
                pad = np.zeros((max_len - e.shape[0], e.shape[1]), dtype=np.float32)
                padded.append(np.concatenate([e, pad], axis=0))
            else:
                padded.append(e)

        input_audio_embeds = mx.array(np.stack(padded))
        audio_embed_sizes = mx.array(np.array(all_sizes))

        # Attention mask
        audio_attention_mask = None
        if len(audios) > 1:
            frame_counts = np.array(all_frame_counts)
            mask = np.arange(max_len)[None, :] < frame_counts[:, None]
            audio_attention_mask = mx.array(mask)

        result = {
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
        }
        if audio_attention_mask is not None:
            result["audio_attention_mask"] = audio_attention_mask

        return result


# =============================================================================
# Tokenizer Utilities
# =============================================================================


def tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
):
    """Tokenize a prompt with <image> placeholders replaced by image_token_index."""
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return input_ids


# =============================================================================
# Processor
# =============================================================================


class _AudioProcessorStub:
    """Minimal stub when no audio processor is configured."""

    sampling_rate = 16000


class Phi4MMProcessor(ProcessorMixin):
    """Combined processor for Phi4-Multimodal (tokenizer + NaFlex image + audio)."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Phi4MMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = Phi4MMImageProcessor()
        # Extract audio_processor from kwargs to avoid ProcessorMixin auto-detection
        self._audio_proc = kwargs.pop("audio_processor", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @property
    def audio_processor(self):
        return self._audio_proc

    @property
    def feature_extractor(self):
        """Alias for audio_processor, used by the framework audio loading pipeline."""
        return (
            self._audio_proc if self._audio_proc is not None else _AudioProcessorStub()
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audios=None,
        audio=None,
        **kwargs,
    ) -> BatchFeature:
        # Accept 'audio' (singular) from the framework's process_inputs
        if audio is not None and audios is None:
            audios = audio
        if images is None and text is None and audios is None:
            raise ValueError(
                "You have to specify at least one of `images`, `text`, or `audios`."
            )

        kwargs.pop("return_tensors", None)

        # Process images
        image_inputs = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            image_inputs = self.image_processor(images)

        # Process audio
        audio_inputs = None
        if audios is not None and self.audio_processor is not None:
            if not isinstance(audios, list):
                audios = [audios]
            # Convert numpy arrays (from load_audio) to (data, sr) tuples
            processed_audios = []
            for a in audios:
                if isinstance(a, tuple):
                    processed_audios.append(a)
                elif isinstance(a, np.ndarray):
                    processed_audios.append((a, 16000))
                else:
                    processed_audios.append(a)
            audio_inputs = self.audio_processor(processed_audios)

        # Process text
        if text is not None:
            if isinstance(text, str):
                texts = [text]
            else:
                texts = list(text)

            # Replace numbered placeholders with internal tokens
            import re

            texts = [re.sub(r"<\|image_\d+\|>", DEFAULT_IMAGE_TOKEN, t) for t in texts]
            texts = [re.sub(r"<\|audio_\d+\|>", DEFAULT_AUDIO_TOKEN, t) for t in texts]

            has_images = any(DEFAULT_IMAGE_TOKEN in t for t in texts)
            has_audio = any(DEFAULT_AUDIO_TOKEN in t for t in texts)

            if has_images and images is not None:
                # Tokenize with image token handling (splits on <image>)
                input_ids_list = []
                for t in texts:
                    ids = tokenizer_image_token(t, self.tokenizer)
                    input_ids_list.append(ids)

                # If audio is also present, expand audio tokens
                if has_audio and audio_inputs is not None:
                    audio_size_iter = iter(audio_inputs["audio_embed_sizes"].tolist())
                    expanded_list = []
                    for ids in input_ids_list:
                        expanded_ids = []
                        for token_id in ids:
                            if token_id == AUDIO_TOKEN_INDEX:
                                embed_size = int(next(audio_size_iter))
                                expanded_ids.extend([AUDIO_TOKEN_INDEX] * embed_size)
                            else:
                                expanded_ids.append(token_id)
                        expanded_list.append(expanded_ids)
                    input_ids_list = expanded_list

                # Pad sequences
                max_len = max(len(ids) for ids in input_ids_list)
                pad_token_id = self.tokenizer.pad_token_id or 0

                padded_ids = []
                attention_masks = []
                for ids in input_ids_list:
                    pad_len = max_len - len(ids)
                    padded_ids.append(ids + [pad_token_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)

                input_ids = mx.array(padded_ids)
                attention_mask = mx.array(attention_masks)
            elif has_audio and audio_inputs is not None:
                # For audio-only, the <|endoftext11|> token needs to be expanded
                # to audio_embed_size tokens in input_ids
                input_ids_list = []
                audio_size_iter = iter(audio_inputs["audio_embed_sizes"].tolist())
                for t in texts:
                    ids = self.tokenizer(t).input_ids
                    expanded_ids = []
                    for token_id in ids:
                        if token_id == AUDIO_TOKEN_INDEX:
                            # Expand single audio token to embed_size copies
                            embed_size = int(next(audio_size_iter))
                            expanded_ids.extend([AUDIO_TOKEN_INDEX] * embed_size)
                        else:
                            expanded_ids.append(token_id)
                    input_ids_list.append(expanded_ids)

                max_len = max(len(ids) for ids in input_ids_list)
                pad_token_id = self.tokenizer.pad_token_id or 0

                padded_ids = []
                attention_masks = []
                for ids in input_ids_list:
                    pad_len = max_len - len(ids)
                    padded_ids.append(ids + [pad_token_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)

                input_ids = mx.array(padded_ids)
                attention_mask = mx.array(attention_masks)
            else:
                # Standard tokenization
                text_inputs = self.tokenizer(
                    texts,
                    return_tensors="np",
                )
                input_ids = mx.array(text_inputs["input_ids"])
                attention_mask = mx.array(text_inputs["attention_mask"])
        else:
            input_ids = None
            attention_mask = None

        # Build output
        data = {}
        if input_ids is not None:
            data["input_ids"] = input_ids
            data["attention_mask"] = attention_mask

        if image_inputs is not None:
            data.update(image_inputs)

        if audio_inputs is not None:
            data.update(audio_inputs)

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = getattr(
            self.image_processor, "model_input_names", ["pixel_values"]
        )
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Load config to get vision and audio parameters
        image_processor_kwargs = {}
        audio_processor = None
        try:
            config_path = (
                model_path / "config.json"
                if is_local
                else Path(
                    __import__("huggingface_hub").hf_hub_download(
                        pretrained_model_name_or_path, "config.json"
                    )
                )
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            image_processor_kwargs["max_num_patches"] = config.get(
                "max_num_patches", 3600
            )
            image_processor_kwargs["min_num_patches"] = config.get(
                "min_num_patches", 256
            )

            vision_config = config.get("vision_config", {})
            if "patch_size" in vision_config:
                image_processor_kwargs["patch_size"] = vision_config["patch_size"]
            else:
                image_processor_kwargs["patch_size"] = 14

            # Get crop_size from embd_layer config
            embd_layer = config.get("embd_layer", {})
            image_embd = embd_layer.get("image_embd_layer", {})
            crop_size = image_embd.get("crop_size", 448)
            pos_emb_size = crop_size // image_processor_kwargs.get("patch_size", 14)
            if "max_num_patches" not in image_processor_kwargs:
                image_processor_kwargs["max_num_patches"] = pos_emb_size**2

            # Audio processor
            audio_proc_config = config.get("audio_processor", {})
            if audio_proc_config:
                audio_cfg = audio_proc_config.get("config", {})
                audio_processor = Phi4MMAudioFeatureExtractor(
                    audio_compression_rate=audio_cfg.get("time_reduction", 8),
                    audio_downsample_rate=1,
                    audio_feat_stride=1,
                )
        except Exception:
            pass

        # Read processor_config.json for correct init kwargs
        proc_cfg_path = model_path / "processor_config.json" if is_local else None
        proc_kwargs = {}
        if proc_cfg_path is not None and proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)

        image_processor = Phi4MMImageProcessor(**image_processor_kwargs)

        # Load chat template
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            try:
                jinja_path = (
                    model_path / "chat_template.jinja"
                    if is_local
                    else Path(
                        __import__("huggingface_hub").hf_hub_download(
                            pretrained_model_name_or_path, "chat_template.jinja"
                        )
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
            audio_processor=audio_processor,
            chat_template=chat_template,
            **proc_kwargs,
        )


# Register with AutoProcessor
install_auto_processor_patch(
    target_model_types=["phi4mm"],
    processor_cls=Phi4MMProcessor,
)
