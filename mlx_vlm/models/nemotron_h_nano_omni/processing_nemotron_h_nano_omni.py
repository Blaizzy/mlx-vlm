"""
Processor class for Nemotron-3 Nano Omni.

Adapted from NVIDIA's reference implementation:
https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/main/processing.py
"""

import math
from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

AudioInput = Union[str, np.ndarray, List[str], List[np.ndarray]]


class NemotronHNanoOmniProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        audio_sampling_rate: int = 16000,
        audio_subsampling_factor: int = 8,
        audio_hop_length: int = 160,
        audio_subsampling_conv_kernel_size: int = 3,
        audio_subsampling_conv_stride: int = 2,
        video_temporal_patch_dim: int = 2,
        **kwargs,
    ):
        self.video_temporal_patch_dim = video_temporal_patch_dim

        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else "<image>"
        )
        self.video_token = (
            tokenizer.video_token if hasattr(tokenizer, "video_token") else "<video>"
        )
        self.audio_token = (
            tokenizer.audio_token
            if hasattr(tokenizer, "audio_token")
            else "<so_embedding>"
        )
        self.audio_start_token = "<so_start>"
        self.audio_end_token = "<so_end>"
        self.image_start_token = (
            tokenizer.image_start_token
            if hasattr(tokenizer, "image_start_token")
            else "<img>"
        )
        self.image_end_token = (
            tokenizer.image_end_token
            if hasattr(tokenizer, "image_end_token")
            else "</img>"
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
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_hop_length = audio_hop_length
        self.audio_subsampling_conv_kernel_size = audio_subsampling_conv_kernel_size
        self.audio_subsampling_conv_stride = audio_subsampling_conv_stride

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        from .image_processing_nemotron_h_nano_omni import (
            NemotronHNanoOmniImageProcessor,
        )

        kwargs.pop("use_fast", None)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        def _read_json(name: str):
            local = Path(pretrained_model_name_or_path) / name
            if local.exists():
                with open(local) as f:
                    return json.load(f)
            try:
                path = hf_hub_download(str(pretrained_model_name_or_path), name)
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return {}

        proc_cfg = _read_json("processor_config.json")
        pre_cfg = _read_json("preprocessor_config.json")

        # Image-processor kwargs come from preprocessor_config.json (or the nested
        # `image_processor` block in processor_config.json on older exports).
        ip_cfg = pre_cfg or proc_cfg.get("image_processor", {})
        ip_kwargs = {
            k: ip_cfg[k]
            for k in (
                "norm_mean",
                "norm_std",
                "patch_size",
                "downsample_ratio",
                "min_num_patches",
                "max_num_patches",
                "max_model_len",
                "video_target_num_patches",
                "video_maintain_aspect_ratio",
            )
            if k in ip_cfg
        }
        image_processor = NemotronHNanoOmniImageProcessor(**ip_kwargs)

        proc_kwargs = {
            k: proc_cfg[k]
            for k in (
                "audio_sampling_rate",
                "audio_subsampling_factor",
                "audio_hop_length",
                "audio_subsampling_conv_kernel_size",
                "audio_subsampling_conv_stride",
                "video_temporal_patch_dim",
            )
            if k in proc_cfg
        }

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
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
        audio: Optional[AudioInput] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None and videos is None and audio is None:
            raise ValueError(
                "Provide at least one of `text`, `images`, `videos`, or `audio`."
            )

        images_kwargs = kwargs.pop("images_kwargs", {}) or {}
        videos_kwargs = kwargs.pop("videos_kwargs", {}) or {}
        audio_kwargs = kwargs.pop("audio_kwargs", {}) or {}
        text_kwargs = kwargs.pop("text_kwargs", {}) or {}
        # Allow flat kwargs (mlx-vlm convention) — fall through to text tokenization.
        text_kwargs = {**kwargs, **text_kwargs}

        image_inputs, videos_inputs, audio_inputs = {}, {}, {}

        if images is not None:
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_num_tokens = image_inputs["num_tokens"]

        if videos is not None:
            self.image_processor._is_video_mode = True
            try:
                videos_inputs = self.image_processor(images=videos, **images_kwargs)
            finally:
                self.image_processor._is_video_mode = False
            video_num_patches = [sum(videos_inputs["num_patches"])]
            videos_inputs["pixel_values_videos"] = videos_inputs["pixel_values"]
            del videos_inputs["pixel_values"]

        audio_num_tokens = []
        if audio is not None:
            audio_clips, audio_num_tokens = self._process_audio(audio, audio_kwargs)
            audio_inputs["sound_clips"] = audio_clips

        if text is None:
            text = []
        if isinstance(text, str):
            text = [text]
        text = list(text)

        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n_tokens = image_num_tokens[index]
                    text[i] = text[i].replace(
                        self.image_token,
                        self.image_start_token
                        + "<|placeholder|>" * n_tokens
                        + self.image_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            assert len(text) == 1, "Video is not supported for batch size > 1"
            video_metadata = videos_kwargs.get("video_metadata", None)
            i = 0
            index = 0
            if self.video_token in text[i]:
                tokens_per_tubelet = videos_inputs["num_tokens"][0]
                each_group = (
                    self.image_start_token
                    + "<|placeholder|>" * tokens_per_tubelet
                    + self.image_end_token
                )
                T = self.video_temporal_patch_dim
                n_frames = video_num_patches[index]
                n_groups = (n_frames + T - 1) // T

                source_fps = (
                    video_metadata.fps
                    if (
                        video_metadata is not None
                        and getattr(video_metadata, "fps", None)
                    )
                    else None
                )
                frames_indices = (
                    getattr(video_metadata, "frames_indices", None)
                    if video_metadata is not None
                    else None
                )
                frame_duration_ms = (
                    int(1000.0 / source_fps) if source_fps is not None else None
                )

                frame_labels = []
                for g in range(n_groups):
                    parts = []
                    for j in range(T):
                        fi = g * T + j
                        if fi >= n_frames:
                            break
                        prefix = "Frame" if j == 0 else "frame"
                        if (
                            source_fps is not None
                            and frames_indices is not None
                            and fi < len(frames_indices)
                        ):
                            ts = int(frames_indices[fi]) * frame_duration_ms / 1000.0
                            parts.append(f"{prefix} {fi+1} sampled at {ts:.2f} seconds")
                        elif source_fps is not None:
                            ts = fi / source_fps
                            parts.append(f"{prefix} {fi+1} sampled at {ts:.2f} seconds")
                        else:
                            parts.append(f"{prefix} {fi+1}")
                    frame_labels.append(" and ".join(parts) + ": ")

                video_prompt = ""
                for g, label in enumerate(frame_labels):
                    if g > 0:
                        video_prompt += "\n"
                    video_prompt += label + each_group

                text[i] = text[i].replace(self.video_token, video_prompt, 1)
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if audio is not None:
            index = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    num_tokens = (
                        audio_num_tokens[index] if index < len(audio_num_tokens) else 1
                    )
                    text[i] = text[i].replace(
                        self.audio_token,
                        self.audio_start_token
                        + "<|audio_placeholder|>" * num_tokens
                        + self.audio_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        text_kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **text_kwargs)

        result = BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **videos_inputs})
        )

        if audio_inputs:
            result["sound_clips"] = audio_inputs["sound_clips"]

        return result

    def _process_audio(self, audio: AudioInput, audio_kwargs: dict):
        sampling_rate = audio_kwargs.get("sampling_rate", self.audio_sampling_rate)

        if not isinstance(audio, list):
            audio = [audio]

        audio_clips = []
        num_tokens = []
        for audio_item in audio:
            if isinstance(audio_item, str):
                waveform = self._load_audio(audio_item, sampling_rate)
            elif isinstance(audio_item, np.ndarray):
                waveform = audio_item.squeeze() if audio_item.ndim > 1 else audio_item
            elif hasattr(audio_item, "numpy"):
                waveform = np.asarray(audio_item).squeeze()
            else:
                raise ValueError(f"Unsupported audio type: {type(audio_item)}")

            audio_clips.append(waveform)
            n_tokens = self._estimate_audio_num_embeddings(len(waveform))
            num_tokens.append(max(1, n_tokens))

        return audio_clips, num_tokens

    def _estimate_audio_num_embeddings(self, audio_length_samples: int) -> int:
        n_frames = 1 + audio_length_samples // self.audio_hop_length
        kernel_size = self.audio_subsampling_conv_kernel_size
        stride = self.audio_subsampling_conv_stride
        num_layers = int(math.log2(self.audio_subsampling_factor))
        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        L = n_frames
        for _ in range(num_layers):
            L = (L + add_pad) // stride + 1
        return L

    def _load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        from mlx_audio.audio_io import read as read_audio
        from mlx_audio.utils import resample_audio

        waveform, sr = read_audio(audio_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            waveform = resample_audio(waveform, sr, target_sr)
        return waveform.astype(np.float32)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["NemotronHNanoOmniProcessor"]


install_auto_processor_patch(
    ["nemotronh_nano_omni_reasoning_v3", "nemotron_h_nano_omni"],
    NemotronHNanoOmniProcessor,
)
