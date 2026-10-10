# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""EmbeddingGemma 2 preprocessing, adapted from the reference Transformers processor.

Uses NumPy/Pillow vision preprocessing and the NumPy Gemma 4 audio extractor.
"""

import copy
from pathlib import Path

import numpy as np
from transformers.audio_utils import AudioInput, is_valid_audio
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput, is_valid_video, make_batched_videos

from ..base import install_auto_processor_patch, to_mlx
from ..gemma4.processing_gemma4 import get_aspect_ratio_preserving_size


class EmbeddingGemma2ImageKwargs(ImagesKwargs, total=False):
    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


class EmbeddingGemma2TextKwargs(TextKwargs, total=False):
    return_text_replacement_offsets: bool


class EmbeddingGemma2ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: EmbeddingGemma2TextKwargs
    images_kwargs: EmbeddingGemma2ImageKwargs
    _defaults = {
        "text_kwargs": {"padding": True},
        "images_kwargs": {"do_convert_rgb": True},
        "audio_kwargs": {},
        "videos_kwargs": {"return_metadata": True},
    }


class EmbeddingGemma2Processor(ProcessorMixin):
    valid_processor_kwargs = EmbeddingGemma2ProcessorKwargs

    def check_argument_for_proper_class(self, argument_name, argument):
        # Transformers exposes a dummy BaseVideoProcessor without TorchVision.
        if argument_name == "video_processor":
            from .video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor

            if isinstance(argument, EmbeddingGemma2VideoProcessor):
                return type(argument)
        return super().check_argument_for_proper_class(argument_name, argument)

    def _load_audio(self, audio, sampling_rate):
        from mlx_vlm.utils import load_audio

        if isinstance(audio, (str, Path)):
            return load_audio(audio, sr=sampling_rate)
        if isinstance(audio, (list, tuple)) and not is_valid_audio(audio):
            return [self._load_audio(item, sampling_rate) for item in audio]
        return audio

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        # Resolve file audio before ProcessorMixin can select a Torch backend.
        audio_from_video = kwargs.pop("load_audio_from_video", False)
        if kwargs.get("tokenize", False):
            conversation = copy.deepcopy(conversation)
            processor_kwargs = kwargs.get("processor_kwargs") or {}
            audio_kwargs = processor_kwargs.get(
                "audio_kwargs", kwargs.get("audio_kwargs", {})
            )
            sampling_rate = kwargs.get(
                "sampling_rate",
                processor_kwargs.get(
                    "sampling_rate",
                    audio_kwargs.get(
                        "sampling_rate", self.feature_extractor.sampling_rate
                    ),
                ),
            )
            conversations = (
                [conversation] if isinstance(conversation[0], dict) else conversation
            )
            for messages in conversations:
                for message in messages:
                    content = message.get("content", []) or []
                    for item in list(content):
                        if isinstance(item, dict) and item.get("type") == "audio":
                            for key in ("audio", "url", "path"):
                                if key in item:
                                    item[key] = self._load_audio(
                                        item[key], sampling_rate
                                    )
                        elif (
                            audio_from_video
                            and isinstance(item, dict)
                            and item.get("type") == "video"
                        ):
                            for key in ("video", "url", "path"):
                                if key in item:
                                    content.append(
                                        {
                                            "type": "audio",
                                            "audio": self._load_audio(
                                                item[key], sampling_rate
                                            ),
                                        }
                                    )
        return super().apply_chat_template(
            conversation, chat_template=chat_template, **kwargs
        )

    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
        self.video_token = "<|video|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_seq_length = audio_seq_length
        self.audio_ms_per_token = audio_ms_per_token
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        self.audio_token = getattr(tokenizer, "audio_token", None)
        self.boa_token = getattr(tokenizer, "boa_token", None)
        self.eoa_token = getattr(tokenizer, "eoa_token", None)
        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: (
            TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
        ) = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        nested_audio = (
            isinstance(audio, (list, tuple))
            and audio
            and all(
                isinstance(sample, (list, tuple))
                and (not sample or not is_valid_audio(sample))
                for sample in audio
            )
        )
        if not text:
            audio_per_sample = (
                [len(sample) for sample in audio]
                if nested_audio
                else (
                    [
                        (
                            len(el)
                            if isinstance(el, (list, tuple))
                            and (not is_valid_audio(el))
                            else 1
                        )
                        for el in audio
                    ]
                    if isinstance(audio, (list, tuple)) and (not is_valid_audio(audio))
                    else None
                )
            )
            videos_per_sample = (
                [
                    (
                        len(make_batched_videos(el))
                        if not (isinstance(el, (list, tuple)) and (not el))
                        else 0
                    )
                    for el in videos
                ]
                if isinstance(videos, (list, tuple)) and (not is_valid_video(videos))
                else None
            )
        # Transformers 5.14 does not flatten nested per-sample audio yet.
        if nested_audio:
            audio = [item for sample in audio for item in sample]
        if audio is not None:
            audio_kwargs = kwargs.get("audio_kwargs") or {}
            audio = self._load_audio(
                audio,
                kwargs.get(
                    "sampling_rate",
                    audio_kwargs.get(
                        "sampling_rate", self.feature_extractor.sampling_rate
                    ),
                ),
            )
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if images is not None:
            images = make_nested_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)
        if not text:
            modality_counts = []
            if images is not None:
                modality_counts.append(
                    (self.image_token, [len(image_list) for image_list in images])
                )
            if videos is not None:
                modality_counts.append(
                    (self.video_token, videos_per_sample or [1] * len(videos))
                )
            if audio is not None:
                modality_counts.append(
                    (self.audio_token, audio_per_sample or [1] * len(audio))
                )
            if modality_counts:
                batch_sizes = {len(counts) for _, counts in modality_counts}
                if len(batch_sizes) > 1:
                    raise ValueError(
                        f"Received inconsistently sized modality batches when `text` is None: {[len(counts) for _, counts in modality_counts]}."
                    )
                batch_size = batch_sizes.pop()
                text = [
                    " ".join(
                        (
                            token
                            for token, counts in modality_counts
                            for _ in range(counts[sample_idx])
                        )
                    )
                    for sample_idx in range(batch_size)
                ]
        return (images, text, videos, audio)

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: (
            TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
        ) = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)
        if text is None and images is None and (videos is None) and (audio is None):
            raise ValueError(
                "You must provide at least one of `text`, `images`, `videos`, or `audio`."
            )
        if audio is not None and (
            self.audio_token is None or self.boa_token is None or self.eoa_token is None
        ):
            raise ValueError(
                "Audio inputs were provided, but the tokenizer does not have an `audio_token` defined."
            )
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                if len(images) != len(text):
                    raise ValueError(
                        f"Received inconsistently sized batches of images ({len(images)}) and text ({len(text)})."
                    )
                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed. Found {n_images_in_text} {self.image_token} tokens and {n_images_in_images} images per sample."
                    )
            elif any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            n_videos_in_text = [sample.count(self.video_token) for sample in text]
            if videos is not None:
                if sum(n_videos_in_text) != len(videos):
                    raise ValueError(
                        f"The total number of {self.video_token} tokens in the prompts should be the same as the number of videos passed. Found {sum(n_videos_in_text)} {self.video_token} tokens and {len(videos)} videos."
                    )
            elif any(n_videos_in_text):
                raise ValueError(
                    f"Found {sum(n_videos_in_text)} {self.video_token} tokens in the text but no videos were passed."
                )
            if self.audio_token is not None:
                n_audio_in_text = [sample.count(self.audio_token) for sample in text]
                if audio is not None:
                    n_audio_passed = (
                        len(audio) if isinstance(audio, (list, tuple)) else 1
                    )
                    if sum(n_audio_in_text) != n_audio_passed:
                        raise ValueError(
                            f"The total number of {self.audio_token} tokens in the prompts should be the same as the number of audio inputs passed. Found {sum(n_audio_in_text)} {self.audio_token} tokens and {n_audio_passed} audio inputs."
                        )
                elif any(n_audio_in_text):
                    raise ValueError(
                        f"Found {sum(n_audio_in_text)} {self.audio_token} tokens in the text but no audio inputs were passed."
                    )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
        return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        num_soft_tokens = video_inputs["num_soft_tokens_per_video"][video_idx]
        add_timestamps = kwargs.get(
            "add_timestamps", self.video_processor.add_timestamps
        )
        if not add_timestamps:
            num_frames = int(video_inputs["num_frames_per_video"][video_idx])
            frame_str = (
                f"{self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
            )
            return "".join([frame_str] * num_frames)
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None:
            raise ValueError(
                "Asked to build a prompt with frame timestamps, but no `fps` was provided in video metadata. The capture rate of already-decoded frames cannot be inferred. Please pass a `VideoMetadata` object with a valid `fps`, or set `add_timestamps=False`."
            )
        timestamp_str = [
            f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
            for seconds in metadata.timestamps
        ]
        return " ".join(
            [
                f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
                for t in timestamp_str
            ]
        )

    def _process_videos(self, videos, **kwargs):
        # Forward per-call timestamp settings on Transformers 5.14 as well.
        inputs = self.video_processor(videos, **kwargs)
        replacements = [
            self.replace_video_token(inputs, index, **kwargs)
            for index in range(len(make_batched_videos(videos)))
        ]
        return inputs, replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        mask = audio_inputs["input_features_mask"][audio_idx]
        t = len(mask)
        for _ in range(2):
            t_out = (t + 2 - 3) // 2 + 1
            mask = mask[::2][:t_out]
            t = len(mask)
        return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"

    def _get_num_multimodal_tokens(
        self, image_sizes=None, audio_lengths=None, **kwargs
    ):
        images_kwargs = dict(
            EmbeddingGemma2ProcessorKwargs._defaults.get("images_kwargs", {})
        )
        images_kwargs.update(kwargs)
        patch_size = (
            images_kwargs.get("patch_size", None) or self.image_processor.patch_size
        )
        pooling_kernel_size = (
            images_kwargs.get("pooling_kernel_size", None)
            or self.image_processor.pooling_kernel_size
        )
        max_soft_tokens = (
            images_kwargs.get("max_soft_tokens", None)
            or self.image_processor.max_soft_tokens
        )
        max_patches = max_soft_tokens * pooling_kernel_size**2
        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = []
            for image_size in image_sizes:
                target_h, target_w = get_aspect_ratio_preserving_size(
                    height=image_size[0],
                    width=image_size[1],
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                )
                patch_height = target_h // patch_size
                patch_width = target_w // patch_size
                num_image_tokens.append(
                    patch_height * patch_width // pooling_kernel_size**2
                )
            num_image_patches = [1] * len(image_sizes)
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )
        if audio_lengths is not None:
            sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
            num_audio_tokens = [
                self._compute_audio_num_tokens(np.zeros(length), sampling_rate)
                for length in audio_lengths
            ]
            vision_data.update({"num_audio_tokens": num_audio_tokens})
        return MultiModalData(**vision_data)

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate: int) -> int:
        num_samples = len(audio_waveform)
        frame_size_for_unfold = self.feature_extractor.frame_length + 1
        pad_left = self.feature_extractor.frame_length // 2
        num_mel_frames = (
            num_samples + pad_left - frame_size_for_unfold
        ) // self.feature_extractor.hop_length + 1
        if num_mel_frames <= 0:
            return 0
        sscp_num_layers = 2
        sscp_kernel, sscp_stride, sscp_padding = (3, 2, 1)
        t = num_mel_frames
        for _ in range(sscp_num_layers):
            t = (t + 2 * sscp_padding - sscp_kernel) // sscp_stride + 1
        return min(t, self.audio_seq_length)

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_soft_tokens_per_image", "num_soft_tokens_per_video"]

    supports_multiple_audio = True

    def to_dict(self):
        # MLX-VLM adds a runtime detokenizer; serialize only constructor settings.
        processor = copy.copy(self)
        saved = self.get_attributes() + [
            "image_seq_length",
            "audio_seq_length",
            "audio_ms_per_token",
            "auto_map",
        ]
        processor.__dict__ = {k: v for k, v in self.__dict__.items() if k in saved}
        return ProcessorMixin.to_dict(processor)

    def _merge_kwargs(self, ModelProcessorKwargs, tokenizer_init_kwargs=None, **kwargs):
        # Tokenizer reload adds unset defaults; do not override audio defaults with None.
        tokenizer_init_kwargs = {
            k: v for k, v in (tokenizer_init_kwargs or {}).items() if v is not None
        }
        return super()._merge_kwargs(
            ModelProcessorKwargs, tokenizer_init_kwargs=tokenizer_init_kwargs, **kwargs
        )

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        audio=None,
        return_tensors=None,
        **kwargs,
    ):
        from transformers.feature_extraction_utils import BatchFeature

        if return_tensors is not None:
            kwargs["return_tensors"] = (
                "np" if return_tensors == "mlx" else return_tensors
            )
        inputs = super().__call__(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if return_tensors != "mlx":
            return inputs
        metadata = {
            key: value
            for key, value in inputs.items()
            if key in self.skip_tensor_conversion
        }
        return BatchFeature(
            {
                **to_mlx({k: v for k, v in inputs.items() if k not in metadata}),
                **metadata,
            }
        )

    @classmethod
    def _get_arguments_from_pretrained(
        cls, pretrained_model_name_or_path, processor_dict=None, **kwargs
    ):
        from transformers import AutoTokenizer
        from transformers.models.gemma4.feature_extraction_gemma4 import (
            Gemma4AudioFeatureExtractor,
        )

        from .image_processing_embedding_gemma2 import EmbeddingGemma2ImageProcessor
        from .video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor

        config = processor_dict or {}
        audio_config = dict(config.get("feature_extractor", {}))
        # The feature extractor saves lengths in samples, but initializes in ms.
        sampling_rate = audio_config.get("sampling_rate", 16000)
        for name in ("frame_length", "hop_length"):
            if name in audio_config:
                audio_config[name + "_ms"] = audio_config[name] * 1000 / sampling_rate
        return [
            Gemma4AudioFeatureExtractor(**audio_config),
            EmbeddingGemma2ImageProcessor(**config.get("image_processor", {})),
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
            EmbeddingGemma2VideoProcessor(**config.get("video_processor", {})),
        ]


install_auto_processor_patch("embedding_gemma2", EmbeddingGemma2Processor)
