from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional

from mlx_lm.models.nemotron_h import ModelArgs as TextConfig

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "radio"
    args: Optional[dict] = None
    version: str = "radio_v2.5-h"
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    image_size: int = 224
    patch_size: int = 16
    max_resolution: int = 2048
    preferred_resolution: Any = None
    adaptor_names: Any = None
    adaptor_configs: Optional[Dict[str, Dict[str, int]]] = None
    vitdet_window_size: Optional[int] = None
    feature_normalizer_config: Optional[dict] = None
    inter_feature_normalizer_config: Optional[dict] = None
    min_num_patches: Optional[int] = None
    max_num_patches: Optional[int] = None
    dynamic_image_min_num_patches: int = 1024
    dynamic_image_max_num_patches: int = 13312
    video_target_num_patches: int = 1024
    video_temporal_patch_size: int = 2
    separate_video_embedder: bool = True
    force_image_size: Optional[int] = None

    def __post_init__(self):
        if self.min_num_patches is not None:
            self.dynamic_image_min_num_patches = self.min_num_patches
        if self.max_num_patches is not None:
            self.dynamic_image_max_num_patches = self.max_num_patches


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "parakeet"
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    feat_in: Optional[int] = None
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float = 0.0
    dropout_positions: float = 0.0
    layerdrop: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    max_position_embeddings: int = 5000
    scale_input: bool = False
    initializer_range: float = 0.02
    projection_hidden_size: int = 4096
    projection_bias: bool = False
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 512
    win_length: int = 400
    preemphasis: float = 0.97

    def __post_init__(self):
        if self.feat_in is None:
            self.feat_in = self.num_mel_bins
        self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    sound_config: Optional[AudioConfig] = None
    model_type: str = "nemotron_h_nano_omni"
    force_image_size: Optional[int] = None
    downsample_ratio: float = 0.5
    template: Optional[str] = None
    ps_version: str = "v1"
    image_tag_type: str = "internvl"
    projector_hidden_size: int = 4096
    vit_hidden_size: int = 1280
    attn_implementation: Optional[str] = None
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    img_context_token_id: Optional[int] = None
    video_context_token_id: Optional[int] = None
    sound_context_token_id: Optional[int] = None
    eos_token_id: Any = None
    image_token_index: Optional[int] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        text_config = TextConfig.from_dict(
            params.pop("text_config", params.pop("llm_config", {}))
        )
        vision_config = VisionConfig.from_dict(params.pop("vision_config", {}))

        raw_sound_config = params.pop("sound_config", None)
        sound_config = (
            AudioConfig.from_dict(raw_sound_config)
            if raw_sound_config is not None
            else None
        )

        allowed = cls.__dataclass_fields__
        kwargs = {k: v for k, v in params.items() if k in allowed}
        config = cls(
            text_config=text_config,
            vision_config=vision_config,
            sound_config=sound_config,
            **kwargs,
        )
        if config.image_token_index is None:
            config.image_token_index = config.img_context_token_id
        return config

    def to_dict(self):
        def convert(value):
            if is_dataclass(value):
                return asdict(value)
            if hasattr(value, "__dict__"):
                return {
                    k: convert(v)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_") and v is not None
                }
            return value

        return {k: convert(v) for k, v in self.__dict__.items() if v is not None}
