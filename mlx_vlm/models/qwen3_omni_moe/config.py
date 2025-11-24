import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class Code2WavConfig(BaseModelConfig):
    model_type: str = ""
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000
    max_position_embeddings: int = 8000
    sliding_window: Optional[int] = 72
    codebook_dim: int = 512
    codebook_size: int = 2048
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    semantic_codebook_size: int = 4096
    vector_quantization_hidden_dimension: int = 512
    layer_scale_initial_scale: float = 0.01
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: List[int] = field(default_factory=lambda: [2, 2])


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "qwen3_omni_moe_audio_encoder"
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_hidden_layers: int = 32
    num_mel_bins: int = 128
    output_dim: int = 2048
    downsample_hidden_size: int = 480
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    dropout: float = 0.0
    initializer_range: float = 0.02
    scale_embedding: bool = False
    conv_chunksize: int = 500
    n_window: int = 50
    n_window_infer: int = 800
    max_source_positions: int = 1500


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_omni_moe_vision_encoder"
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    out_hidden_size: int = 2048
    num_heads: int = 16
    image_size: int = 768
    patch_size: int = 16
    spatial_patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: int = 2
    in_channels: int = 3
    in_chans: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    initializer_range: float = 0.02
    apply_vit_abs_pos_embed: bool = True
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])


@dataclass
class TextConfig(BaseModelConfig):
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    model_type: str = "qwen3_omni_moe_text_encoder"
    norm_topk_prob: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default"}
    )
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    router_aux_loss_coef: float = 0.001
    use_qk_norm: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class CodePredictorConfig(BaseModelConfig):
    model_type: str = "qwen3_omni_moe_talker_code_predictor"
    num_hidden_layers: int = 5
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    max_position_embeddings: int = 32768
    vocab_size: int = 2048
    num_code_groups: int = 16
    max_window_layers: int = 28
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    layer_types: List[str] = field(default_factory=lambda: ["full_attention"] * 5)
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = None
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None


@dataclass
class ThinkerConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: AudioConfig
    model_type: str = "qwen3_omni_moe_thinker"
    dtype: str = "bfloat16"
    initializer_range: float = 0.02
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151675
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    user_token_id: int = 872
    position_id_per_seconds: int = 13
    seconds_per_chunk: int = 2

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_config = TextConfig.from_dict(params.pop("text_config", {}))
        vision_config = VisionConfig.from_dict(params.pop("vision_config", {}))
        audio_config = AudioConfig.from_dict(params.pop("audio_config", {}))

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )


@dataclass
class TalkerConfig(BaseModelConfig):
    text_config: TextConfig
    code_predictor_config: CodePredictorConfig
    model_type: str = "qwen3_omni_moe_talker"
    accept_hidden_layer: int = 24
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151675
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    vision_start_token_id: int = 151652
    num_code_groups: int = 16
    output_router_logits: bool = False
    position_id_per_seconds: int = 13
    seconds_per_chunk: int = 2
    spatial_merge_size: int = 2
    thinker_hidden_size: int = 2048
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_nothink_id: int = 2155
    codec_pad_id: int = 2148
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    speaker_id: Dict[str, int] = field(
        default_factory=lambda: {"chelsie": 2301, "ethan": 2302, "aiden": 2303}
    )

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_config = TextConfig.from_dict(params.pop("text_config", {}))
        code_predictor_config = CodePredictorConfig.from_dict(
            params.pop("code_predictor_config", {})
        )

        return cls(
            text_config=text_config,
            code_predictor_config=code_predictor_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )


@dataclass
class ModelConfig(BaseModelConfig):
    thinker_config: ThinkerConfig
    talker_config: TalkerConfig
    code2wav_config: Code2WavConfig
    model_type: str = "qwen3_omni_moe"
    dtype: str = "bfloat16"
    enable_audio_output: bool = True
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    system_token_id: int = 8948
    user_token_id: int = 872
    assistant_token_id: int = 77091
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        thinker_config = ThinkerConfig.from_dict(params.pop("thinker_config", {}))
        talker_config = TalkerConfig.from_dict(params.pop("talker_config", {}))
        code2wav_config = Code2WavConfig.from_dict(params.pop("code2wav_config", {}))

        return cls(
            thinker_config=thinker_config,
            talker_config=talker_config,
            code2wav_config=code2wav_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )
