from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ViTConfig(BaseModelConfig):
    model_type: str = "sam3_vit_model"
    hidden_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 4736
    hidden_act: str = "gelu"
    image_size: int = 1008
    patch_size: int = 14
    num_channels: int = 3
    window_size: int = 24
    global_attn_indexes: List[int] = field(default_factory=lambda: [7, 15, 23, 31])
    qkv_bias: bool = True
    rope_theta: float = 10000.0
    pretrain_image_size: int = 336
    layer_norm_eps: float = 1e-6
    layer_scale_init_value: Optional[float] = None
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class VisionEncoderConfig(BaseModelConfig):
    model_type: str = "sam3_vision_model"
    backbone_config: Optional[dict] = None
    fpn_hidden_size: int = 256
    fpn_kernel_size: int = 2
    fpn_stride: int = 2
    scale_factors: List[float] = field(default_factory=lambda: [4.0, 2.0, 1.0, 0.5])
    num_feature_levels: int = 3
    backbone_feature_sizes: List[List[int]] = field(
        default_factory=lambda: [[288, 288], [144, 144], [72, 72]]
    )
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        if isinstance(self.backbone_config, dict):
            self.backbone_config = ViTConfig.from_dict(self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = ViTConfig()


@dataclass
class TextEncoderConfig(BaseModelConfig):
    model_type: str = "clip_text_model"
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    vocab_size: int = 49408
    max_position_embeddings: int = 32
    projection_dim: int = 512
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    pad_token_id: int = 1


@dataclass
class DETREncoderConfig(BaseModelConfig):
    model_type: str = "sam3_detr_encoder"
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "relu"
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6


@dataclass
class DETRDecoderConfig(BaseModelConfig):
    model_type: str = "sam3_detr_decoder"
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    num_queries: int = 200
    intermediate_size: int = 2048
    hidden_act: str = "relu"
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    box_rpb_mode: str = "log"
    use_presence_token: bool = True


@dataclass
class GeometryEncoderConfig(BaseModelConfig):
    model_type: str = "sam3_geometry_encoder"
    hidden_size: int = 256
    num_layers: int = 3
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "relu"
    dropout: float = 0.1
    roi_size: int = 7
    layer_norm_eps: float = 1e-6


@dataclass
class DetectorMaskDecoderConfig(BaseModelConfig):
    model_type: str = "sam3_mask_decoder"
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_upsampling_stages: int = 3
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6


@dataclass
class DetectorConfig(BaseModelConfig):
    model_type: str = "sam3"
    vision_config: Optional[dict] = None
    text_config: Optional[dict] = None
    detr_encoder_config: Optional[dict] = None
    detr_decoder_config: Optional[dict] = None
    geometry_encoder_config: Optional[dict] = None
    mask_decoder_config: Optional[dict] = None
    initializer_range: float = 0.02

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionEncoderConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionEncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config = TextEncoderConfig.from_dict(self.text_config)
        elif self.text_config is None:
            self.text_config = TextEncoderConfig()

        if isinstance(self.detr_encoder_config, dict):
            self.detr_encoder_config = DETREncoderConfig.from_dict(
                self.detr_encoder_config
            )
        elif self.detr_encoder_config is None:
            self.detr_encoder_config = DETREncoderConfig()

        if isinstance(self.detr_decoder_config, dict):
            self.detr_decoder_config = DETRDecoderConfig.from_dict(
                self.detr_decoder_config
            )
        elif self.detr_decoder_config is None:
            self.detr_decoder_config = DETRDecoderConfig()

        if isinstance(self.geometry_encoder_config, dict):
            self.geometry_encoder_config = GeometryEncoderConfig.from_dict(
                self.geometry_encoder_config
            )
        elif self.geometry_encoder_config is None:
            self.geometry_encoder_config = GeometryEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = DetectorMaskDecoderConfig.from_dict(
                self.mask_decoder_config
            )
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = DetectorMaskDecoderConfig()


@dataclass
class TrackerMaskDecoderConfig(BaseModelConfig):
    hidden_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    mlp_dim: int = 2048
    hidden_act: str = "gelu"
    dynamic_multimask_via_stability: bool = True
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98


@dataclass
class PromptEncoderConfig(BaseModelConfig):
    hidden_size: int = 256
    image_size: int = 1008
    patch_size: int = 14
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    scale: int = 1


@dataclass
class TrackerConfig(BaseModelConfig):
    model_type: str = "sam3_tracker_video"
    image_size: int = 1008
    vision_config: Optional[dict] = None
    mask_decoder_config: Optional[dict] = None
    prompt_encoder_config: Optional[dict] = None

    # Memory attention
    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 4
    memory_attention_num_attention_heads: int = 1
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_feed_forward_hidden_act: str = "relu"
    memory_attention_dropout: float = 0.1
    memory_attention_rope_dropout: float = 0.1
    memory_attention_rope_theta: float = 10000.0
    memory_attention_rope_feat_sizes: List[int] = field(
        default_factory=lambda: [72, 72]
    )
    memory_attention_downsample_rate: int = 1

    # Memory encoder
    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64

    # Mask downsampler
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"

    # Memory fuser (CXBlock)
    memory_fuser_embed_dim: int = 256
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_num_layers: int = 2
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_hidden_act: str = "gelu"

    # Tracker settings
    num_maskmem: int = 7
    max_cond_frame_num: int = 4
    max_object_pointers_in_encoder: int = 16
    multimask_output_in_sam: bool = True
    multimask_output_for_tracking: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1

    # Memory encoding params
    sigmoid_bias_for_mem_enc: float = -10.0
    sigmoid_scale_for_mem_enc: float = 20.0

    # Occlusion / temporal
    enable_occlusion_spatial_embedding: bool = True
    enable_temporal_pos_encoding_for_object_pointers: bool = True

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionEncoderConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = TrackerMaskDecoderConfig.from_dict(
                self.mask_decoder_config
            )
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = TrackerMaskDecoderConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = PromptEncoderConfig.from_dict(
                self.prompt_encoder_config
            )
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = PromptEncoderConfig()


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "sam3_video"
    detector_config: Optional[dict] = None
    tracker_config: Optional[dict] = None
    initializer_range: float = 0.02
    low_res_mask_size: int = 288

    # Tracking / association thresholds
    det_nms_thresh: float = 0.1
    assoc_iou_thresh: float = 0.1
    trk_assoc_iou_thresh: float = 0.5
    high_conf_thresh: float = 0.8
    high_iou_thresh: float = 0.8
    new_det_thresh: float = 0.7
    score_threshold_detection: float = 0.5
    fill_hole_area: int = 16
    max_num_objects: int = 10000

    # Tracker management
    init_trk_keep_alive: int = 30
    max_trk_keep_alive: int = 30
    min_trk_keep_alive: int = -1
    hotstart_delay: int = 15
    hotstart_dup_thresh: int = 8
    hotstart_unmatch_thresh: int = 8
    recondition_every_nth_frame: int = 16
    recondition_on_trk_masks: bool = False
    decrease_trk_keep_alive_for_empty_masklets: bool = False
    suppress_unmatched_only_within_hotstart: bool = True
    suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.7

    # Placeholder configs for mlx-vlm compatibility
    text_config: Optional[dict] = None
    vision_config: Optional[dict] = None

    def __post_init__(self):
        if isinstance(self.detector_config, dict):
            self.detector_config = DetectorConfig.from_dict(self.detector_config)
        elif self.detector_config is None:
            self.detector_config = DetectorConfig()

        if isinstance(self.tracker_config, dict):
            self.tracker_config = TrackerConfig.from_dict(self.tracker_config)
        elif self.tracker_config is None:
            self.tracker_config = TrackerConfig()

        # Set text_config and vision_config from detector for mlx-vlm compatibility
        if self.text_config is None:
            self.text_config = self.detector_config.text_config
        if self.vision_config is None:
            self.vision_config = self.detector_config.vision_config
