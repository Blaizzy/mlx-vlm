"""SAM 3.1 configuration — extends SAM 3 configs with multiplex support."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig
from ..sam3.config import (
    DetectorMaskDecoderConfig,
    DETRDecoderConfig,
    DETREncoderConfig,
    GeometryEncoderConfig,
    PromptEncoderConfig,
    TextEncoderConfig,
    ViTConfig,
)


@dataclass
class VisionEncoderConfig(BaseModelConfig):
    """SAM 3.1 vision encoder — TriViTDetNeck with 3 scales."""

    model_type: str = "sam3_vision_model"
    backbone_config: Optional[dict] = None
    fpn_hidden_size: int = 256
    fpn_kernel_size: int = 2
    fpn_stride: int = 2
    # SAM 3.1: only 3 scales (no 0.5x downsample)
    scale_factors: List[float] = field(default_factory=lambda: [4.0, 2.0, 1.0])
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
class TrackerMaskDecoderConfig(BaseModelConfig):
    """SAM 3.1 tracker mask decoder — multiplex version."""

    hidden_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    mlp_dim: int = 2048
    dynamic_multimask_via_stability: bool = True
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98
    # SAM 3.1 multiplex
    multiplex_count: int = 16
    # Propagation decoder has no single-mask token (multimask tokens only)
    multimask_outputs_only: bool = False
    # Use multimask tokens (not the single-mask token) for object pointers
    use_multimask_token_for_obj_ptr: bool = True


@dataclass
class TrackerConfig(BaseModelConfig):
    """SAM 3.1 tracker — Object Multiplex."""

    model_type: str = "sam3.1_tracker_video"
    image_size: int = 1008
    vision_config: Optional[dict] = None
    mask_decoder_config: Optional[dict] = None
    prompt_encoder_config: Optional[dict] = None

    # Multiplex
    multiplex_count: int = 16

    # Memory attention (decoupled)
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_hidden_size: int = 256
    memory_attention_num_attention_heads: int = 8
    memory_attention_num_layers: int = 4
    memory_attention_rope_feat_sizes: List[int] = field(
        default_factory=lambda: [72, 72]
    )
    memory_attention_rope_theta: float = 10000.0

    # Memory encoder — mask downsampler (SAM 3.1: dim = out_dim = 256)
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_first_channels: int = 16  # actual from checkpoint
    mask_downsampler_input_size: int = 1152  # resize masks to this pre-convs
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_padding: int = 1
    mask_downsampler_stride: int = 2
    memory_encoder_hidden_size: int = 256

    # Memory fuser (CXBlock)
    memory_fuser_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_kernel_size: int = 7
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_num_layers: int = 2
    memory_fuser_padding: int = 3

    # Memory retrieval over past frames
    max_cond_frame_num: int = 4
    max_object_pointers_in_encoder: int = 16
    memory_temporal_stride_for_eval: int = 1
    num_maskmem: int = 7
    save_image_features: bool = True
    use_maskmem_tpos_v2: bool = True

    # Memory encoding: mask -> memory (SAM 3.1: sigmoid(logits) * 2.0 - 1.0)
    apply_sigmoid_to_mask_logits_for_mem_enc: bool = True
    sigmoid_bias_for_mem_enc: float = -1.0
    sigmoid_scale_for_mem_enc: float = 2.0
    # Extra per-object channel marking conditioning objects
    condition_as_mask_input: bool = True
    condition_as_mask_input_bg: float = 0.0
    condition_as_mask_input_fg: float = 1.0

    # SAM head behavior (multimask, mask-as-output)
    directly_add_no_mem_embed: bool = True
    multimask_max_pt_num: int = 1
    multimask_min_pt_num: int = 0
    multimask_output_for_tracking: bool = True
    multimask_output_in_sam: bool = True
    num_multimask_outputs: int = 3
    use_mask_input_as_output_without_sam: bool = True

    # Object presence scores and pointers
    add_output_suppression_embeddings: bool = True
    fixed_no_obj_ptr: bool = True
    object_score_logit_threshold: float = 0.0
    pred_obj_scores: bool = True
    use_linear_no_obj_ptr: bool = True
    use_no_obj_ptr: bool = True
    use_obj_ptrs_in_encoder: bool = True

    def __post_init__(self):
        # The facebook/sam3.1 HF config carries a stale SAM 3 (non-multiplex)
        # tracker_config. Re-pin the SAM 3.1 multiplex architectural constants
        # (see build_sam3_multiplex_video_model in the SAM 3 repo).
        if self.model_type == "sam3_tracker_video":
            self.memory_attention_num_attention_heads = 8
            self.sigmoid_scale_for_mem_enc = 2.0
            self.sigmoid_bias_for_mem_enc = -1.0

        if isinstance(self.vision_config, dict):
            self.vision_config = VisionEncoderConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = TrackerMaskDecoderConfig.from_dict(
                self.mask_decoder_config
            )
            # Propagation decoder: multimask tokens only (no single-mask token)
            self.mask_decoder_config.multimask_outputs_only = True
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = TrackerMaskDecoderConfig(
                multimask_outputs_only=True
            )

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = PromptEncoderConfig.from_dict(
                self.prompt_encoder_config
            )
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = PromptEncoderConfig()


@dataclass
class DetectorConfig(BaseModelConfig):
    """SAM 3.1 detector config."""

    model_type: str = "sam3.1"
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
class ModelConfig(BaseModelConfig):
    """SAM 3.1 top-level model config."""

    model_type: str = "sam3.1_video"
    detector_config: Optional[dict] = None
    tracker_config: Optional[dict] = None
    initializer_range: float = 0.02
    low_res_mask_size: int = 288

    # Tracking / association thresholds (same as SAM 3)
    det_nms_thresh: float = 0.1
    assoc_iou_thresh: float = 0.1
    trk_assoc_iou_thresh: float = 0.5
    high_conf_thresh: float = 0.8
    high_iou_thresh: float = 0.8
    new_det_thresh: float = 0.7
    score_threshold_detection: float = 0.5
    fill_hole_area: int = 16
    max_num_objects: int = 10000

    # Placeholder for mlx-vlm compatibility
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

        if self.text_config is None:
            self.text_config = self.detector_config.text_config
        if self.vision_config is None:
            self.vision_config = self.detector_config.vision_config
