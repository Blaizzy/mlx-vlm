from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    """Stock Qwen3-4B-Instruct-2507. Field-for-field identical to `models/qwen3/config.py`.

    Kept as its own dataclass only because Mage-VL nests it under `text_config` and stores rope
    under transformers-5.x `rope_parameters` rather than a flat `rope_theta` (see `from_dict`).
    """

    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        params = dict(params)
        # transformers 5.x moved rope out to a nested block; Mage-VL's config.json uses it.
        # Flatten it before the base filter drops it on the floor and we silently inherit the
        # 5e6 default — a wrong rope_theta is exactly the kind of thing that parity-passes at
        # short sequence lengths and diverges at long ones.
        rope = params.pop("rope_parameters", None)
        if isinstance(rope, dict):
            if "rope_theta" in rope:
                params["rope_theta"] = rope["rope_theta"]
            rope_type = rope.get("rope_type", "default")
            if rope_type not in (None, "default"):
                params["rope_scaling"] = {**rope, "type": rope_type}
        return super().from_dict(params)


@dataclass
class VisionConfig(BaseModelConfig):
    """Mage-ViT — a 24-layer pre-norm ViT with a 4:6:6 split 3D RoPE."""

    model_type: str = "mage_vl_vision"
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 16
    image_size: int = 448
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    layer_norm_type: str = "layer_norm"
    hidden_act: str = "gelu"
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    out_hidden_size: int = 2560
    text_hidden_size: int = 2560
    # Temporal attention window: the encoder attends block-diagonally over groups of this many
    # frames. For a single image (t=1) this collapses to one block, i.e. plain full attention.
    frame_windows_size: int = 4
    # False in the VLM (the Siglip2 pooling head exists only in the standalone Mage-ViT).
    use_head: bool = False
    max_position_embeddings: int = 8192
    use_patch_position_encoding: bool = False
    patch_position_encoding_type: str = "absolute"
    attention_dropout: float = 0.0
    tokens_per_second: int = 1


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "mage_vl"
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    eos_token_id: Optional[list] = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        params = dict(params)
        params["text_config"] = TextConfig.from_dict(params.get("text_config", {}))
        params["vision_config"] = VisionConfig.from_dict(
            params.get("vision_config", {})
        )
        return super().from_dict(params)
