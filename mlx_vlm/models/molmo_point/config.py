import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "molmo2"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: Tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 729

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class AdapterConfig(BaseModelConfig):
    model_type: str = "molmo_point"
    vit_layers: Tuple[int, ...] = (-3, -9)
    pooling_attention_mask: bool = False
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "silu"
    intermediate_size: int = 12288
    text_hidden_size: int = 4096
    positional_embeddings: Optional[int] = None


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "molmo2_text"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    qkv_bias: bool = False
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    hidden_act: str = "silu"
    max_position_embeddings: int = 37376
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict] = None
    rope_scaling_layers: Optional[List[int]] = None
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    tie_word_embeddings: bool = False


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "molmo_point"
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    adapter_config: AdapterConfig = None

    # Token IDs
    image_start_token_id: int = 151936
    image_end_token_id: int = 151937
    image_patch_id: int = 151938
    image_high_res_id: int = 151938
    image_col_id: int = 151939
    image_non_indexable_patch_id: int = 151942
    frame_start_token_id: int = 151943
    frame_end_token_id: int = 151944
    patch_token_id: int = 151947
    subpatch_token_id: int = 151948
    location_token_id: int = 151949
    use_frame_special_tokens: bool = True

    # Point prediction config
    patch_location: Optional[str] = "3x3"
    no_more_points_class: bool = True
    patch_embed_dim: int = 512
    patch_embedding_kind: str = "image_feature0"
    embed_selected_vit_patch: Optional[str] = "linear"
    embed_location: bool = False
    layer_norm_x: bool = True
    norm_logits: bool = True
    mask_patches: Optional[str] = "always"
    mask_subpatches: str = "inference"
    mask_repeats: Optional[str] = "inference"
    token_prediction_rotary: str = "one_d"
    token_prediction_rotary_theta: Optional[float] = 50000.0

    @classmethod
    def from_dict(cls, params):
        updated_params = {}
        updated_params.update(
            {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        )

        if "text_config" in params:
            updated_params["text_config"] = TextConfig.from_dict(params["text_config"])
        if "vit_config" in params:
            updated_params["vision_config"] = VisionConfig.from_dict(
                params["vit_config"]
            )
        elif "vision_config" in params:
            updated_params["vision_config"] = VisionConfig.from_dict(
                params["vision_config"]
            )
        if "adapter_config" in params:
            updated_params["adapter_config"] = AdapterConfig.from_dict(
                params["adapter_config"]
            )
        return cls(**updated_params)

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def num_attention_heads(self):
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self):
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.text_config.head_dim

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def eos_token_id(self):
        return None  # Let the tokenizer handle EOS
