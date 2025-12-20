import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VitConfig(BaseModelConfig):
    model_type: str = "molmo2"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 25  # Note: HF config says 27 but weights only have 25
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    image_patch_size: int = 14
    image_num_pos: int = 729
    image_default_input_size: List[int] = field(default_factory=lambda: [378, 378])
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    float32_attention: bool = True
    attn_implementation: str = "sdpa"

    @classmethod
    def from_dict(cls, params):
        # Workaround: HuggingFace config says 27 layers but weights only have 25
        # Override to use 25 layers
        if params.get("num_hidden_layers", 25) > 25:
            params = dict(params)  # Don't modify original
            params["num_hidden_layers"] = 25
        return super().from_dict(params)

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class AdapterConfig(BaseModelConfig):
    model_type: str = "molmo2"
    hidden_size: int = 1152
    intermediate_size: int = 9728
    text_hidden_size: int = 2560
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "silu"
    vit_layers: List[int] = field(default_factory=lambda: [-3, -9])
    image_feature_dropout: float = 0.0
    pooling_attention_mask: bool = True
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    float32_attention: bool = True
    attn_implementation: str = "sdpa"


@dataclass
class VisionConfig(BaseModelConfig):
    vit_config: VitConfig = field(default_factory=VitConfig)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)

    @classmethod
    def from_dict(cls, params):
        vit_cfg = params.get("vit_config", {})
        adapter_cfg = params.get("adapter_config", {})
        return cls(
            vit_config=VitConfig.from_dict(vit_cfg),
            adapter_config=AdapterConfig.from_dict(adapter_cfg),
        )


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "molmo2"
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    max_position_embeddings: int = 36864
    rope_theta: float = 5000000.0
    rope_scaling: Optional[dict] = None
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"
    qkv_bias: bool = False
    use_cache: bool = True
    norm_after: bool = False


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "molmo2"

    image_start_token_id: int = 151936
    low_res_image_start_token_id: int = 151940
    image_end_token_id: int = 151937
    image_low_res_id: int = 151942
    image_patch_id: int = 151938
    image_col_id: int = 151939
    frame_start_token_id: int = 151943
    frame_end_token_id: int = 151944
    use_frame_special_tokens: bool = False

    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Normalize how the repo loads configs: always provide `vision_config`.
        if not params.get("vision_config"):
            params["vision_config"] = {
                "vit_config": params.get("vit_config", {}),
                "adapter_config": params.get("adapter_config", {}),
            }

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
