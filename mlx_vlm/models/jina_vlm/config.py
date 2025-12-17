from dataclasses import dataclass, field
from typing import Tuple

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Vision encoder configuration for Jina VLM."""

    model_type: str = "jina_vlm"
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    head_dim: int = 72
    patch_size: int = 14
    image_size: int = 378
    num_channels: int = 3
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    use_bias: bool = True
    use_cls_token: bool = False
    post_layer_norm: bool = True
    activation: str = "gelu_pytorch_tanh"
    vit_layers: Tuple[int, ...] = (-4, -10)
    output_size: int = 2048
    # Connector config
    pooling_h: int = 2
    pooling_w: int = 2
    connector_hidden_size: int = 6144


@dataclass
class TextConfig(BaseModelConfig):
    """Text decoder configuration for Jina VLM."""

    model_type: str = "jina_vlm"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False


@dataclass
class ModelConfig(BaseModelConfig):
    """Full Jina VLM configuration."""

    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "jina_vlm"
    vocab_size: int = 151936
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    pad_token_id: int = 151643
    image_token_index: int = 151940  # <|image|>
    image_token_id: int = 151940  # <|image|>
    image_start_token_id: int = 151666  # <im_start>
    image_end_token_id: int = 151667  # <im_end>
    image_patch_token_id: int = 151665  # <im_patch>
    image_column_token_id: int = 151668  # <im_col>
    ignore_index: int = -100
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # Parse vision config
        vision_cfg = params.get("vision_config", {})
        vision_block = vision_cfg.get("block_config", {})
        vision_attn = vision_block.get("attn_config", {})
        vision_ffn = vision_block.get("ffn_config", {})
        vl_connector = vision_cfg.get("vl_connector_config", {})
        connector_mlp = vl_connector.get("mlp_projector_config", {})

        vision_config = VisionConfig(
            hidden_size=vision_cfg.get("hidden_size", 1152),
            num_hidden_layers=vision_cfg.get("n_layers", 27),
            num_attention_heads=vision_attn.get("n_heads", 16),
            head_dim=vision_attn.get("head_dim", 72),
            patch_size=vision_cfg.get("patch_size", 14),
            image_size=(
                vision_cfg.get("input_size", [378, 378])[0]
                if isinstance(vision_cfg.get("input_size"), list)
                else 378
            ),
            num_channels=vision_cfg.get("n_channels", 3),
            intermediate_size=vision_ffn.get("size", 4304),
            use_bias=vision_attn.get("q_bias", True),
            use_cls_token=vision_cfg.get("use_cls_token", False),
            post_layer_norm=vision_cfg.get("post_lnorm", True),
            activation=vision_ffn.get("activation_type", "gelu_pytorch_tanh"),
            vit_layers=tuple(vision_cfg.get("vit_layers", [-4, -10])),
            output_size=vision_cfg.get("output_size", 2048),
            pooling_h=vl_connector.get("pooling_h", 2),
            pooling_w=vl_connector.get("pooling_w", 2),
            connector_hidden_size=connector_mlp.get("size", 6144),
        )

        # Parse text config
        text_cfg = params.get("text_config", {})
        text_block = text_cfg.get("block_config", {})
        text_attn = text_block.get("attn_config", {})
        text_ffn = text_block.get("ffn_config", {})
        text_lnorm = text_block.get("lnorm_config", {})

        text_config = TextConfig(
            hidden_size=text_cfg.get("hidden_size", 2048),
            num_hidden_layers=text_cfg.get(
                "n_layers", text_cfg.get("num_hidden_layers", 28)
            ),
            num_attention_heads=text_attn.get("n_heads", 16),
            num_key_value_heads=text_attn.get("n_kv_heads", 8),
            head_dim=text_attn.get("head_dim", 128),
            vocab_size=text_cfg.get("vocab_size", 151936),
            additional_vocab_size=text_cfg.get("additional_vocab_size", 128),
            intermediate_size=text_ffn.get("size", 6144),
            rms_norm_eps=text_lnorm.get("eps", 1e-6),
            rope_theta=text_cfg.get("rope_theta", 1000000.0),
            max_position_embeddings=text_cfg.get("max_sequence_length", 40960),
            use_qk_norm=text_attn.get("q_lnorm", True),
            tie_word_embeddings=text_cfg.get("tie_word_embeddings", False),
        )

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            model_type=params.get("model_type", "jina_vlm"),
            vocab_size=params.get("vocab_size", text_config.vocab_size),
            bos_token_id=params.get("bos_token_id", 151643),
            eos_token_id=params.get("eos_token_id", 151643),
            pad_token_id=params.get("pad_token_id", 151643),
            image_token_index=params.get("image_token_index", 151940),
            tie_word_embeddings=params.get("tie_word_embeddings", False),
        )
