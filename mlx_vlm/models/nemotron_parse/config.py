import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration class for Nemotron-Parse C-RADIO vision encoder."""

    model_type: str = "nemotron_parse_vision"
    hidden_size: int = 1280
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_layers: int = 32
    patch_size: int = 16
    image_size: Tuple[int, int] = (2048, 1664)
    num_cls_tokens: int = 4
    num_register_tokens: int = 4
    summary_idxs: List[int] = field(default_factory=lambda: [0, 1, 2])
    neck_dim: int = 1024
    final_size: Tuple[int, int] = (2048, 1664)
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration class for Nemotron-Parse mBART decoder."""

    model_type: str = "nemotron_parse"
    d_model: int = 1024
    decoder_attention_heads: int = 16
    decoder_ffn_dim: int = 4096
    decoder_layers: int = 10
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    init_std: float = 0.02
    decoder_layerdrop: float = 0.0
    scale_embedding: bool = True
    use_cache: bool = True
    max_position_embeddings: int = 9000
    vocab_size: int = 72256
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration class for Nemotron-Parse."""

    vision_config: VisionConfig
    text_config: TextConfig
    model_type: str = "nemotron_parse"
    vocab_size: int = 72256
    max_position_embeddings: int = 9000
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
    image_size: Tuple[int, int] = (2048, 1664)
    max_sequence_length: int = 9000
    tie_word_embeddings: bool = True
    class_token_start_idx: int = 52315

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls(
                vision_config=VisionConfig(),
                text_config=TextConfig(),
            )

        encoder_params = params.get("encoder", {})
        decoder_params = params.get("decoder", {})

        vision_config = VisionConfig.from_dict(encoder_params)
        text_config = TextConfig.from_dict(decoder_params)

        # Top-level decoder_start_token_id overrides the decoder sub-config.
        if (
            "decoder_start_token_id" in params
            and params["decoder_start_token_id"] is not None
        ):
            text_config.decoder_start_token_id = params["decoder_start_token_id"]

        # Promote image-level fields to vision config if present.
        if "image_size" in params:
            vision_config.image_size = tuple(params["image_size"])
        if "max_sequence_length" in params:
            params = dict(params)
            params.setdefault("max_position_embeddings", params["max_sequence_length"])

        model_params = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        model_params["vision_config"] = vision_config
        model_params["text_config"] = text_config

        return cls(**model_params)
