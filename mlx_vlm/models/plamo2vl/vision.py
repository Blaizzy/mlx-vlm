import inspect
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..llava.vision import ClipVisionModel, VisionConfig as SiglipVisionConfig


@dataclass
class VisionConfig:
    image_feature_size: int = 1152
    image_proj_hidden_size: int = 4096
    image_token_id: int = 100002
    model_type: str = "siglip_vision_model"
    vision_encoder_attention_dropout: float = 0.0
    vision_encoder_hidden_act: str = "gelu_pytorch_tanh"
    vision_encoder_hidden_size: int = 1152
    vision_encoder_image_size: int = 384
    vision_encoder_intermediate_size: int = 4304
    vision_encoder_layer_norm_eps: float = 1e-6
    vision_encoder_num_attention_heads: int = 16
    vision_encoder_num_channels: int = 3
    vision_encoder_num_hidden_layers: int = 27
    vision_encoder_patch_size: int = 14

    @classmethod
    def from_dict(cls, params):
        values = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        values["model_type"] = values.get("model_type") or "siglip_vision_model"
        return cls(**values)


class VisionEncoderWrapper(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        encoder_config = SiglipVisionConfig(
            model_type="siglip_vision_model",
            hidden_size=config.vision_encoder_hidden_size,
            num_attention_heads=config.vision_encoder_num_attention_heads,
            patch_size=config.vision_encoder_patch_size,
            num_hidden_layers=config.vision_encoder_num_hidden_layers,
            intermediate_size=config.vision_encoder_intermediate_size,
            image_size=config.vision_encoder_image_size,
            num_channels=config.vision_encoder_num_channels,
            layer_norm_eps=config.vision_encoder_layer_norm_eps,
            hidden_act=config.vision_encoder_hidden_act,
        )
        self.model = ClipVisionModel(encoder_config)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        pixel_values = pixel_values.transpose(0, 2, 3, 1)
        _, hidden_states, _ = self.model(pixel_values, output_hidden_states=False)
        hidden_states = self.model.post_layernorm(hidden_states)
        return hidden_states.reshape(-1, hidden_states.shape[-1])


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoderWrapper(config)

    def __call__(self, input_ids: mx.array, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim == 5:
            if pixel_values.shape[0] != 1:
                raise ValueError("Only batch size 1 is supported for tiled images")
            pixel_values = pixel_values.reshape((-1,) + tuple(pixel_values.shape[2:]))
        elif pixel_values.ndim != 4:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        image_features = self.vision_encoder(pixel_values)

        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")

        image_mask = input_ids == self.config.image_token_id
        num_image_tokens = int(mx.sum(image_mask).item())
        if image_features.shape[0] != num_image_tokens:
            raise ValueError(
                "Image features and image token count do not match: "
                f"{image_features.shape[0]} != {num_image_tokens}"
            )

        positions = mx.cumsum(image_mask.astype(mx.int32), axis=1) - 1
        positions = mx.maximum(positions, 0)
        expanded = mx.take(image_features, positions[0], axis=0)[None, :, :]
        mask = mx.expand_dims(image_mask, -1)
        zeros = mx.zeros(expanded.shape, dtype=expanded.dtype)
        return mx.where(mask, expanded, zeros)
