from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..attention import VisionAttention as Attention
from ..kernels import bicubic_interpolate
from ..mlp import GELUMLP as MLP
from .config import VisionConfig


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)
            h = x

        return encoder_states


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            input_dims=config.num_channels * self.patch_size * self.patch_size,
            output_dims=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: mx.array,
        spatial_shapes: mx.array,
        max_length: int,
    ) -> mx.array:
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = mx.zeros(
            (batch_size, max_length, embed_dim),
            dtype=source_dtype,
        )

        positional_embeddings = positional_embeddings.transpose(2, 0, 1)[None, :]
        for i in range(batch_size):
            height, width = spatial_shapes[i].tolist()

            resized_embeddings = bicubic_interpolate(
                positional_embeddings,
                size=(height, width),
            )

            resized_embeddings = resized_embeddings.reshape(
                embed_dim, height * width
            ).transpose(1, 0)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def __call__(
        self, pixel_values: mx.array, spatial_shapes: mx.array = None
    ) -> mx.array:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class SigLip2VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embeddings(x, spatial_shapes=spatial_shapes)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )
        last_hidden_state = self.post_layernorm(encoder_outputs[-1])
        return encoder_outputs, x, last_hidden_state


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["phi4-siglip", "siglip2_vision_model"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SigLip2VisionModel(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        spatial_shapes: Optional[mx.array] = None,
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states, spatial_shapes)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights
