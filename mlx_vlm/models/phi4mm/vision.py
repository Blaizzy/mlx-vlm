import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Special token ids for compatibility
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility

import inspect
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VisionConfig:
    model_type: str = "phi4mm"
    hidden_size: int = 1152
    num_attention_heads: int = 12
    patch_size: int = 14
    num_hidden_layers: int = 27
    intermediate_size: int = 4304
    image_size: int = 448
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    vocab_size: int = 32000
    attention_dropout: float = 0.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    img_processor: Optional[dict] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


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

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, H, W, _ = x.shape
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)

        if mask is None:
            mask = mx.ones(
                (
                    B,
                    H // self.patch_size,
                    W // self.patch_size,
                ),
                dtype=mx.bool_,
            )

        max_nb_patches_h, max_nb_patches_w = (
            H // self.patch_size,
            W // self.patch_size,
        )
        boundaries = np.linspace(
            1 / self.num_patches, 1.0, self.num_patches, endpoint=False
        )
        position_ids = np.zeros((B, max_nb_patches_h * max_nb_patches_w), dtype=int)

        for batch_idx, p_attn_mask in enumerate(mask):
            p_attn_mask = np.array(p_attn_mask)
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0, :].sum()

            fractional_coords_h = np.linspace(0, 1, nb_patches_h, endpoint=False)
            fractional_coords_w = np.linspace(0, 1, nb_patches_w, endpoint=False)

            bucket_coords_h = (
                np.digitize(fractional_coords_h, boundaries, right=True) - 1
            )
            bucket_coords_w = (
                np.digitize(fractional_coords_w, boundaries, right=True) - 1
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.reshape(-1)] = pos_ids

        embeddings = patch_embeddings
        embeddings += self.position_embedding(mx.array(position_ids))
        return embeddings


class MHA(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.in_proj = nn.Linear(dims, dims * 3, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries: mx.array, kv: mx.array, mask=None):
        B, L, D = queries.shape

        qkv = self.in_proj(queries)
        _, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class SigLipMultiheadAttentionPoolingHead(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.probe = mx.ones(
            (
                1,
                1,
                config.hidden_size,
            )
        )
        self.attention = MHA(
            config.hidden_size, num_heads=config.num_attention_heads, bias=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        x = self.attention(self.probe, x)[0]

        residual = x
        x = self.layernorm(x)
        x = residual + self.mlp(x)

        return x[:, 0]


class ReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def __call__(self, x):
        return mx.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding[0], self.padding[1]),
                (self.padding[0], self.padding[1]),
            ),
        )


class SigLIPVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model", "phi4mm"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def __call__(
        self,
        x: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.embeddings(x, mask=patch_attention_mask)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )
        pooler_output = self.post_layernorm(encoder_outputs[0])
        pooler_output = self.head(pooler_output)
        return pooler_output, x, encoder_outputs


class VisionModel(nn.Module):
    """Image embedding."""

    def __init__(self, config: VisionConfig = None, **kwargs):
        super().__init__()

        if config is None:
            config = VisionConfig()

        # Get hidden size
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        # Dropout
        self.drop = None
        if hasattr(config, "embd_pdrop") or hasattr(config, "embed_pdrop"):
            embd_drop = (
                config.embd_pdrop
                if hasattr(config, "embd_pdrop")
                else config.embed_pdrop
            )
            self.embd_drop = embd_drop  # Store as attribute, applied in forward

        # Load SigLIP model
        self.img_processor = SigLIPVisionModel(config)

        # Get positional embedding shape
        per_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = per_weight.shape
        H = int(math.sqrt(L))
        assert H**2 == L

        if H % 2 != 0:
            self.img_processor_padding = ReflectionPad2d((0, 1, 0, 1))
            H += 1

        image_dim_out = D
        self.num_img_tokens = (H // 2) ** 2
        self.base_feat_height_target = H

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        # HD transform settings
        self.use_hd_transform = kwargs.get("use_hd_transform", False)
        self.with_learnable_separator = kwargs.get("with_learnable_separator", False)
        self.hd_transform_order = kwargs.get("hd_transform_order", "glb_sub")
        self.freeze_img_processor = kwargs.get("freeze_img_processor", False)
        self.crop_size = kwargs.get("crop_size", 336)

        # Image token compression
        self.image_token_compression_cls = kwargs.get(
            "image_token_compression_cls", None
        )
        if self.image_token_compression_cls == "avg_pool_2d":
            # Simple avg pooling, not a full implementation
            self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
            self.base_feat_height_reduction = 1
            self.base_feat_height_target = self.base_feat_height_target // 2
        elif self.image_token_compression_cls is None:
            self.image_token_compression = None
            self.base_feat_height_reduction = 2
        else:
            raise NotImplementedError(
                f"image_token_compression_cls = {self.image_token_compression_cls}"
            )

        # Validate HD transform and learnable separator settings
        assert (
            self.use_hd_transform == self.with_learnable_separator
        ), "use_hd_transform and with_learnable_separator should have same value"

        if self.with_learnable_separator:
            assert self.use_hd_transform, "learnable separator is only for hd transform"
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = mx.zeros(
                (1, 1, self.image_dim_out * self.base_feat_height_reduction**2)
            )
            self.sub_GN = mx.zeros(
                (1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2)
            )

        # Projection layer
        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.img_projection = nn.Linear(image_dim_out, hidden_size)
        elif projection_cls == "mlp" and self.use_hd_transform:
            dim_projection = 3072
            depth = 2
            layers = [
                nn.Linear(
                    image_dim_out * self.base_feat_height_reduction**2, dim_projection
                )
            ]
            for _ in range(1, depth):
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            self.img_projection = layers
        elif projection_cls == "mlp":
            # Follow llava-v1.5's implementation
            dim_projection = 3072
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            self.img_projection = layers
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}")

        self.vocab_size = config.vocab_size
        self.img_features = None

        # Layer settings
        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get("layer_idx", -2)
            self.type_feature = config.img_processor.get("type_feature", "patch")
        else:
            self.layer_idx = -2
            self.type_feature = "patch"

    def set_img_features(self, img_features):
        self.img_features = img_features

    def set_img_sizes(self, img_sizes):
        self.img_sizes = img_sizes

    def set_img_attn_mask(self, image_attention_mask):
        self.image_attention_mask = image_attention_mask

    def get_img_features(self, img_embeds, attention_mask=None):
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        # Process the image through the image processor
        if self.freeze_img_processor:
            if attention_mask is not None:
                img_processor_output = self.img_processor(
                    img_embeds,
                    patch_attention_mask=attention_mask,
                    output_hidden_states=True,
                    # patch_attention_mask=attention_mask,
                )
            else:
                img_processor_output = self.img_processor(
                    img_embeds, output_hidden_states=True
                )
        else:
            if attention_mask is not None:
                img_processor_output = self.img_processor(
                    img_embeds,
                    patch_attention_mask=attention_mask,
                    output_hidden_states=True,
                    # patch_attention_mask=attention_mask,
                )
            else:
                img_processor_output = self.img_processor(
                    img_embeds, output_hidden_states=True
                )

        img_feature = img_processor_output[-1][LAYER_IDX]

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature

            if self.image_token_compression is not None:
                # Reshape to 2D tensor for pooling
                width = int(math.sqrt(patch_feature.shape[1]))
                patch_feature = mx.reshape(
                    patch_feature, (-1, width, width, patch_feature.shape[-1])
                )

                # Convert to NCHW
                patch_feature = mx.transpose(patch_feature, (0, 3, 1, 2))

                # Apply padding if needed
                if getattr(self, "img_processor_padding", None) is not None:
                    # Simplified padding with zeros
                    patch_feature = mx.pad(
                        patch_feature, ((0, 0), (0, 0), (0, 1), (0, 1))
                    )

                # Apply pooling (simple average pooling as placeholder)
                patch_feature = mx.mean(
                    mx.reshape(
                        patch_feature,
                        (
                            patch_feature.shape[0],
                            patch_feature.shape[1],
                            patch_feature.shape[2] // 2,
                            2,
                            patch_feature.shape[3] // 2,
                            2,
                        ),
                    ),
                    axis=(3, 5),
                )

                # Convert back to NHWC
                patch_feature = mx.transpose(patch_feature, (0, 2, 3, 1))

                # Reshape back to sequence
                patch_feature = mx.reshape(
                    patch_feature,
                    (
                        -1,
                        patch_feature.shape[1] * patch_feature.shape[2],
                        patch_feature.shape[3],
                    ),
                )

            elif getattr(self, "img_processor_padding", None) is not None:
                # Handle padding without compression
                width = int(math.sqrt(patch_feature.shape[1]))
                patch_feature = mx.reshape(
                    patch_feature, (-1, width, width, patch_feature.shape[-1])
                )

                # Convert to NCHW
                patch_feature = mx.transpose(patch_feature, (0, 3, 1, 2))

                # Apply padding
                patch_feature = mx.pad(patch_feature, ((0, 0), (0, 0), (0, 1), (0, 1)))

                # Convert back to NHWC
                patch_feature = mx.transpose(patch_feature, (0, 2, 3, 1))

                # Reshape back to sequence
                patch_feature = mx.reshape(
                    patch_feature,
                    (
                        -1,
                        patch_feature.shape[1] * patch_feature.shape[2],
                        patch_feature.shape[3],
                    ),
                )

            return patch_feature

        elif TYPE_FEATURE == "cls_patch":
            if self.image_token_compression is not None:
                # Extract cls token and patches
                patch_feature = img_feature[:, 1:]
                cls_feature = img_feature[:, 0:1]

                # Reshape patches for compression
                width = int(math.sqrt(patch_feature.shape[1]))
                patch_feature = mx.reshape(
                    patch_feature, (-1, width, width, patch_feature.shape[-1])
                )

                # Apply pooling (simplified)
                patch_feature = mx.reshape(
                    patch_feature,
                    (
                        -1,
                        patch_feature.shape[1] // 2,
                        2,
                        patch_feature.shape[2] // 2,
                        2,
                        patch_feature.shape[3],
                    ),
                )
                patch_feature = mx.mean(patch_feature, axis=(2, 4))

                # Flatten back to sequence
                patch_feature = mx.reshape(
                    patch_feature,
                    (
                        -1,
                        patch_feature.shape[1] * patch_feature.shape[2],
                        patch_feature.shape[3],
                    ),
                )

                # Combine with cls token
                img_feature = mx.concatenate([cls_feature, patch_feature], axis=1)

            return img_feature

        # Fallback - shouldn't reach here with proper configuration
        raise NotImplementedError(f"Feature type {TYPE_FEATURE} not implemented")

    def __call__(self, input_ids, input_embeds, image_sizes=None, **kwargs):
        if isinstance(input_ids, tuple):
            # Handle pipeline parallel case
            input_ids, input_embeds = input_ids

        img_embeds = input_embeds

        # Get image sizes
        if image_sizes is None and "image_sizes" in kwargs:
            image_sizes = kwargs["image_sizes"]
        img_sizes = image_sizes

        # Handle cached image features
        if self.img_features is not None:
            img_embeds = self.img_features
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        # Handle attention mask
        if self.image_attention_mask is not None:
            image_attention_mask = self.image_attention_mask
            self.image_attention_mask = None
        elif "image_attention_mask" in kwargs:
            image_attention_mask = kwargs["image_attention_mask"]
        else:
            image_attention_mask = None

        # Reshape input_ids if needed
        input_shape = input_ids.shape
        input_ids = mx.reshape(input_ids, (-1, input_shape[-1]))

        # Find positions of image tokens
        positions = mx.array(np.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID)[0])

        # Default values for fake image forward and selection
        fake_image_forward = False
        select = False
        hd_transform = False

        # Get target device and dtype from projection layer
        if isinstance(self.img_projection, list):  # nn.Sequential in MLX
            target_dtype = mx.float32  # Default dtype
        else:  # Single linear layer
            target_dtype = mx.float32

        num_img_tokens = self.num_img_tokens
        if len(positions) > 0:
            if self.use_hd_transform and img_sizes is not None and len(img_sizes) > 0:
                hd_transform = True
                assert (
                    img_embeds.ndim == 5
                ), f"img_embeds should be 5D for hd transform, got {img_embeds.ndim}D"

                # img_embeds: (num_images, max_num_crops, 3, H, W)
                bs = img_embeds.shape[0]

                # Process image features
                if image_attention_mask is not None and len(image_attention_mask) > 0:
                    img_features = self.get_img_features(
                        mx.reshape(img_embeds, (-1,) + img_embeds.shape[2:]),
                        attention_mask=mx.reshape(
                            image_attention_mask.astype(mx.bool_),
                            (-1,) + image_attention_mask.shape[2:],
                        ),
                    )
                else:
                    img_features = self.get_img_features(
                        mx.reshape(img_embeds, (-1,) + img_embeds.shape[2:])
                    )

                # HD transform parameters
                base_feat_height_target = self.base_feat_height_target
                base_resolution = self.crop_size
                base_feat_height_reduction = self.base_feat_height_reduction

                # Check feature dimensions
                base_feat_height = base_feat_width = int(
                    math.sqrt(img_features.shape[1])
                )
                assert (
                    base_feat_height == base_feat_height_target
                    and base_feat_width == base_feat_height_target
                ), f"Feature height/width should be {base_feat_height_target}, got {base_feat_height}"

                # Reshape features for HD transform
                # bs x max_num_crops x (HxW) x C
                img_features = mx.reshape(
                    img_features,
                    (bs, -1, base_feat_height * base_feat_width, self.image_dim_out),
                )
                C = self.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []

                # Process each image in batch
                for _bs in range(bs):
                    # Get image dimensions
                    if isinstance(img_sizes, mx.array):
                        h, w = img_sizes[_bs]
                    else:
                        h, w = img_sizes[_bs]

                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = (h * w).tolist()

                    # Process global image feature (first crop)
                    global_img_feature = img_features[_bs, :1]

                    # HD transform for global feature
                    # Reshape: 1 x (HxW) x C -> 1 x H x W x C
                    glb_img = mx.reshape(global_img_feature, (1, H, H, C))

                    # Further reshape for block-wise processing
                    # 1 x H x W x C -> 1 x (H/R) x R x (W/R) x R x C -> 1 x (H/R) x (W/R) x (RxRxC)
                    glb_img = mx.reshape(
                        glb_img,
                        (
                            1,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        ),
                    )
                    glb_img = mx.transpose(glb_img, (0, 1, 3, 2, 4, 5))
                    glb_img = mx.reshape(
                        glb_img,
                        (
                            1,
                            H // base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )

                    # Create separator
                    temp_glb_GN = mx.repeat(
                        self.sub_GN, H // base_feat_height_reduction, axis=1
                    )

                    # Add separator
                    glb_img = mx.concatenate([glb_img, temp_glb_GN], axis=2)
                    glb_img = mx.reshape(
                        glb_img,
                        (
                            1,
                            -1,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )

                    # Process sub-image features (remaining crops)
                    sub_img = img_features[_bs, 1:]

                    # Only use necessary crops (discard padding)
                    sub_img = sub_img[:B_]

                    # HD transform for sub images
                    # Reshape: B_ x (HxW) x C -> B_ x H x W x C
                    sub_img = mx.reshape(sub_img, (B_, H, H, C))

                    # Further reshape for block-wise processing
                    sub_img = mx.reshape(
                        sub_img,
                        (
                            B_,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        ),
                    )
                    sub_img = mx.transpose(sub_img, (0, 1, 3, 2, 4, 5))
                    sub_img = mx.reshape(
                        sub_img,
                        (
                            B_,
                            -1,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )

                    # Reshape to spatial layout
                    sub_img = mx.reshape(
                        sub_img,
                        (
                            1,
                            h,
                            w,
                            H // base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )
                    sub_img = mx.transpose(sub_img, (0, 1, 3, 2, 4, 5))
                    sub_img = mx.reshape(
                        sub_img,
                        (
                            1,
                            h * H // base_feat_height_reduction,
                            w * H // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )

                    # Handle attention mask for useful regions
                    if (
                        image_attention_mask is not None
                        and len(image_attention_mask) > 0
                    ):
                        # Extract attention mask for this image, reshape to match spatial layout
                        reshaped_mask = image_attention_mask[
                            _bs, 1 : B_ + 1, 0::2, 0::2
                        ]
                        reshaped_mask = mx.reshape(
                            reshaped_mask,
                            (
                                1,
                                h,
                                w,
                                H // base_feat_height_reduction,
                                H // base_feat_height_reduction,
                            ),
                        )
                        reshaped_mask = mx.transpose(reshaped_mask, (0, 1, 3, 2, 4))
                        reshaped_mask = mx.reshape(
                            reshaped_mask,
                            (
                                1,
                                h * H // base_feat_height_reduction,
                                w * H // base_feat_height_reduction,
                            ),
                        )

                        # Calculate useful dimensions
                        useful_height = int(mx.sum(reshaped_mask[0, :, 0]).item())
                        useful_width = int(mx.sum(reshaped_mask[0, 0, :]).item())

                        # Crop to useful region
                        sub_img = sub_img[:, :useful_height, :useful_width]

                        # Create separator of appropriate size
                        temp_sub_GN = mx.repeat(self.sub_GN, useful_height, axis=1)

                        # Calculate output length
                        mask_sum = int(
                            mx.sum(
                                image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
                            ).item()
                        )
                        temp_len = (
                            mask_sum
                            + (useful_height + 1)
                            + H // base_feat_height_reduction
                        )
                    else:
                        # No mask, use full feature map
                        temp_sub_GN = mx.repeat(
                            self.sub_GN, h * H // base_feat_height_reduction, axis=1
                        )
                        temp_len = int(
                            (h * w + 1) * self.num_img_tokens
                            + 1
                            + (h + 1) * H // base_feat_height_reduction
                        )

                    # Add separator
                    sub_img = mx.concatenate([sub_img, temp_sub_GN], axis=2)
                    sub_img = mx.reshape(
                        sub_img,
                        (
                            1,
                            -1,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        ),
                    )

                    # Combine global and sub features based on transform order
                    if self.hd_transform_order == "glb_sub":
                        output_imgs.append(
                            mx.concatenate([glb_img, self.glb_GN, sub_img], axis=1)
                        )
                    elif self.hd_transform_order == "sub_glb":
                        output_imgs.append(
                            mx.concatenate([sub_img, self.glb_GN, glb_img], axis=1)
                        )
                    else:
                        raise NotImplementedError(
                            f"hd_transform_order = {self.hd_transform_order}"
                        )

                    # Verify length
                    assert (
                        temp_len == output_imgs[-1].shape[1]
                    ), f"Expected length {temp_len}, got {output_imgs[-1].shape[1]}"
                    output_len.append(temp_len)

                # Set number of image tokens
                num_img_tokens = output_len

                # Project image features
                img_set_tensor = []
                for _output_img in output_imgs:
                    h = _output_img
                    for img_proj in self.img_projection:
                        h = img_proj(h)
                    img_set_tensor.append(h)

                select = True
            else:
                raise NotImplementedError("Only HD transform is implemented")
            select = True
        else:
            # Create a fake image tensor for training
            if True:  # Equivalent to self.training in PyTorch
                # Create a dummy image embedding
                img_embeds = mx.zeros(
                    (1, 3, self.crop_size, self.crop_size), dtype=target_dtype
                )

                # Process the dummy embedding
                tt = mx.reshape(self.get_img_features(img_embeds), (-1, 1024))

                # Apply projection
                if self.use_hd_transform:
                    img_set_tensor = self.img_projection(
                        mx.reshape(
                            tt,
                            (
                                -1,
                                self.image_dim_out * self.base_feat_height_reduction**2,
                            ),
                        )
                        * self.glb_GN[0]
                        * self.sub_GN[0, 0]
                    )
                else:
                    img_set_tensor = self.img_projection(tt)

                fake_image_forward = True

        # Get token embeddings from word embedding table
        hidden_states = kwargs["wte"](input_ids)

        if select:
            if hd_transform:
                # Create a new hidden states tensor with image embeddings inserted
                # This is a more functional approach compared to PyTorch's in-place operations

                # Find positions of image tokens in the input sequence
                pos_indices = np.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID)
                batch_indices, seq_indices = pos_indices

                # Combine all image embeddings into a single tensor
                merged_img_tensor = mx.concatenate(
                    [img.reshape(-1, img.shape[-1]) for img in img_set_tensor]
                )

                # Initialize new hidden states with the original values
                new_hidden_states = hidden_states.tolist()

                # Update hidden states with image embeddings
                for i in range(len(batch_indices)):
                    b_idx, s_idx = int(batch_indices[i]), int(seq_indices[i])
                    img_embedding = merged_img_tensor[i].tolist()
                    # Update the Python list directly
                    new_hidden_states[b_idx][s_idx] = img_embedding

                # Convert back to mx.array
                new_hidden_states = mx.array(
                    new_hidden_states, dtype=hidden_states.dtype
                )

                hidden_states = new_hidden_states
            else:
                raise NotImplementedError("Only HD transform is implemented")

        # Add the fake image contribution for training stability
        if fake_image_forward and True:  # Equivalent to self.training
            # Adding a small contribution that will be zero but maintains gradient flow
            hidden_states = hidden_states + (0 * img_set_tensor[0]).sum()

        # Apply dropout if needed
        if self.drop is not None and True:  # Equivalent to self.training
            hidden_states = nn.Dropout(self.embd_drop)(hidden_states)

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if "model.embed_tokens_extend.image_embed.img_processor.head" in k:
                if "attention.in_proj_weight" in k:
                    new_k = k.replace(
                        "attention.in_proj_weight", "attention.in_proj.weight"
                    )
                    sanitized_weights[new_k] = v
                elif "attention.in_proj_bias" in k:
                    new_k = k.replace(
                        "attention.in_proj_bias", "attention.in_proj.bias"
                    )
                    sanitized_weights[new_k] = v

                else:
                    sanitized_weights[k] = v
            elif "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
