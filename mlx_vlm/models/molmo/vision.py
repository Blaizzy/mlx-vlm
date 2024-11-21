import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    model_type: str = "molmo"
    num_channels: int = 3
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    hidden_size: int = 18944
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 23
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "gelu"
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    d_model: int = 3584
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    vit_layers: Optional[List[int]] = field(default_factory=lambda: [-2, -9])
    image_pooling_2d: str = "attention-meanq"
    image_padding_embed: str = "pad_and_partial_pad"
    intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.image_patch_size * self.image_patch_size * 3

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size

    @property
    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class MLP(nn.Module):
    def __init__(self, config: VisionConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size,
            bias=False,
        )
        self.w2 = nn.Linear(
            self.hidden_size,
            config.d_model,
            bias=False,
        )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        return x


class ViTMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.image_emb_dim, config.image_mlp_dim, bias=True)
        self.w2 = nn.Linear(config.image_mlp_dim, config.image_emb_dim, bias=True)
        self.act = nn.GELU(approx="fast")

    def __call__(self, x: mx.array) -> mx.array:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: VisionConfig, is_vit_layer: Optional[bool] = True):
        super().__init__()
        self.config = config
        self.embed_dim = config.image_emb_dim
        self.num_heads = config.image_num_heads
        self.head_dim = config.image_head_dim
        self.num_key_value_heads = config.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.is_vit_layer = is_vit_layer

        n_layers = (
            1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)
        )

        self.wq = nn.Linear(
            n_layers * self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.wk = nn.Linear(
            n_layers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            n_layers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def _split_heads(self, hidden_states, num_heads) -> mx.array:
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states) -> mx.array:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, x: mx.array, kv: mx.array = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        if kv is None:
            k = x
            v = x
        else:
            k = kv
            v = kv
        q = self._split_heads(self.wq(x), self.num_heads).transpose(0, 2, 1, 3)

        k = self._split_heads(self.wk(k), self.num_key_value_heads).transpose(
            0, 2, 1, 3
        )
        v = self._split_heads(self.wv(v), self.num_key_value_heads).transpose(
            0, 2, 1, 3
        )

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = attn.transpose(0, 2, 1, 3)
        out = self._merge_heads(out)
        out = self.wo(out)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim, eps=config.image_norm_eps
        )
        self.ffn_norm = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ResidualAttentionBlocks(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.resblocks = [
            ResidualAttentionBlock(config) for _ in range(config.image_num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = []
        for block in self.resblocks:
            x = block(x)
            h.append(x)
        return h


def _expand_token(token, batch_size: int):
    return mx.broadcast_to(
        mx.reshape(token, (1, 1, -1)), (batch_size, 1, token.shape[-1])
    )


def pad_to_multiple(x, target_size, pad_mode="edge", pad_value=0):
    """
    Pad the last dimension of input tensor to match target size.

    Args:
        x: Input tensor with shape [..., D]
        target_size: Desired size for the last dimension
        pad_mode: Padding mode ('constant', 'reflect', etc.)
        pad_value: Value to use for constant padding

    Returns:
        Padded tensor with shape [..., target_size]
    """
    current_size = x.shape[-1]

    # Return early if no padding needed
    if current_size == target_size:
        return x

    # Ensure target size is larger
    if current_size > target_size:
        raise ValueError(
            f"Current size {current_size} is larger than target size {target_size}"
        )

    # Calculate padding needed
    pad_size = target_size - current_size

    # Create padding configuration
    # No padding for batch and channel dimensions (0,0), only pad the last dim
    pad_config = [(0, 0)] * (len(x.shape) - 1) + [(0, pad_size)]

    return mx.pad(x, pad_width=pad_config, mode=pad_mode, constant_values=pad_value)


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.class_embedding = mx.zeros((config.image_emb_dim,))
        self.positional_embedding = mx.zeros(
            (config.image_num_pos, config.image_emb_dim)
        )
        self.patch_embedding = nn.Linear(
            config.intermediate_size,
            config.image_emb_dim,
            bias=False,
        )
        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)
        self.transformer = ResidualAttentionBlocks(config)

    def add_pos_emb(self, x: mx.array, patch_num: int) -> mx.array:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        # Reshape into 2D grid
        pos_emb_size = int(pos_emb.shape[0] ** 0.5)
        pos_emb = mx.reshape(pos_emb, (pos_emb_size, pos_emb_size, pos_emb.shape[1]))

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Reshape for upsampling (add batch and channel dims)
            pos_emb = mx.expand_dims(pos_emb, 0)
            pos_emb = mx.transpose(pos_emb, (0, 3, 1, 2))

            # Create and apply upsampler
            upsampler = nn.Upsample(
                scale_factor=(
                    patch_num_0 / pos_emb.shape[2],
                    patch_num_1 / pos_emb.shape[3],
                ),
                mode="linear",  # MLX doesn't have bicubic, using linear as closest alternative
                align_corners=False,
            )
            pos_emb = upsampler(pos_emb)

            # Restore original dimensions
            pos_emb = mx.transpose(pos_emb, (0, 2, 3, 1))
            pos_emb = mx.squeeze(pos_emb, 0)

        pos_emb = mx.reshape(pos_emb, (-1, pos_emb.shape[-1]))

        # Expand cls_emb and pos_emb
        expanded_cls = cls_emb[None, :, :]
        expanded_pos = pos_emb[None, :, :]

        # Concatenate and add to x
        pos_embedding = mx.concatenate([expanded_cls, expanded_pos], axis=1)
        x = x + pos_embedding
        return x

    def __call__(self, x: mx.array, patch_num: int = None) -> List[mx.array]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        # (Optional) Due to quantization, pad around the image to match intermediate_size
        x = pad_to_multiple(x, self.config.intermediate_size)

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        expanded_class_emb = _expand_token(self.class_embedding, x.shape[0])
        expanded_class_emb = expanded_class_emb

        x = mx.concatenate([expanded_class_emb, x], axis=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "molmo":
            raise ValueError(
                f"Model type {self.model_type} not supported. Currently only 'molmo' is supported"
            )
        self.image_vit = VisionTransformer(config)
        self.num_prefix_tokens = 1

        self.image_pooling_2d = MultiHeadDotProductAttention(config, is_vit_layer=False)
        self.image_projector = MLP(config, config.image_emb_dim)
        self.pad_embed = mx.zeros((2, config.image_emb_dim * 2))

    def encode_image(self, images: mx.array) -> mx.array:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape

        # Check for -1 values across dimensions 1 and 2
        reshaped_images = mx.reshape(images, (B * T, N, D))
        mask = ~mx.all(reshaped_images == -1, axis=(1, 2), keepdims=True)

        # Output all hidden states
        images = reshaped_images
        image_features = self.image_vit(images)

        if cfg.vit_layers is not None:
            features = []
            for layer in cfg.vit_layers:
                features.append(image_features[layer])
            image_features = mx.concatenate(features, axis=-1)
        else:
            image_features = image_features[-1]

        cls_embed = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = mx.reshape(image_features, (B, T, N, -1))

        cls_embed = mx.reshape(cls_embed, (B, T, -1)) if cls_embed is not None else None

        return image_features, cls_embed

    def __call__(
        self, images: mx.array, image_masks: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        cfg = self.config

        batch_size, num_image = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = image_masks == 0
                pad_embed = mx.reshape(self.pad_embed, (1, 1, 1, -1))
                image_features = image_features + pad_embed * mx.expand_dims(
                    all_pad, -1
                )
            elif cfg.image_padding_embed == "regress":
                pad_embed = mx.reshape(self.pad_embed, (1, 1, 1, -1))
                image_features = image_features + pad_embed * mx.expand_dims(
                    mx.maximum(image_masks, mx.zeros_like(image_masks)), -1
                )
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = mx.reshape(self.pad_embed, (2, 1, 1, 1, -1))
                all_pad = image_masks == 0
                partial_pad = mx.logical_and(image_masks < 1, mx.logical_not(all_pad))
                partial_pad = partial_pad
                all_pad = all_pad
                image_features = image_features + pad_embed[0] * mx.expand_dims(
                    all_pad, -1
                )
                image_features = image_features + pad_embed[1] * mx.expand_dims(
                    partial_pad, -1
                )
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = mx.reshape(
            image_features, (batch_size, num_image) + cfg.image_num_patch + (-1,)
        )

        if cfg.image_num_patch[0] % cfg.image_pooling_h == 1:
            # Pad so we can still pool 2x2 patches
            image_features = mx.pad(
                image_features, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)]
            )

        # image pooling
        # MLX equivalent of einops rearrange
        h_blocks = image_features.shape[2] // cfg.image_pooling_h
        w_blocks = image_features.shape[3] // cfg.image_pooling_w
        image_features = mx.reshape(
            mx.transpose(
                mx.reshape(
                    image_features,
                    (
                        batch_size,
                        num_image,
                        h_blocks,
                        cfg.image_pooling_h,
                        w_blocks,
                        cfg.image_pooling_w,
                        -1,
                    ),
                ),
                (0, 1, 2, 4, 3, 5, 6),
            ),
            (
                batch_size * num_image * h_blocks * w_blocks,
                cfg.image_pooling_h * cfg.image_pooling_w,
                -1,
            ),
        )

        if cfg.image_pooling_2d == "attention-meanq":
            query = mx.mean(image_features, axis=-2, keepdims=True)
            image_features = self.image_pooling_2d(query, image_features)
        elif cfg.image_pooling_2d not in {"none", "stack"}:
            image_features = self.image_pooling_2d(
                image_features[:, :1, :], image_features
            )

        h, w = cfg.llm_patches_per_crop
        image_features = mx.reshape(image_features, (batch_size, num_image, h * w, -1))

        # # MLP layer to map the feature
        image_features = self.image_projector(image_features)

        return image_features, cls_embed
