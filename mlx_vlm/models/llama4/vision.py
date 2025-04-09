import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str
    attention_dropout: float
    hidden_act: str
    hidden_size: int
    image_size: int
    initializer_range: float
    intermediate_size: int
    multi_modal_projector_bias: bool
    norm_eps: float
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    pixel_shuffle_ratio: float
    projector_dropout: float
    projector_input_dim: int
    projector_output_dim: int
    rope_theta: float
    vision_feature_layer: int
    vision_feature_select_strategy: str
    vision_output_dim: int

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


activations = {}


def check_and_log_stats(name, tensor, capture_activations=True):
    """Helper function to check for anomalies and log stats."""
    if not capture_activations:
        return

    print(f"--- Activation Stats: {name} ---")
    # Check for NaNs/Infs
    has_nan = mx.isnan(tensor).any()
    has_inf = mx.isinf(tensor).any()
    if has_nan:
        print(f"WARNING: Found NaN in {name}")
    if has_inf:
        print(f"WARNING: Found Inf in {name}")

    # Calculate and print stats (ensure computation happens)
    min_val = mx.min(tensor).item()
    max_val = mx.max(tensor).item()
    mean_val = mx.mean(tensor).item()
    std_val = mx.std(tensor).item()
    print(f"  Shape: {tensor.shape}")
    print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print("-" * (len(name) + 24))

    # Store the tensor itself
    activations[name] = tensor


class Llama4MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=False,
        )

    def __call__(self, image_features):
        hidden_states = self.linear_1(image_features)
        return hidden_states


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    reshaped_tensor = input_tensor.reshape(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    reshaped_tensor = reshaped_tensor.reshape(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    output_tensor = reshaped_tensor.reshape(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(
            config.projector_input_dim // (self.pixel_shuffle_ratio**2)
        )
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP(config, bias=False, is_projector=True)

    def __call__(self, encoded_patches: mx.array) -> mx.array:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


# TODO there is a different RoPE for vision encoder, defined as below
def reshape_for_broadcast(freqs_ci: mx.array, query: mx.array):
    ndim = query.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
    return freqs_ci.reshape(*shape)


def view_as_complex(x):
    """
    Convert a tensor with shape (..., 2) to a complex tensor with shape (...).

    Args:
        x: A real tensor with last dimension of size 2.

    Returns:
        A complex tensor with size one less than the input.
    """
    # Ensure the last dimension is size 2
    assert x.shape[-1] == 2, f"Last dimension must be 2, got {x.shape[-1]}"

    # Get real and imaginary parts
    real, imag = x[..., 0], x[..., 1]

    # Create complex tensor
    return real + 1j * imag


def view_as_real(x):
    """
    Convert a complex tensor with shape (...) to a real tensor with shape (..., 2).

    Args:
        x: A complex tensor.

    Returns:
        A real tensor with an extra dimension of size 2.
    """
    # Get real and imaginary parts
    real = mx.real(x)
    imag = mx.imag(x)

    # Combine into a tensor with last dimension 2
    return mx.stack([real, imag], axis=-1)


def vision_apply_rotary_emb(
    query: mx.array,
    key: mx.array,
    freqs_ci: mx.array,
) -> Tuple[mx.array, mx.array]:

    query_ = view_as_complex(query.astype(mx.float32).reshape(*query.shape[:-1], -1, 2))
    key_ = view_as_complex(key.astype(mx.float32).reshape(*key.shape[:-1], -1, 2))
    freqs_ci = reshape_for_broadcast(freqs_ci=freqs_ci, query=query_)
    query_out = view_as_real(query_ * freqs_ci).flatten(3)
    key_out = view_as_real(key_ * freqs_ci).flatten(3)
    return query_out.astype(query.dtype), key_out.astype(key.dtype)


class Llama4VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.embed_dim, bias=True
        )

    def __call__(
        self,
        hidden_states: mx.array,
        freqs_ci: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ):
        B, L, D = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(B, L, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(B, L, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(B, L, self.num_heads, -1)

        query_states, key_states = vision_apply_rotary_emb(
            query_states, key_states, freqs_ci=freqs_ci
        )

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scale
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Llama4VisionMLP(nn.Module):
    def __init__(self, config, bias=True, is_projector=False):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU(approx="precise")  # ACT2FN[config.hidden_act]
        self.is_projector = is_projector
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Determine dimensions for first linear layer based on whether this is a projector
        fc1_input_dim = self.intermediate_size if is_projector else self.hidden_size
        fc1_output_dim = (
            config.projector_input_dim if is_projector else self.intermediate_size
        )

        self.fc1 = nn.Linear(fc1_input_dim, fc1_output_dim, bias=bias)

        # Determine dimensions for second linear layer
        fc2_input_dim = (
            config.projector_output_dim if is_projector else self.intermediate_size
        )
        fc2_output_dim = (
            config.projector_output_dim if is_projector else self.hidden_size
        )

        self.fc2 = nn.Linear(fc2_input_dim, fc2_output_dim, bias=bias)

        self.is_projector = is_projector

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        if self.is_projector:
            return self.activation_fn(self.fc2(hidden_states))

        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Llama4VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Llama4VisionAttention(config)
        self.mlp = Llama4VisionMLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        hidden_state: mx.array,
        freqs_ci: mx.array,
        mask: Optional[mx.array] = None,
    ):
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(
            hidden_state,
            freqs_ci=freqs_ci,
            mask=mask,
        )
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        return hidden_state


class Llama4VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Llama4VisionEncoderLayer`].

    Args:
        config: VisionConfig
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layers = [
            Llama4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.config = config

    def __call__(
        self,
        hidden_states: mx.array,
        freqs_ci: mx.array,  # TODO move this to an attribute instead of keeping it around
        mask: Optional[mx.array] = None,
    ):

        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_state=hidden_states,
                mask=mask,
                freqs_ci=freqs_ci,
            )
            check_and_log_stats(f"encoder_layer_{i}", hidden_states)

        return hidden_states


class Llama4UnfoldConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = config.patch_size
        self.linear = nn.Linear(
            config.num_channels * kernel_size[0] * kernel_size[1],
            config.hidden_size,
            bias=False,
        )

    def _pair(self, x):
        """Convert input to a pair of values."""
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x, x)

    def unfold(self, input_tensor):
        """
        Extract sliding local blocks from a batched input tensor (MLX implementation).

        This is equivalent to PyTorch's nn.functional.unfold or im2col operation.

        Args:
            input_tensor: Input tensor of shape (B, C, H, W)

        Returns:
            Unfolded tensor of shape (B, C*kernel_height*kernel_width, L)
            where L is the number of blocks
        """
        # Convert to pairs
        kernel_size = self._pair(self.kernel_size)
        stride = self._pair(self.stride)
        padding = (0, 0)  # No padding in the original code
        dilation = (1, 1)  # Default dilation

        # Input shape
        batch_size, channels, height, width = input_tensor.shape

        # Calculate output dimensions
        height_out = (
            height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        width_out = (
            width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1

        # Initialize output arrays
        blocks = []

        # Extract blocks
        for i in range(0, height - kernel_size[0] * dilation[0] + 1, stride[0]):
            for j in range(0, width - kernel_size[1] * dilation[1] + 1, stride[1]):
                # Extract the block for all channels
                block = []
                for di in range(kernel_size[0]):
                    for dj in range(kernel_size[1]):
                        h_idx = i + di * dilation[0]
                        w_idx = j + dj * dilation[1]
                        # Get the block for all channels and add to our list
                        block.append(input_tensor[:, :, h_idx, w_idx])

                # Stack the channel-blocks
                block = mx.stack(block, axis=1)  # Shape: (B, k*k, C)
                block = mx.transpose(block, [0, 2, 1])  # Shape: (B, C, k*k)
                blocks.append(block)

        # Stack all blocks together
        result = mx.stack(blocks, axis=-1)  # Shape: (B, C, k*k, L)

        # Reshape to match PyTorch's unfold output format: (B, C*k*k, L)
        result = mx.reshape(
            result,
            (
                batch_size,
                channels * kernel_size[0] * kernel_size[1],
                height_out * width_out,
            ),
        )

        return result

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.linear(hidden_states)
        return hidden_states


# class Llama4VisionRotaryEmbedding:
#     def __init__(self, config):
#         super().__init__()
#         idx = config.image_size // config.patch_size
#         img_idx = mx.arange(idx**2, dtype=mx.int32).reshape(idx**2, 1)
#         img_idx = mx.concatenate([img_idx, img_idx[:1]], axis=0)
#         img_idx[-1, -1] = -2  # ID_CLS_TOKEN
#         frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
#         frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
#         freq_dim = config.hidden_size // config.num_attention_heads // 2
#         rope_freq = 1.0 / (
#             config.rope_theta
#             ** (
#                 mx.arange(0, freq_dim, 2, dtype=mx.float32)[: (freq_dim // 2)]
#                 / freq_dim
#             )
#         )

#         # Expand dimensions for frequencies_x and frequencies_y
#         freqs_x_expanded = (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
#         freqs_y_expanded = (frequencies_y + 1)[..., None] * rope_freq[None, None, :]

#         def repeat_interleave(tensor, repeats, dim=-1):
#             # Get the shape
#             shape = list(tensor.shape)

#             # Reshape to add an extra dimension for repeating
#             tensor = mx.reshape(tensor, shape[:-1] + [shape[-1], 1])

#             # Repeat along the new dimension
#             tensor = mx.repeat(tensor, repeats, axis=-1)

#             # Reshape to flatten the last two dimensions
#             return mx.reshape(tensor, shape[:-1] + [shape[-1] * repeats])

#         # Apply interleaving
#         freqs_x = repeat_interleave(freqs_x_expanded, 2)
#         freqs_y = repeat_interleave(freqs_y_expanded, 2)
#         freqs = mx.concatenate([freqs_x, freqs_y], axis=-1).astype(mx.float32)[..., ::2]
#         # Replaced masked_fill with where
#         mask = img_idx.reshape(-1, 1, 1) < 0
#         freqs = mx.where(mask, mx.zeros_like(freqs), freqs)
#         freq_cis = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)
#         freq_cis = view_as_complex(freq_cis)
#         self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

#     def __call__(self, hidden_states):
#         return self.freqs_ci

import torch


class Llama4VisionRotaryEmbedding:
    def __init__(self, config):
        super().__init__()
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
        frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim)
        )
        freqs_x = (
            (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        freq_cis = torch.view_as_complex(
            torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        )
        self.freqs_ci = mx.array(freq_cis)  # idx**2, idx**2, idx * 2

    def __call__(self, hidden_states):
        return self.freqs_ci


class VisionModel(nn.Module):
    base_model_prefix = "vision_model"
    config_class = VisionConfig

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(config)

        self.rotary_embedding = Llama4VisionRotaryEmbedding(config)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.model = Llama4VisionEncoder(config)
        self.vision_adapter = Llama4VisionPixelShuffleMLP(config)

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def __call__(
        self,
        pixel_values: mx.array,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        capture_activations: Optional[bool] = True,
    ):

        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1

        hidden_state = self.patch_embedding(pixel_values)
        check_and_log_stats("patch_embedding", hidden_state)

        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks,
            num_patches,
            hidden_dim,
        )
        self.class_embedding = self.scale * mx.random.normal((self.hidden_size,))
        class_embedding = mx.tile(self.class_embedding, (hidden_state.shape[0], 1, 1))

        print(f"Class embedding: {class_embedding}")
        hidden_state = mx.concatenate([hidden_state, class_embedding], axis=1)
        num_patches += 1
        check_and_log_stats("after_cls_concat", hidden_state)

        # Position embeddings
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media,
            num_chunks,
            num_patches,
            hidden_dim,
        )
        self.positional_embedding_vlm = self.scale * mx.random.normal(
            (self.num_patches, self.hidden_size)
        )
        positional_embedding = self.positional_embedding_vlm.astype(hidden_state.dtype)
        hidden_state = hidden_state + positional_embedding
        check_and_log_stats("after_pos_embedding", hidden_state)

        hidden_state = self.layernorm_pre(hidden_state)
        check_and_log_stats("after_layernorm_pre", hidden_state)

        hidden_state = hidden_state.reshape(batch_size_times_num_tiles, -1, hidden_dim)
        freqs_ci = self.rotary_embedding(pixel_values)

        hidden_state = self.model(
            hidden_state,
            mask=None,
            freqs_ci=freqs_ci,
        )
        check_and_log_stats("encoder_output", hidden_state)

        hidden_state = self.layernorm_post(hidden_state)
        check_and_log_stats("after_layernorm_post", hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        final_hidden_state = self.vision_adapter(hidden_state)
        check_and_log_stats("final_adapter_output", final_hidden_state)

        # Return only the final state
        return final_hidden_state

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if (
                "position_ids" in k
                or "class_embedding" in k
                or "positional_embedding_vlm" in k
            ):
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v

        return sanitized_weights
