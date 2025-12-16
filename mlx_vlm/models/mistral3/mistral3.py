from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..pixtral import VisionModel
from .config import ModelConfig
from .language import LanguageModel


def _pair(x) -> Tuple[int, int]:
    """Convert input to a pair of values."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def unfold(
    input: mx.array,
    kernel_size: Union[int, Tuple[int, int], List[int]],
    dilation: Union[int, Tuple[int, int], List[int]] = 1,
    padding: Union[int, Tuple[int, int], List[int]] = 0,
    stride: Union[int, Tuple[int, int], List[int]] = 1,
) -> mx.array:
    """
    Extract sliding local blocks from a batched input tensor (MLX implementation).

    This is equivalent to PyTorch's nn.functional.unfold or im2col operation.

    Args:
        input: Input tensor of shape (B, C, H, W)
        kernel_size: Size of the sliding blocks
        dilation: Controls the spacing between kernel elements
        padding: Controls the amount of implicit padding
        stride: Controls the stride between blocks

    Returns:
        Unfolded tensor of shape (B, C*kernel_height*kernel_width, L)
        where L is the number of blocks
    """
    # Convert to pairs
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    stride = _pair(stride)

    # Input shape
    batch_size, channels, height, width = input.shape

    # Add padding if needed
    if padding[0] > 0 or padding[1] > 0:
        padding_shape = (
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1]),
        )
        input = mx.pad(input, padding_shape)

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
    for i in range(
        0, height + 2 * padding[0] - kernel_size[0] * dilation[0] + 1, stride[0]
    ):
        for j in range(
            0, width + 2 * padding[1] - kernel_size[1] * dilation[1] + 1, stride[1]
        ):
            # Extract the block for all channels
            block = []
            for di in range(kernel_size[0]):
                for dj in range(kernel_size[1]):
                    h_idx = i + di * dilation[0]
                    w_idx = j + dj * dilation[1]
                    # Get the block for all channels and add to our list
                    block.append(input[:, :, h_idx, w_idx])

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


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def __call__(self, image_features: mx.array, image_sizes: mx.array) -> mx.array:

        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        image_features = image_features.astype(mx.bfloat16)
        image_sizes = mx.array(image_sizes)

        # Split the image features into chunks based on tokens_per_image
        split_indices = []
        current_index = 0
        for tokens in tokens_per_image:
            split_indices.append(current_index + tokens)
            current_index += tokens

        # Perform the split
        chunks = mx.split(image_features, split_indices[:-1], axis=1)

        permuted_tensor = []
        for image_index, image_tokens in enumerate(chunks):

            # Reshape image_tokens into a 2D grid
            if image_tokens.shape[1] > 0:
                h, w = image_sizes[image_index].tolist()

                image_grid = image_tokens.reshape(h, w, d).transpose(2, 0, 1)[None, ...]

                grid = unfold(
                    image_grid,
                    kernel_size=self.spatial_merge_size,
                    stride=self.spatial_merge_size,
                )
                grid = grid.reshape(d * self.spatial_merge_size**2, -1).T
                permuted_tensor.append(grid)

        image_features = mx.concatenate(permuted_tensor, axis=0)
        image_features = self.merging_layer(image_features)
        return image_features[None, ...]


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm = nn.RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)

        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def __call__(self, x: mx.array, image_sizes: mx.array) -> mx.array:
        x = self.norm(x)

        x = self.patch_merger(x, image_sizes)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.vision_feature_layer = config.vision_feature_layer

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_sizes = kwargs.get("image_sizes", None)

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the output hidden states from the vision model
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Pass pixel_values as list of images, as each image is individually run through conv2d and position encoding
        # Reference code from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py#L479C9-L479C21
        # and mistral_inference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py#L85
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_index, image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Merge image features into input embeddings at image token positions.

        Args:
            image_token_index: Token ID for image placeholder
            image_features: Vision features from the projector [1, num_features, hidden_dim]
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Updated input embeddings with image features inserted
        """
        # Remove the extra batch dimension from image_features if present
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)  # [num_features, hidden_dim]

        # Positions of <image> tokens in input_ids
        image_positions = input_ids == image_token_index

        # Get dimensions
        batch_size, seq_len = input_ids.shape

        # Process each batch item
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            # Get mask for this batch
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                # Extract features for this batch
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                # Validate we have the right number of features
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                # Create indices for gathering
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)

                # Gather features
                gathered_features = batch_features[feature_indices]

                # Combine with original embeddings
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                # No image tokens in this batch item
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        # Stack all batch outputs
        return mx.stack(batch_outputs, axis=0)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        logits = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
        )
        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" in key and "vision_model" not in key:
                if "transformer" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")
                if "patch_conv" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")
                if "ln_pre" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")

            elif "vision_encoder" in key and "vision_tower" not in key:
                if "transformer" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )
                if "patch_conv" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )
                if "ln_pre" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )

            elif "model.language_model" in key and "language_model.model" not in key:
                key = key.replace("model.language_model", "language_model.model")

            elif "lm_head" in key and "language_model" not in key:
                key = key.replace("lm_head", "language_model.lm_head")

            elif "model.vision_projection" in key:
                key = key.replace("model.vision_projection", "multi_modal_projector")

            return key

        return {transform_key(k): v for k, v in weights.items()}

    @property
    def layers(self):
        return self.language_model.model.layers
