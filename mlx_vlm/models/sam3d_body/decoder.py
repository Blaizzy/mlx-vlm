"""Promptable decoder wrapper for SAM 3D Body."""

import mlx.core as mx
import mlx.nn as nn

from .layers import LayerNorm32
from .transformer import TransformerDecoderLayer


class PromptableDecoder(nn.Module):
    """Wraps N TransformerDecoderLayers with a final norm.

    Takes image features (B, H_p, W_p, C) and token embeddings (B, N, D),
    runs cross-attention decoder layers, returns processed tokens.

    Supports intermediate predictions: at each layer, the caller can extract
    a pose prediction and update keypoint tokens before the next layer.
    """

    def __init__(
        self,
        dims: int = 1024,
        context_dims: int = 1280,
        depth: int = 6,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
    ):
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(
                token_dims=dims,
                context_dims=context_dims,
                num_heads=num_heads,
                head_dims=head_dims,
                mlp_dims=mlp_dims,
                repeat_pe=True,
                skip_first_pe=(i == 0),
            )
            for i in range(depth)
        ]
        self.norm_final = LayerNorm32(dims, eps=1e-6)

    def __call__(
        self,
        tokens: mx.array,
        image_embedding: mx.array,
        token_pe: mx.array = None,
        image_pe: mx.array = None,
        token_to_pose_fn=None,
        kp_update_fn=None,
    ) -> tuple[mx.array, list]:
        """
        Args:
            tokens: (B, N, D) token embeddings
            image_embedding: (B, H_p, W_p, C) image features (NHWC from backbone)
            token_pe: (B, N, D) positional encoding for tokens
            image_pe: (B, H_p, W_p, C) positional encoding for image
            token_to_pose_fn: callable(normed_tokens, layer_idx) -> pose_output dict
                Called at every layer to get intermediate pose predictions.
            kp_update_fn: callable(tokens, token_pe, pose_output, layer_idx, image_features) -> (tokens, token_pe)
                Called at non-final layers to update keypoint tokens using
                predicted 2D/3D keypoints from intermediate predictions.
        Returns:
            (normed_tokens, all_intermediate_outputs)
        """
        B = image_embedding.shape[0]
        # Flatten spatial dims: (B, H_p, W_p, C) -> (B, H*W, C)
        context = image_embedding.reshape(B, -1, image_embedding.shape[-1])
        context_pe = None
        if image_pe is not None:
            context_pe = image_pe.reshape(B, -1, image_pe.shape[-1])

        all_outputs = []
        for i, layer in enumerate(self.layers):
            tokens, context = layer(
                tokens, context, x_pe=token_pe, context_pe=context_pe
            )

            # Intermediate predictions at every layer
            normed = self.norm_final(tokens)
            if token_to_pose_fn is not None:
                pose_output = token_to_pose_fn(normed, i)
                all_outputs.append(pose_output)

                # Update keypoint tokens for next layer (skip after last layer)
                if kp_update_fn is not None and i < len(self.layers) - 1:
                    tokens, token_pe = kp_update_fn(
                        tokens, token_pe, pose_output, i, image_embedding
                    )

        return self.norm_final(tokens), all_outputs
