"""Sapiens2 top-level model: backbone + decode head + weight sanitize.

The PyTorch checkpoint is a flat tensor bag keyed as
  backbone.<...>.weight
  decode_head.<...>.weight
with no `model.` wrapper.  Sanitize's job is to:

1. Transpose 4-D conv tensors:
     Conv2d           (out, in, kH, kW) → (out, kH, kW, in)
     ConvTranspose2d  (in,  out, kH, kW) → (out, kH, kW, in)
2. Rename internal list-container names back to plain indices so that MLX's
   serialization of `_UpsampleBlock.children_list` lines up with PT's flat
   `upsample_blocks.<i>.0.weight` layout.
"""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .heads import build_head
from .vision import Sapiens2Backbone


# 4-D weight names in the checkpoint that are ConvTranspose2d (need [in,out,H,W]→[out,H,W,in]).
# Everything else with ndim==4 is a Conv2d and gets [out,in,H,W]→[out,H,W,in].
CONV_TRANSPOSE_HINTS = ("deconv_layers.",)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = Sapiens2Backbone(config.backbone_config)
        self.decode_head = build_head(config.head_config)

    def __call__(self, pixel_values: mx.array):
        featmap = self.backbone(pixel_values)
        return self.decode_head(featmap)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch checkpoint keys/shapes to the MLX module layout."""
        sanitized: Dict[str, mx.array] = {}
        for k, v in weights.items():
            new_k = k

            # Strip optional "model." prefix (none expected for current checkpoints,
            # but matches the pattern used by other ports).
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]

            # Inject `children_list.` so decode_head.upsample_blocks.<i>.<j>.weight
            # lands on the MLX module's `<i>.children_list.<j>` attribute path.
            # Only the first child (index 0) carries weights.
            #   PT: upsample_blocks.2.0.weight
            #   MLX:upsample_blocks.2.children_list.0.weight
            if ".upsample_blocks." in new_k:
                parts = new_k.split(".")
                # Find "upsample_blocks" and inject after the index (<i>)
                for i, tok in enumerate(parts):
                    if tok == "upsample_blocks":
                        # parts[i+1] = <i>, parts[i+2] = <j>
                        if i + 2 < len(parts) and parts[i + 2].isdigit():
                            parts.insert(i + 2, "children_list")
                        break
                new_k = ".".join(parts)

            # 4-D weight transpose
            if v.ndim == 4:
                if any(hint in new_k for hint in CONV_TRANSPOSE_HINTS):
                    # ConvTranspose2d: (in, out, kH, kW) → (out, kH, kW, in)
                    v = v.transpose(1, 2, 3, 0)
                else:
                    # Conv2d: (out, in, kH, kW) → (out, kH, kW, in)
                    v = v.transpose(0, 2, 3, 1)

            sanitized[new_k] = v
        return sanitized
