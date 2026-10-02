"""DPT head with temporal motion modules for Video Depth Anything."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dpt import Scratch as DPTScratch
from ..dpt import reassemble_layers
from ..interpolate import resize_bilinear_nhwc
from .config import ModelConfig
from .motion import TemporalModule


class Scratch(DPTScratch):
    def __init__(self, out_channels: List[int], features: int):
        super().__init__(out_channels, features)
        self.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1
        )
        # A plain list keeps checkpoint indices: output_conv2.0, output_conv2.2
        self.output_conv2 = [
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        ]


class DPTHeadTemporal(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        in_channels = config.embed_dim
        features = config.features
        out_channels = config.out_channels

        self.projects = [
            nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0)
            for c in out_channels
        ]
        self.resize_layers = reassemble_layers(out_channels)
        self.scratch = Scratch(out_channels, features)

        motion_kwargs = dict(
            num_attention_heads=config.num_attention_heads,
            num_transformer_block=config.num_transformer_block,
            num_attention_blocks=config.num_attention_blocks,
            norm_num_groups=config.norm_num_groups,
            temporal_max_len=config.num_frames,
            pos_embedding_type=config.pe,
        )
        self.motion_modules = [
            TemporalModule(in_channels=out_channels[2], **motion_kwargs),
            TemporalModule(in_channels=out_channels[3], **motion_kwargs),
            TemporalModule(in_channels=features, **motion_kwargs),
            TemporalModule(in_channels=features, **motion_kwargs),
        ]

    def __call__(
        self,
        out_features: List[Tuple[mx.array, mx.array]],
        patch_h: int,
        patch_w: int,
        frame_length: int,
        micro_batch_size: int = 4,
    ) -> mx.array:
        out = []
        for i, (x, _cls_token) in enumerate(out_features):
            # (B*T, N, C) -> (B*T, patch_h, patch_w, C) channel-last
            x = x.reshape(-1, patch_h, patch_w, x.shape[-1])
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        B = layer_1.shape[0] // frame_length

        def motion(i, x):
            # (B*T, H, W, C) -> (B, T, H, W, C) -> temporal module -> back
            x = x.reshape(B, frame_length, *x.shape[1:])
            x = self.motion_modules[i](x)
            return x.reshape(B * frame_length, *x.shape[2:])

        layer_3 = motion(0, layer_3)
        layer_4 = motion(1, layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        path_4 = motion(2, path_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[1:3])
        path_3 = motion(3, path_3)

        def head(path_3, layer_2_rn, layer_1_rn):
            path_2 = self.scratch.refinenet2(
                path_3, layer_2_rn, size=layer_1_rn.shape[1:3]
            )
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = resize_bilinear_nhwc(
                out,
                (patch_h * self.patch_size, patch_w * self.patch_size),
                align_corners=True,
            )
            # The reference runs the output head in full precision
            out = out.astype(mx.float32)
            for layer in self.scratch.output_conv2:
                out = layer(out)
            return out

        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            return head(path_3, layer_2_rn, layer_1_rn)

        chunks = []
        for i in range(0, batch_size, micro_batch_size):
            sl = slice(i, i + micro_batch_size)
            chunk = head(path_3[sl], layer_2_rn[sl], layer_1_rn[sl])
            mx.eval(chunk)
            chunks.append(chunk)
        return mx.concatenate(chunks, axis=0)
