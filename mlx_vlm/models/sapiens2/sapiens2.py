"""Sapiens2 model wrapper: backbone + task decode head (inference only)."""

import re
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .backbone import Sapiens2Backbone
from .config import ModelConfig
from .heads import DeconvHead, PixelShuffleHead

# q/k/v projection keys of the checkpoint, merged into ``wqkv`` on load.
_QKV_KEY = re.compile(r"^(.*\.attn\.)w([qkv])\.(weight|bias)$")

# Predictor conv attribute name per task, as in the original checkpoints.
_PREDICTOR_NAMES = {
    "seg": "conv_seg",
    "pose": "conv_pose",
    "normal": "conv_normal",
    "pointmap": "conv_pointmap",
    "matting": "conv_matting",
}


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.task = config.task

        self.backbone = Sapiens2Backbone(config)

        self.decode_head = None
        if self.task != "backbone":
            if config.head_config is None:
                raise ValueError(
                    f"Task {self.task!r} requires a head_config in config.json"
                )
            self.decode_head = self._build_head()

    def _build_head(self) -> nn.Module:
        config = self.config
        hc = config.head_config
        num_labels = config.num_labels
        if num_labels is None:
            raise ValueError("num_labels (or id2label) is required for task heads")

        if hc.use_pixel_shuffle:
            scale_final = None
            if hc.scale_conv_out_channels is not None:
                scale_final = [
                    hc.scale_final_input_size,
                    *(hc.scale_final_hidden_sizes or []),
                    1,
                ]
            return PixelShuffleHead(
                in_channels=config.hidden_size,
                num_labels=num_labels,
                predictor_name=_PREDICTOR_NAMES[self.task],
                upsample_channels=hc.upsample_out_channels,
                conv_out_channels=hc.conv_out_channels,
                conv_kernel_sizes=hc.conv_kernel_sizes,
                scale_conv_out_channels=hc.scale_conv_out_channels,
                scale_conv_kernel_sizes=hc.scale_conv_kernel_sizes,
                scale_final_layer=scale_final,
            )
        return DeconvHead(
            in_channels=config.hidden_size,
            num_labels=num_labels,
            predictor_name=_PREDICTOR_NAMES[self.task],
            deconv_out_channels=hc.upsample_out_channels,
            deconv_kernel_sizes=hc.upsample_kernel_sizes,
            conv_out_channels=hc.conv_out_channels,
            conv_kernel_sizes=hc.conv_kernel_sizes,
        )

    def __call__(self, pixel_values: mx.array) -> Dict[str, mx.array]:
        """pixel_values: (B, H, W, 3) channel-last, ImageNet-normalized.

        Returns a dict of channel-last outputs (see README). Keys depend on
        the task: ``logits`` (seg), ``heatmaps`` (pose), ``normals``,
        ``pointmaps`` + ``scales``, ``alphas`` + ``foregrounds`` (matting),
        or ``last_hidden_state`` + ``pooler_output`` (backbone).
        """
        tokens, (h, w) = self.backbone(pixel_values)

        if self.task == "backbone":
            return {
                "last_hidden_state": tokens,
                "pooler_output": tokens[:, 0],
            }

        patch_tokens = tokens[:, self.backbone.num_prefix_tokens :, :]
        feature_map = patch_tokens.reshape(
            tokens.shape[0], h, w, self.config.hidden_size
        )
        out = self.decode_head(feature_map)

        if self.task == "pointmap":
            pointmaps, scales = out
            return {"pointmaps": pointmaps, "scales": scales}
        if self.task == "matting":
            matting = mx.sigmoid(out)
            return {"foregrounds": matting[..., :3], "alphas": matting[..., 3:]}
        if self.task == "seg":
            return {"logits": out}
        if self.task == "pose":
            return {"heatmaps": out}
        if self.task == "normal":
            return {"normals": out}
        raise ValueError(f"Unknown task {self.task!r}")

    @staticmethod
    def _merge_qkv(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Concatenate wq/wk/wv (weight and bias) into wqkv, in that order."""
        weights = dict(weights)
        groups: Dict[tuple, Dict[str, str]] = {}
        for k in weights:
            if m := _QKV_KEY.match(k):
                groups.setdefault((m[1], m[3]), {})[m[2]] = k
        for (prefix, kind), names in groups.items():
            if len(names) == 3:
                weights[f"{prefix}wqkv.{kind}"] = mx.concatenate(
                    [weights.pop(names[n]) for n in "qkv"], axis=0
                )
        return weights

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Adapt checkpoint weights to this module layout.

        - q/k/v projections are merged into ``wqkv``.
        - Pretrain checkpoints store backbone weights without the
          ``backbone.`` prefix used by the task checkpoints.
        - Torch conv kernels are relaid to channel-last: Conv2d
          (out, in, kh, kw) -> (out, kh, kw, in); ConvTranspose2d
          (in, out, kh, kw) -> (out, kh, kw, in). A weight is transposed only
          when its shape does not already match the parameter.
        - Every floating tensor is cast to the dtype of the attention weights:
          a mixed-precision file would otherwise promote the whole residual
          stream to float32 at the first norm or bias.
        """
        targets = {k: v.shape for k, v in tree_flatten(self.parameters())}
        sanitized = {}
        for k, v in self._merge_qkv(weights).items():
            if k not in targets and "backbone." + k in targets:
                k = "backbone." + k
            target = targets.get(k)
            if target is not None and v.ndim == 4 and v.shape != target:
                if "deconv_layers" in k:
                    cand = v.transpose(1, 2, 3, 0)
                else:
                    cand = v.transpose(0, 2, 3, 1)
                if cand.shape == target:
                    v = cand
            sanitized[k] = v

        dtype = next(
            (v.dtype for k, v in sanitized.items() if k.endswith("attn.wqkv.weight")),
            None,
        )
        if dtype is not None:
            sanitized = {
                k: (
                    v.astype(dtype)
                    if mx.issubdtype(v.dtype, mx.floating) and v.dtype != dtype
                    else v
                )
                for k, v in sanitized.items()
            }
        return sanitized
