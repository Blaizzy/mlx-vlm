"""Top-level RT-DETRv2 model.

Composes the vision tower (backbone + hybrid encoder), the per-level
projection that bridges encoder output to decoder input, the encoder
query-selection warm-up, and the deformable-attention decoder.

Weight sanitization shares its rename table with `convert.py`: that
module is the single source of truth so the in-process loader
(`mlx_vlm.utils.load_model`) and the offline checkpoint converter agree
on every key.
"""

from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .convert import rename as _rename
from .convert import should_drop as _should_drop
from .transformer import MLP, Decoder, generate_anchors
from .vision import VisionTower


class _DecoderInputProj(nn.Module):
    """1x1 Conv + BN bridging an encoder feature to `d_model`.

    HF saves this as `Sequential[Conv2d, BatchNorm2d]`, so keys are
    `.{N}.0.X` and `.{N}.1.X`; the rename pipeline maps those to
    `.{N}.conv.X` and `.{N}.bn.X`.
    """

    def __init__(self, in_c: int, out_c: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(out_c, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class _EncOutput(nn.Module):
    """Linear + LayerNorm encoder query head.

    HF saves it as `Sequential[Linear, LayerNorm]`; rename pipeline maps
    `.0.` to `.fc.` and `.1.` to `.ln.`.
    """

    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.ln(self.fc(x))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.vision = VisionTower(config)

        d = config.d_model
        eps = config.batch_norm_eps
        ln_eps = config.layer_norm_eps

        # decoder_in_channels gives the input width per level; some configs
        # have encoder_hidden_dim != d_model so this is a real projection.
        self.decoder_input_proj = [
            _DecoderInputProj(in_c, d, eps=eps) for in_c in config.decoder_in_channels
        ]

        self.enc_output = _EncOutput(d, eps=ln_eps)
        self.enc_score_head = nn.Linear(d, config.num_labels)
        self.enc_bbox_head = MLP(d, d, 4, num_layers=3)

        # Training-only contrastive denoising group; kept as a real
        # parameter so the checkpoint remains fine-tunable.
        self.denoising_class_embed = nn.Embedding(config.num_labels + 1, d)

        self.decoder = Decoder(config._transformer_config)

    def __call__(self, pixel_values: mx.array) -> Dict[str, mx.array]:
        """
        Args:
            pixel_values: (B, image_size, image_size, 3) NHWC in [0, 1].
        Returns dict:
            pred_logits: (B, num_queries, num_labels)
            pred_boxes:  (B, num_queries, 4) normalized (cx, cy, w, h) in [0, 1]
            intermediate_logits, intermediate_reference_points: per-layer
              trajectories for training / auxiliary losses
            last_hidden_state: (B, num_queries, d_model)
        """
        enc_features = self.vision(pixel_values)

        proj = [self.decoder_input_proj[i](f) for i, f in enumerate(enc_features)]
        spatial_shapes: List[Tuple[int, int]] = [(f.shape[1], f.shape[2]) for f in proj]
        flat = mx.concatenate(
            [f.reshape(f.shape[0], f.shape[1] * f.shape[2], f.shape[3]) for f in proj],
            axis=1,
        )

        # Encoder query selection: score every position, take top-K.
        anchors, valid_mask = generate_anchors(tuple(spatial_shapes), dtype=flat.dtype)
        memory = flat * valid_mask.astype(flat.dtype)
        output_memory = self.enc_output(memory)
        enc_scores = self.enc_score_head(output_memory)
        enc_coord_logits = self.enc_bbox_head(output_memory) + anchors

        K = self.config.num_queries
        scores_max = enc_scores.max(axis=-1)
        topk_idx = mx.argpartition(-scores_max, K - 1, axis=1)[:, :K]
        topk_scores = mx.take_along_axis(scores_max, topk_idx, axis=1)
        order = mx.argsort(-topk_scores, axis=1)
        topk_idx = mx.take_along_axis(topk_idx, order, axis=1)

        gather_idx_b = mx.broadcast_to(topk_idx[:, :, None], (topk_idx.shape[0], K, 4))
        ref_points_unact = mx.take_along_axis(enc_coord_logits, gather_idx_b, axis=1)
        gather_idx_d = mx.broadcast_to(
            topk_idx[:, :, None], (topk_idx.shape[0], K, output_memory.shape[-1])
        )
        target = mx.stop_gradient(
            mx.take_along_axis(output_memory, gather_idx_d, axis=1)
        )

        dec_out = self.decoder(
            target=target,
            reference_points_unact=ref_points_unact,
            encoder_hidden_states=flat,
            spatial_shapes=tuple(spatial_shapes),
        )

        return {
            "pred_logits": dec_out["intermediate_logits"][:, -1],
            "pred_boxes": dec_out["intermediate_reference_points"][:, -1],
            "intermediate_logits": dec_out["intermediate_logits"],
            "intermediate_reference_points": dec_out["intermediate_reference_points"],
            "last_hidden_state": dec_out["last_hidden_state"],
        }

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Translate HF RT-DETRv2 weight keys to MLX layout.

        Three transformations happen per key:
          1. Name rewrite via the `convert.rename` pipeline (strips the
             `model.` prefix, then applies submodule-specific rules).
          2. Conv2d weight layout transpose: PyTorch (out, in, kH, kW) →
             MLX NHWC (out, kH, kW, in).
          3. Drop list: keys in `DROP_PATTERNS` are excluded. Currently
             this is just `*.num_batches_tracked` — MLX `nn.BatchNorm`
             has no slot for that counter and trainers re-initialise it
             to 0 on resume.
        """
        out: Dict[str, mx.array] = {}
        for k, v in weights.items():
            if _should_drop(k):
                continue
            new_k = _rename(k)
            if new_k.endswith(".conv.weight") and v.ndim == 4:
                v = v.transpose(0, 2, 3, 1)
            out[new_k] = v
        return out

    @staticmethod
    def quant_predicate(path: str, module) -> bool:
        """Layers to quantize for q4/q8 artifacts.

        Skip the backbone (small, accuracy-sensitive) and any conv or
        small embedding modules; quantize Linear layers whose dims are
        multiples of 64.
        """
        if "backbone." in path:
            return False
        if "conv" in path:
            return False
        if any(k in path for k in ("query_embed", "denoising_class_embed")):
            return False
        if hasattr(module, "weight"):
            shape = module.weight.shape
            if any(d % 64 != 0 for d in shape):
                return False
        return isinstance(module, nn.Linear)
