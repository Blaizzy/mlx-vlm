"""PP-DocLayoutV3 detector in MLX.

Backbone (HGNetV2) + hybrid encoder + deformable decoder + reading-order
head. Inference only: full page in (1024x1024, [0,1]), boxes + labels +
reading order out. Mirrors transformers' PP-DocLayoutV3 architecture.
"""

import re
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from ..rt_detr_v2.transformer import inverse_sigmoid
from .backbone import HGNetV2Backbone
from .config import ModelConfig
from .decoder import (
    Decoder,
    GlobalPointer,
    MLPHead,
    decode_order,
    mask_to_box_coordinate,
)
from .encoder import HybridEncoder, Proj

# Rename pipeline shared by Model.sanitize and convert_layout.
RENAME_RULES = [
    (r"^model\.", ""),
    (r"^backbone\.model\.", "backbone."),
    (r"^encoder\.encoder\.", "encoder.aifi."),
    (r"^encoder_input_proj\.(\d+)\.0\.", r"encoder_input_proj.\1.conv."),
    (r"^encoder_input_proj\.(\d+)\.1\.", r"encoder_input_proj.\1.bn."),
    (r"^decoder_input_proj\.(\d+)\.0\.", r"decoder_input_proj.\1.conv."),
    (r"^decoder_input_proj\.(\d+)\.1\.", r"decoder_input_proj.\1.bn."),
    (r"^enc_output\.0\.", "enc_output.fc."),
    (r"^enc_output\.1\.", "enc_output.ln."),
    (r"\.convolution\.", ".conv."),
    (r"\.normalization\.", ".bn."),
    (r"\.norm\.", ".bn."),
]

DROP_PATTERNS = [
    r"\.num_batches_tracked$",
    r"^model\.denoising_class_embed",  # training-only (num_denoising=0)
]


def rename_key(key: str) -> str:
    for pat, repl in RENAME_RULES:
        key = re.sub(pat, repl, key)
    return key


def should_drop(key: str) -> bool:
    return any(re.search(p, key) for p in DROP_PATTERNS)


class EncOutput(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.ln(self.fc(x))


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        eps = config.layer_norm_eps
        self.backbone = HGNetV2Backbone(config.backbone_config, config.batch_norm_eps)
        backbone_outs = [128, 512, 1024, 2048]
        self.encoder_input_proj = [
            Proj(c, config.encoder_hidden_dim, 1, 1, None, config.batch_norm_eps)
            for c in backbone_outs[1:]
        ]
        self.encoder = HybridEncoder(config)
        self.enc_output = EncOutput(d, eps)
        self.enc_score_head = nn.Linear(d, config.num_labels)
        self.enc_bbox_head = MLPHead(d, d, 4, num_layers=3)
        self.decoder_input_proj = [
            Proj(c, d, 1, 1, None, config.batch_norm_eps)
            for c in config.decoder_in_channels
        ]
        self.decoder = Decoder(config, nn.ReLU())
        self.decoder_order_head = [
            nn.Linear(d, d) for _ in range(config.decoder_layers)
        ]
        self.decoder_global_pointer = GlobalPointer(config)
        self.decoder_norm = nn.LayerNorm(d, eps=eps)
        self.mask_query_head = MLPHead(d, d, config.num_prototypes, num_layers=3)

    def __call__(self, pixel_values: mx.array) -> dict:
        from ..rt_detr_v2.transformer import generate_anchors

        cfg = self.config
        feats = self.backbone(pixel_values)
        x4, feats = feats[0], feats[1:]
        proj = [p(f) for p, f in zip(self.encoder_input_proj, feats)]
        pan, mask_feat = self.encoder(proj, x4)

        sources = [p(f) for p, f in zip(self.decoder_input_proj, pan)]
        shapes = tuple((s.shape[1], s.shape[2]) for s in sources)
        flat = mx.concatenate(
            [s.reshape(s.shape[0], -1, s.shape[-1]) for s in sources], axis=1
        )
        anchors, valid_mask = generate_anchors(shapes)
        anchors = anchors.astype(flat.dtype)

        memory = valid_mask.astype(flat.dtype) * flat
        output_memory = self.enc_output(memory)
        enc_class = self.enc_score_head(output_memory)
        enc_coord = self.enc_bbox_head(output_memory) + anchors
        topk = mx.argsort(-enc_class.max(axis=-1), axis=1)[:, : cfg.num_queries]
        B = flat.shape[0]
        gather_idx = mx.broadcast_to(topk[:, :, None], (B, cfg.num_queries, 4))
        ref_unact = mx.take_along_axis(enc_coord, gather_idx, axis=1)
        gather_d = mx.broadcast_to(topk[:, :, None], (B, cfg.num_queries, cfg.d_model))
        target = mx.stop_gradient(mx.take_along_axis(output_memory, gather_d, axis=1))

        if cfg.mask_enhanced:
            # Mask-enhanced init: query masks -> tight boxes -> reference points.
            out_q0 = self.decoder_norm(target)
            mq = self.mask_query_head(out_q0)
            mB, mH, mW = mask_feat.shape[0], mask_feat.shape[1], mask_feat.shape[2]
            enc_masks = (
                mq @ mask_feat.reshape(mB, mH * mW, -1).transpose(0, 2, 1)
            ).reshape(mB, cfg.num_queries, mH, mW)
            ref_unact = inverse_sigmoid(mask_to_box_coordinate(enc_masks > 0))

        dec = self.decoder(
            target,
            ref_unact,
            flat,
            shapes,
            self.enc_bbox_head,
            self.enc_score_head,
            self.decoder_norm,
            self.decoder_order_head,
            self.decoder_global_pointer,
            self.mask_query_head,
            mask_feat,
        )
        return {
            "logits": dec["intermediate_logits"][:, -1],
            "pred_boxes": dec["intermediate_reference_points"][:, -1],
            "order_logits": dec["out_order_logits"][:, -1],
        }

    @staticmethod
    def sanitize(weights: dict) -> dict:
        # Idempotency via key prefix, not shape: a shape heuristic cannot
        # distinguish NCHW from NHWC here (e.g. stem convs are (C,3,3,3),
        # identical under the transpose), so the raw `model.` prefix -- a
        # marker only unconverted checkpoints carry -- is the exact signal.
        # Converted keys are fixpoints of rename_key, hence untouched.
        out = {}
        for k, v in weights.items():
            if should_drop(k):
                continue
            nk = rename_key(k)
            if nk.endswith(".conv.weight") and v.ndim == 4 and nk != k:
                v = v.transpose(0, 2, 3, 1)
            out[nk] = v
        return out

    def detect(
        self,
        image,
        conf: float = 0.5,
        img_size: int = 1024,
    ) -> List[dict]:
        """Boxes in viewer schema: [y0,x0,y1,x1] 0-1000 + label/order/score."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        w0, h0 = image.size
        im = image.convert("RGB").resize((img_size, img_size))
        x = mx.array(np.array(im).astype(np.float32) / 255.0)[None, ...]
        out = self(pixel_values=x)
        mx.eval(out["logits"], out["pred_boxes"], out["order_logits"])
        logits = np.array(out["logits"][0].astype(mx.float32))
        boxes = np.array(out["pred_boxes"][0].astype(mx.float32))

        scores = 1.0 / (1.0 + np.exp(-logits))
        labels = scores.argmax(axis=-1)
        best = scores.max(axis=-1)
        keep = np.nonzero(best > conf)[0]
        if len(keep) == 0:
            return []
        # Reading order over the kept sub-block only.
        sub = out["order_logits"][0][mx.array(keep)][:, mx.array(keep)]
        sequence = np.array(decode_order(sub)).tolist()
        rank = [0] * len(sequence)
        for r, pos in enumerate(sequence):
            rank[pos] = r + 1

        id2label = self.config.id2label or {}
        content = []
        for j, qi in enumerate(keep):
            cx, cy, w, h = (float(v) for v in boxes[qi])
            x0, y0 = cx - w / 2, cy - h / 2
            x1, y1 = cx + w / 2, cy + h / 2
            content.append(
                {
                    "bbox": [
                        round(y0 * 1000, 1),
                        round(x0 * 1000, 1),
                        round(y1 * 1000, 1),
                        round(x1 * 1000, 1),
                    ],
                    "label": id2label.get(int(labels[qi]), str(int(labels[qi]))),
                    "reading_order": rank[j],
                    "score": round(float(best[qi]), 3),
                }
            )
        return content


__all__ = [
    "Model",
    "LayoutModel",
    "EncOutput",
    "rename_key",
    "should_drop",
]


# Backwards-compatible alias.
LayoutModel = Model
