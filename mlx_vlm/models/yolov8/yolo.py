"""YOLOv8 detection model for MLX.

Ports the Ultralytics YOLOv8n architecture to MLX, handling the tensor layout
difference between PyTorch (B, C, H, W) and MLX (B, H, W, C).

Reference: ultralytics/ultralytics/nn/modules/block.py, head.py, tasks.py
"""

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU (or identity)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        pad = p if p is not None else k // 2
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=pad, groups=g, bias=False
        )
        self.bn = nn.BatchNorm(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def __call__(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck with optional residual connection."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = [
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        ]

    def __call__(self, x):
        y = list(mx.split(self.cv1(x), 2, axis=-1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(mx.concatenate(y, axis=-1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (3 sequential max-pools)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))


class DFL(nn.Module):
    """Distribution Focal Loss: converts 16-bin distribution to 4 distances."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # Fixed weights [0, 1, ..., reg_max-1].
        self.weight = mx.arange(reg_max, dtype=mx.float32)

    def __call__(self, x):
        # x: (B, 4*reg_max, n_anchors)
        b, c, n = x.shape
        # Reshape to (B*4, n, reg_max), softmax over reg_max, dot with weights.
        x = x.reshape(b * 4, n, self.reg_max)
        x = mx.softmax(x, axis=2)
        # Weighted sum: (B*4, n, 1) = (B*4, n, reg_max) @ (reg_max, 1).
        x = mx.matmul(x, self.weight.reshape(-1, 1))
        return x.reshape(b, 4, n)


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


class Detect(nn.Module):
    """YOLOv8 detection head (anchor-free)."""

    def __init__(self, nc=80, ch=(64, 128, 256), reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # 80 + 64 = 144

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        # Box regression branches (one per scale). Use named attributes
        # because MLX has no ModuleList.
        self.cv2_0 = nn.Sequential(
            Conv(ch[0], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, reg_max * 4, 1)
        )
        self.cv2_1 = nn.Sequential(
            Conv(ch[1], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, reg_max * 4, 1)
        )
        self.cv2_2 = nn.Sequential(
            Conv(ch[2], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, reg_max * 4, 1)
        )
        # Classification branches (one per scale).
        self.cv3_0 = nn.Sequential(
            Conv(ch[0], c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)
        )
        self.cv3_1 = nn.Sequential(
            Conv(ch[1], c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)
        )
        self.cv3_2 = nn.Sequential(
            Conv(ch[2], c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)
        )

        self.dfl = DFL(reg_max)

    def __call__(self, xs):
        """
        Args:
            xs: list of 3 feature maps [(B, H8, W8, C3), (B, H16, W16, C4), (B, H32, W32, C5)]
        Returns:
            (B, 8400, 84) — 4 box coords + 80 class scores.
        """
        boxes = []
        scores = []
        cv2_list = [self.cv2_0, self.cv2_1, self.cv2_2]
        cv3_list = [self.cv3_0, self.cv3_1, self.cv3_2]
        for i, x in enumerate(xs):
            bx = cv2_list[i](x)  # (B, H, W, 64)
            sc = cv3_list[i](x)  # (B, H, W, 80)
            b, h, w, _ = bx.shape
            n_anchors = h * w
            # Flatten spatial dims: (B, n_anchors, C) then transpose to (B, C, n_anchors).
            boxes.append(bx.reshape(b, n_anchors, self.reg_max * 4).transpose(0, 2, 1))
            scores.append(sc.reshape(b, n_anchors, self.nc).transpose(0, 2, 1))

        boxes = mx.concatenate(boxes, axis=-1)  # (B, 64, 8400)
        scores = mx.concatenate(scores, axis=-1)  # (B, 80, 8400)

        # DFL decoding: (B, 64, 8400) -> (B, 4, 8400).
        dfl_out = self.dfl(boxes)
        return dfl_out, scores


# ---------------------------------------------------------------------------
# Full YOLOv8 model
# ---------------------------------------------------------------------------


class YOLOv8(nn.Module):
    """YOLOv8n detection model (backbone + FPN/PANet neck + Detect head).

    For OmniParser's ``icon_detect`` module.
    """

    # Layer spec: (from, module, args)
    #   from: -1 = previous layer, or list of layer indices to concatenate.
    #   args: constructor kwargs (channel args only; repeats handled in __init__).
    LAYERS = [
        # Backbone
        (-1, "conv", {"c1": 3, "c2": 16, "k": 3, "s": 2}),  # 0
        (-1, "conv", {"c1": 16, "c2": 32, "k": 3, "s": 2}),  # 1
        (-1, "c2f", {"c1": 32, "c2": 32, "n": 1, "shortcut": True}),  # 2
        (-1, "conv", {"c1": 32, "c2": 64, "k": 3, "s": 2}),  # 3
        (-1, "c2f", {"c1": 64, "c2": 64, "n": 2, "shortcut": True}),  # 4  P3
        (-1, "conv", {"c1": 64, "c2": 128, "k": 3, "s": 2}),  # 5
        (-1, "c2f", {"c1": 128, "c2": 128, "n": 2, "shortcut": True}),  # 6  P4
        (-1, "conv", {"c1": 128, "c2": 256, "k": 3, "s": 2}),  # 7
        (-1, "c2f", {"c1": 256, "c2": 256, "n": 1, "shortcut": True}),  # 8  P5
        (-1, "sppf", {"c1": 256, "c2": 256, "k": 5}),  # 9
        # FPN neck (top-down)
        (-1, "upsample", {}),  # 10
        ([-1, 6], "concat", {}),  # 11
        (-1, "c2f", {"c1": 384, "c2": 128, "n": 2, "shortcut": False}),  # 12
        (-1, "upsample", {}),  # 13
        ([-1, 4], "concat", {}),  # 14
        (-1, "c2f", {"c1": 192, "c2": 64, "n": 2, "shortcut": False}),  # 15  P3/8 out
        # PANet neck (bottom-up)
        (-1, "conv", {"c1": 64, "c2": 64, "k": 3, "s": 2}),  # 16
        ([-1, 12], "concat", {}),  # 17
        (-1, "c2f", {"c1": 192, "c2": 128, "n": 2, "shortcut": False}),  # 18  P4/16 out
        (-1, "conv", {"c1": 128, "c2": 128, "k": 3, "s": 2}),  # 19
        ([-1, 9], "concat", {}),  # 20
        (-1, "c2f", {"c1": 384, "c2": 256, "n": 2, "shortcut": False}),  # 21  P5/32 out
    ]

    # Detection scales: (layer_index, stride).
    DETECT_SCALES = [(15, 8), (18, 16), (21, 32)]

    def __init__(self, nc=80, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        # Build layers as a Python list (MLX has no ModuleList).
        self.layers = []
        for from_idx, module_name, args in self.LAYERS:
            if module_name == "conv":
                self.layers.append(Conv(**args))
            elif module_name == "c2f":
                self.layers.append(C2f(**args))
            elif module_name == "sppf":
                self.layers.append(SPPF(**args))
            elif module_name == "upsample":
                self.layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            elif module_name == "concat":
                self.layers.append(None)  # no parameters
            else:
                raise ValueError(f"Unknown module: {module_name}")

        # Detection head.
        self.detect = Detect(nc=nc, ch=(64, 128, 256), reg_max=reg_max)

    def __call__(self, x):
        """
        Args:
            x: (B, H, W, 3) float32, pixel values in [0, 1].
        Returns:
            dfl_out: (B, 4, 8400) decoded distances.
            scores:  (B, 80, 8400) raw class logits.
        """
        feats = []
        for i, (from_idx, module_name, _) in enumerate(self.LAYERS):
            if isinstance(from_idx, list):
                # Concat: gather inputs from specified layer indices.
                inp = [feats[j] for j in from_idx]
                out = mx.concatenate(inp, axis=-1)
            elif from_idx == -1:
                inp = feats[-1] if feats else x
                out = self.layers[i](inp)
            else:
                out = self.layers[i](feats[from_idx])
            feats.append(out)

        # Collect detection scale outputs.
        scale_feats = [feats[idx] for idx, _ in self.DETECT_SCALES]
        return self.detect(scale_feats)


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors for a list of feature maps.

    Args:
        feats: list of (B, H, W, C) feature maps.
        strides: list of int stride values.
    Returns:
        anchor_points: (8400, 2) float32
        stride_tensor: (8400, 1) float32
    """
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        b, h, w, _ = feat.shape
        sx = mx.arange(w, dtype=mx.float32) + grid_cell_offset
        sy = mx.arange(h, dtype=mx.float32) + grid_cell_offset
        sy, sx = mx.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(mx.stack([sx.reshape(-1), sy.reshape(-1)], axis=-1))
        stride_tensor.append(mx.full((h * w, 1), stride, dtype=mx.float32))
    return mx.concatenate(anchor_points, axis=0), mx.concatenate(stride_tensor, axis=0)


def dist2bbox(distance, anchor_points, xywh=True):
    """Decode predicted distances (ltrb) to bounding boxes (xywh or xyxy).

    Args:
        distance: (B, 4, n_anchors) — left, top, right, bottom distances.
        anchor_points: (n_anchors, 2) — center coordinates.
        xywh: if True, return (cx, cy, w, h); if False, return (x1, y1, x2, y2).
    Returns:
        (B, 4, n_anchors) bounding boxes.
    """
    lt, rb = distance[:, :2], distance[:, 2:]  # (B, 2, N)
    # anchor_points: (N, 2) -> (1, 2, N) for broadcasting.
    anc = anchor_points.T.reshape(1, 2, -1)  # (1, 2, N)
    x1y1 = anc - lt  # (B, 2, N)
    x2y2 = anc + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return mx.concatenate([c_xy, wh], axis=1)
    return mx.concatenate([x1y1, x2y2], axis=1)


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (pure MLX)
# ---------------------------------------------------------------------------


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes (xyxy format).

    Args:
        box1: (N, 4) — xyxy
        box2: (M, 4) — xyxy
    Returns:
        (N, M) IoU matrix.
    """
    x1 = mx.maximum(box1[:, 0:1], box2[:, 0].reshape(1, -1))
    y1 = mx.maximum(box1[:, 1:2], box2[:, 1].reshape(1, -1))
    x2 = mx.minimum(box1[:, 2:3], box2[:, 2].reshape(1, -1))
    y2 = mx.minimum(box1[:, 3:4], box2[:, 3].reshape(1, -1))
    inter = mx.maximum(x2 - x1, 0) * mx.maximum(y2 - y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1.reshape(-1, 1) + area2.reshape(1, -1) - inter
    return inter / mx.maximum(union, 1e-7)


def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45, max_det=300):
    """Apply NMS to YOLOv8 output.

    Args:
        prediction: (B, 84, 8400) — 4 box coords + 80 class scores.
        conf_thresh: minimum confidence to keep a detection.
        iou_thresh: IoU threshold for NMS.
        max_det: maximum detections per image.
    Returns:
        list of (N, 6) tensors, each row: [x1, y1, x2, y2, score, class_id].
    """
    bs = prediction.shape[0]
    output = []

    for i in range(bs):
        pred = prediction[i].T  # (8400, 84)

        # Split boxes and scores.
        boxes_xywh = pred[:, :4]  # (8400, 4)
        class_scores = pred[:, 4:]  # (8400, 80)

        # Max class score and class id.
        max_scores = mx.max(class_scores, axis=1)  # (8400,)
        class_ids = mx.argmax(class_scores, axis=1)  # (8400,)

        # Filter by confidence. Sort by score, take top-k, then threshold.
        k = min(int(mx.sum(max_scores > conf_thresh)), max_det)
        if k == 0:
            output.append(mx.zeros((0, 6)))
            continue
        topk_idx = mx.argsort(max_scores)[::-1][:k]
        boxes_xywh = mx.take(boxes_xywh, topk_idx, axis=0)
        max_scores = mx.take(max_scores, topk_idx, axis=0)
        class_ids = mx.take(class_ids, topk_idx, axis=0)

        if boxes_xywh.shape[0] == 0:
            output.append(mx.zeros((0, 6)))
            continue

        # Convert xywh to xyxy.
        cx, cy, w, h = (
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = mx.stack([x1, y1, x2, y2], axis=-1)

        # Sort by score (descending).
        order = mx.argsort(max_scores)[::-1]
        boxes_xyxy = mx.take(boxes_xyxy, order, axis=0)
        max_scores = mx.take(max_scores, order, axis=0)
        class_ids = mx.take(class_ids, order, axis=0)

        # Greedy NMS.
        keep = []
        indices = mx.arange(boxes_xyxy.shape[0])
        while indices.shape[0] > 0 and len(keep) < max_det:
            idx = indices[0]
            keep.append(int(idx))
            if indices.shape[0] == 1:
                break

            remaining = indices[1:]
            ious = box_iou(
                mx.take(boxes_xyxy, idx.reshape(1), axis=0),
                mx.take(boxes_xyxy, remaining, axis=0),
            ).squeeze(0)
            # Keep boxes that are NOT overlapping beyond the threshold.
            keep_mask = ious < iou_thresh
            # Convert to Python for filtering (NMS loop is sequential, small arrays).
            remaining_py = [int(x) for x in remaining]
            mask_py = [int(x) for x in keep_mask]
            indices = mx.array([r for r, m in zip(remaining_py, mask_py) if m])

        keep_arr = mx.array(keep)
        scores = mx.take(max_scores, keep_arr, axis=0)
        cls = mx.take(class_ids, keep_arr, axis=0)
        boxes = mx.take(boxes_xyxy, keep_arr, axis=0)
        result = mx.concatenate(
            [boxes, scores.reshape(-1, 1), cls.reshape(-1, 1)], axis=-1
        )
        output.append(result)

    return output


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def sanitize_yolo(weights_file):
    """Convert PyTorch YOLOv8 state dict to MLX format.

    Transposes Conv2d weight tensors from (O, I, H, W) to (O, H, W, I).
    BatchNorm running stats are kept as-is (1-D, no transpose needed).
    """
    mlx_weights = {}
    with safe_open(weights_file, framework="pt") as st:
        for key in st.keys():
            tensor = st.get_tensor(key)
            if tensor.ndim == 4:
                tensor = tensor.permute(0, 2, 3, 1)
            mlx_weights[key] = mx.array(tensor.numpy())
    return mlx_weights


def load_weights(model, mlx_weights, prefix="model."):
    """Load sanitized weights into a YOLOv8 model.

    Maps PyTorch key names (model.{i}.{submodule}) to the MLX module tree.
    Handles Conv, BatchNorm, and Conv2d (in Detect head) weight loading.
    """
    # Group weights by layer index.
    layer_weights = {}
    for key, val in mlx_weights.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        subkey = ".".join(parts[1:])
        if layer_idx not in layer_weights:
            layer_weights[layer_idx] = {}
        layer_weights[layer_idx][subkey] = val

    # Load into model layers.
    for layer_idx, weights in layer_weights.items():
        if layer_idx < len(model.layers):
            module = model.layers[layer_idx]
            if module is None:
                continue  # concat layers have no weights
            # Build state dict from subkeys.
            state = {}
            for k, v in weights.items():
                # Convert flat key path to nested attribute access.
                # e.g., "cv1.conv.weight" -> module.cv1.conv.weight = v
                attrs = k.split(".")
                obj = module
                for attr in attrs[:-1]:
                    if attr.isdigit():
                        obj = obj[int(attr)]
                    else:
                        obj = getattr(obj, attr)
                setattr(obj, attrs[-1], v)
        elif layer_idx == 22:
            # Detect head.
            _load_detect_weights(model.detect, weights)


def _load_detect_weights(detect, weights):
    """Load weights into the Detect head module."""
    for key, val in weights.items():
        parts = key.split(".")
        obj = detect
        for attr in parts[:-1]:
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        setattr(obj, parts[-1], val)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = YOLOv8(nc=80)
    x = mx.random.normal((1, 640, 640, 3))
    dfl_out, scores = model(x)
    print(f"DFL output shape: {dfl_out.shape}")  # (1, 4, 8400)
    print(f"Scores shape: {scores.shape}")  # (1, 80, 8400)

    # Anchor generation.
    feats = [
        mx.zeros((1, 80, 80, 64)),
        mx.zeros((1, 40, 40, 128)),
        mx.zeros((1, 20, 20, 256)),
    ]
    anchors, strides = make_anchors(feats, [8, 16, 32])
    print(f"Anchors shape: {anchors.shape}")  # (8400, 2)
    print(f"Strides shape: {strides.shape}")  # (8400, 1)

    # Decode boxes.
    decoded = dist2bbox(dfl_out, anchors)
    print(f"Decoded boxes shape: {decoded.shape}")  # (1, 4, 8400)

    # Full pipeline: decode + scores.
    cls_scores = mx.sigmoid(scores)
    pred = mx.concatenate([decoded, cls_scores], axis=1)  # (1, 84, 8400)
    print(f"Full prediction shape: {pred.shape}")  # (1, 84, 8400)

    # NMS.
    detections = non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45)
    print(
        f"NMS output: {len(detections)} images, first has {detections[0].shape[0]} detections"
    )
