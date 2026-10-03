"""YOLO11 detection model for MLX.

Ports the Ultralytics YOLO11 detection architecture (C3k2 backbone, SPPF,
C2PSA attention neck, anchor-free Detect head with DFL box decoding) to MLX,
handling the tensor layout difference between PyTorch (B, C, H, W) and
MLX (B, H, W, C).

The reference checkpoint is microsoft/OmniParser-v2.0 ``icon_detect/model.pt``
(nc=1, "icon" class), used as the icon detector of the OmniParser pipeline.

Reference: ultralytics/ultralytics/nn/modules/{block,conv,head}.py
"""

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU (or identity)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
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

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, g=g)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k(nn.Module):
    """CSP Bottleneck with 3 convolutions (C3 with 3x3 kernels)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # Python list of modules; weights are loaded by index (digit keys).
        self.m = [Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)]

    def __call__(self, x):
        b, h, w, _ = x.shape
        feat = self.m[0](self.cv1(x))
        for m in self.m[1:]:
            feat = m(feat)
        return self.cv3(mx.concatenate([feat, self.cv2(x)], axis=-1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (C3k2 container).

    In YOLO11 the inner blocks are C3k (``c3k=True``); the ``m`` list holds
    one C3k per repeat.
    """

    def __init__(self, c1, c2, n=1, c3k=True, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = [
            (
                C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else Bottleneck(self.c, self.c, shortcut, g)
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        y = list(mx.split(self.cv1(x), 2, axis=-1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(mx.concatenate(y, axis=-1))


# Alias matching the Ultralytics module name used in the checkpoint.
C3k2 = C2f


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (3 sequential max-pools)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))


class Attention(nn.Module):
    """Position-sensitive multi-head attention over HxW tokens.

    NHWC port of ultralytics nn.modules.block.Attention.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def __call__(self, x):
        b, h, w, c = x.shape
        n = h * w
        heads, kd, hd = self.num_heads, self.key_dim, self.head_dim

        qkv = self.qkv(x)  # (B, H, W, heads*(2*kd+hd))
        # (B, N, heads, 2kd+hd) then per-head q, k, v
        qkv = qkv.reshape(b, n, heads, 2 * kd + hd)
        q, k, v = mx.split(qkv, [kd, 2 * kd], axis=-1)
        # -> (B, heads, N, d)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # attn[b,h,i,j] = softmax_j( (q_i . k_j) * scale )
        attn = mx.matmul(q * self.scale, mx.swapaxes(k, -1, -2))  # (B,heads,N,N)
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(mx.swapaxes(v, -1, -2), mx.swapaxes(attn, -1, -2))
        out = mx.swapaxes(out, -1, -2)  # (B,heads,N,hd)

        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, h, w, c)
        v_spatial = mx.transpose(v, (0, 2, 1, 3)).reshape(b, h, w, c)
        return self.proj(out + self.pe(v_spatial))


class PSABlock(nn.Module):
    """Attention + FFN block with residual connections."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def __call__(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """CSP module with position-sensitive attention (YOLO11 neck block)."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        heads = max(c1 // 128, 1)  # ultralytics: max(c // 64, 1) with c = c1/2
        self.m = [PSABlock(self.c, attn_ratio=0.5, num_heads=heads) for _ in range(n)]

    def __call__(self, x):
        a, b = mx.split(self.cv1(x), 2, axis=-1)
        feat = self.m[0](b)
        for m in self.m[1:]:
            feat = m(feat)
        return self.cv2(mx.concatenate([a, feat], axis=-1))


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


class DFL(nn.Module):
    """Distribution Focal Loss: converts 16-bin distribution to 4 distances."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # Matches ultralytics DFL.conv.weight (fixed [0..reg_max-1]).
        self.weight = mx.arange(reg_max, dtype=mx.float32).reshape(1, 1, 1, -1)

    def __call__(self, x):
        # x: (B, 4*reg_max, n_anchors)
        b, c, n = x.shape
        x = x.transpose(0, 2, 1).reshape(b * n, 4, self.reg_max)
        x = mx.softmax(x, axis=-1)
        # Weighted sum over reg_max -> (B*n, 4, 1)
        x = mx.matmul(x, self.weight.reshape(-1, 1))
        return x.reshape(b, n, 4).transpose(0, 2, 1)


class Detect(nn.Module):
    """YOLO11 detection head: anchor-free, light classification branch."""

    def __init__(self, nc=1, ch=(256, 512, 512), reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        c2 = max(ch[0] // 4, 16, reg_max * 4)  # 64

        # Box branch: Conv3x3 -> Conv3x3 -> Conv2d1x1 (per scale).
        self.cv2 = [
            nn.Sequential(
                Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, reg_max * 4, 1)
            )
            for c in ch
        ]
        # Classification branch (DW head): (DWConv -> Conv) x2 -> Conv2d1x1.
        c3 = max(min(256, nc * 400), 256)  # 256 for the reference checkpoint
        self.cv3 = [
            nn.Sequential(
                nn.Sequential(
                    Conv(c, c, 3, g=c),  # DWConv
                    Conv(c, c3, 1),
                ),
                nn.Sequential(
                    Conv(c3, c3, 3, g=c3),  # DWConv
                    Conv(c3, c3, 1),
                ),
                nn.Conv2d(c3, nc, 1),
            )
            for c in ch
        ]

        self.dfl = DFL(reg_max)

    def __call__(self, xs):
        """Args: xs = [P3, P4, P5] feature maps, NHWC.
        Returns: (B, 4 + nc, N) decoded xywh boxes (input pixels) + sigmoid scores.
        """
        cv2 = self.cv2
        cv3 = self.cv3
        strides = [8, 16, 32]

        boxes, scores, anchors, stride_tensor = [], [], [], []
        for i, x in enumerate(xs):
            b, h, w, _ = x.shape
            bx = cv2[i](x)  # (B, H, W, 64)
            sc = cv3[i](x)  # (B, H, W, nc)
            boxes.append(bx.reshape(b, h * w, self.reg_max * 4).transpose(0, 2, 1))
            scores.append(sc.reshape(b, h * w, self.nc).transpose(0, 2, 1))
            # Anchor points in grid coordinates, row-major over (H, W).
            sx = mx.arange(w, dtype=mx.float32) + 0.5
            sy = mx.arange(h, dtype=mx.float32) + 0.5
            gy, gx = mx.meshgrid(sy, sx, indexing="ij")
            anchors.append(mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1))
            stride_tensor.append(mx.full((h * w, 1), float(strides[i])))

        boxes = mx.concatenate(boxes, axis=-1)  # (B, 64, N)
        scores = mx.concatenate(scores, axis=-1)  # (B, nc, N)
        anchor_points = mx.concatenate(anchors, axis=0)  # (N, 2)
        strides_t = mx.concatenate(stride_tensor, axis=0)  # (N, 1)

        # DFL decode: (B, 64, N) -> (B, 4, N) ltrb distances in grid units.
        dist = self.dfl(boxes)
        # dist2bbox (xywh) + stride scaling -> pixel coordinates.
        anc = anchor_points.T.reshape(1, 2, -1)  # (1, 2, N)
        lt, rb = dist[:, :2, :], dist[:, 2:, :]
        x1y1 = anc - lt
        x2y2 = anc + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        dbox = mx.concatenate([c_xy, wh], axis=1) * strides_t.T.reshape(1, 1, -1)

        return mx.concatenate([dbox, mx.sigmoid(scores)], axis=1)  # (B, 4+nc, N)


# ---------------------------------------------------------------------------
# Full YOLO11 model (OmniParser icon_detect topology)
# ---------------------------------------------------------------------------


class YOLO11(nn.Module):
    """YOLO11 detection model for OmniParser's icon_detect module.

    Layer table mirrors microsoft/OmniParser-v2.0 icon_detect ``model.pt``
    (all C3k2 blocks have n=1 with C3k inner blocks).
    """

    LAYERS = [
        # (from, module, args)
        (-1, "conv", {"c1": 3, "c2": 64, "k": 3, "s": 2}),  # 0
        (-1, "conv", {"c1": 64, "c2": 128, "k": 3, "s": 2}),  # 1
        (-1, "c3k2", {"c1": 128, "c2": 256, "n": 1, "e": 0.25}),  # 2
        (-1, "conv", {"c1": 256, "c2": 256, "k": 3, "s": 2}),  # 3
        (-1, "c3k2", {"c1": 256, "c2": 512, "n": 1, "e": 0.25}),  # 4
        (-1, "conv", {"c1": 512, "c2": 512, "k": 3, "s": 2}),  # 5
        (-1, "c3k2", {"c1": 512, "c2": 512, "n": 1, "e": 0.5}),  # 6
        (-1, "conv", {"c1": 512, "c2": 512, "k": 3, "s": 2}),  # 7
        (-1, "c3k2", {"c1": 512, "c2": 512, "n": 1, "e": 0.5}),  # 8
        (-1, "sppf", {"c1": 512, "c2": 512, "k": 5}),  # 9
        (-1, "c2psa", {"c1": 512, "c2": 512, "n": 1, "e": 0.5}),  # 10
        (-1, "upsample", {}),  # 11
        ([-1, 6], "concat", {}),  # 12
        (-1, "c3k2", {"c1": 1024, "c2": 512, "n": 1, "e": 0.5}),  # 13
        (-1, "upsample", {}),  # 14
        ([-1, 4], "concat", {}),  # 15
        (-1, "c3k2", {"c1": 1024, "c2": 256, "n": 1, "e": 0.5}),  # 16  P3/8
        (-1, "conv", {"c1": 256, "c2": 256, "k": 3, "s": 2}),  # 17
        ([-1, 13], "concat", {}),  # 18
        (-1, "c3k2", {"c1": 768, "c2": 512, "n": 1, "e": 0.5}),  # 19  P4/16
        (-1, "conv", {"c1": 512, "c2": 512, "k": 3, "s": 2}),  # 20
        ([-1, 10], "concat", {}),  # 21
        (-1, "c3k2", {"c1": 1024, "c2": 512, "n": 1, "e": 0.5}),  # 22  P5/32
    ]

    DETECT_FROM = [16, 19, 22]  # P3/8, P4/16, P5/32

    def __init__(self, nc=1, ch=(256, 512, 512), reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        # Python list of layers; weights load by index (mirrors nn.Sequential).
        self.layers = []
        for _, module_name, args in self.LAYERS:
            if module_name == "conv":
                self.layers.append(Conv(**args))
            elif module_name == "c3k2":
                self.layers.append(C3k2(**args))
            elif module_name == "sppf":
                self.layers.append(SPPF(**args))
            elif module_name == "c2psa":
                self.layers.append(C2PSA(**args))
            elif module_name == "upsample":
                self.layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            elif module_name == "concat":
                self.layers.append(None)  # no parameters
            else:
                raise ValueError(f"Unknown module: {module_name}")

        self.detect = Detect(nc=nc, ch=ch, reg_max=reg_max)

    def __call__(self, x):
        """Args: x: (B, H, W, 3) float32, RGB in [0, 1].
        Returns: (B, 4 + nc, N) decoded boxes (xywh, input pixels) + scores.
        """
        feats = []
        for i, (from_idx, module_name, _) in enumerate(self.LAYERS):
            if isinstance(from_idx, list):
                out = mx.concatenate([feats[j] for j in from_idx], axis=-1)
            elif from_idx == -1:
                inp = feats[-1] if feats else x
                out = self.layers[i](inp)
            else:
                out = self.layers[i](feats[from_idx])
            feats.append(out)

        return self.detect([feats[i] for i in self.DETECT_FROM])


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (pure MLX)
# ---------------------------------------------------------------------------


def box_iou(box1, box2):
    """IoU between two sets of xyxy boxes. Args: (N, 4), (M, 4). Returns (N, M)."""
    x1 = mx.maximum(box1[:, 0:1], box2[:, 0].reshape(1, -1))
    y1 = mx.maximum(box1[:, 1:2], box2[:, 1].reshape(1, -1))
    x2 = mx.minimum(box1[:, 2:3], box2[:, 2].reshape(1, -1))
    y2 = mx.minimum(box1[:, 3:4], box2[:, 3].reshape(1, -1))
    inter = mx.maximum(x2 - x1, 0) * mx.maximum(y2 - y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1.reshape(-1, 1) + area2.reshape(1, -1) - inter
    return inter / mx.maximum(union, 1e-7)


def xywh2xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return mx.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def non_max_suppression(
    prediction, conf_thresh=0.05, iou_thresh=0.1, max_det=300, max_nms=30000
):
    """Greedy NMS on YOLO11 output.

    Args:
        prediction: (B, 4 + nc, N) xywh boxes (pixels) + class scores.
    Returns:
        list of (n, 6) arrays per image: [x1, y1, x2, y2, score, class_id].
    """
    output = []
    for i in range(prediction.shape[0]):
        pred = prediction[i].T  # (N, 4+nc)
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        max_scores = mx.max(class_scores, axis=1)
        class_ids = mx.argmax(class_scores, axis=1)

        candidate_count = int(mx.sum(max_scores > conf_thresh).item())
        if candidate_count == 0:
            output.append(mx.zeros((0, 6)))
            continue
        topk_idx = mx.argsort(max_scores)[::-1][: min(candidate_count, max_nms)]
        boxes_xyxy = xywh2xyxy(mx.take(boxes_xywh, topk_idx, axis=0))
        max_scores = mx.take(max_scores, topk_idx, axis=0)
        class_ids = mx.take(class_ids, topk_idx, axis=0)
        offsets = class_ids.reshape(-1, 1) * (mx.max(mx.abs(boxes_xyxy)) + 1)
        nms_boxes = boxes_xyxy + offsets

        # Sequential greedy suppression (small arrays; host-side loop).
        keep = []
        indices = list(range(boxes_xyxy.shape[0]))
        while indices and len(keep) < max_det:
            idx = indices[0]
            keep.append(idx)
            if len(indices) == 1:
                break
            ious = box_iou(
                nms_boxes[idx : idx + 1],
                mx.take(nms_boxes, mx.array(indices[1:]), axis=0),
            ).squeeze(0)
            mask = [int(m) for m in (ious < iou_thresh)]
            indices = [j for j, m in zip(indices[1:], mask) if m]

        keep_arr = mx.array(keep)
        result = mx.concatenate(
            [
                mx.take(boxes_xyxy, keep_arr, axis=0),
                mx.take(max_scores, keep_arr, axis=0).reshape(-1, 1),
                mx.take(class_ids, keep_arr, axis=0).reshape(-1, 1),
            ],
            axis=-1,
        )
        output.append(result)
    return output


# ---------------------------------------------------------------------------
# Weight loading (Ultralytics PyTorch state dict -> MLX module tree)
# ---------------------------------------------------------------------------


def load_weights(model, mlx_weights, prefix="model."):
    """Load an MLX-layout Ultralytics state dict into a YOLO11 model.

    Expects conv weights already transposed to (O, H, W, I); see convert.py.
    Puts the model in eval mode so BatchNorm uses running statistics.
    """
    model.eval()
    layer_weights = {}
    for key, val in mlx_weights.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        subkey = ".".join(parts[1:])
        layer_weights.setdefault(layer_idx, {})[subkey] = val

    for layer_idx, weights in layer_weights.items():
        if layer_idx < len(model.layers):
            module = model.layers[layer_idx]
            if module is None:
                if weights:
                    raise ValueError(f"unexpected weights for concat layer {layer_idx}")
                continue
            _load_into(module, weights)
        elif layer_idx == len(model.layers):
            _load_into(model.detect, weights)
        else:
            raise ValueError(f"layer index {layer_idx} out of range")


def _load_into(module, weights):
    """Set attributes following dot-separated keys.

    Digit segments index Python lists directly; MLX ``nn.Sequential`` names
    its children "0", "1", ... so ``getattr`` covers both cases.
    """

    def resolve(obj, name):
        if isinstance(obj, list):
            return obj[int(name)]
        layers = getattr(obj, "layers", None)  # MLX Sequential
        if layers is not None and name.isdigit():
            return layers[int(name)]
        return getattr(obj, name)

    for key, val in weights.items():
        attrs = key.split(".")
        obj = module
        for attr in attrs[:-1]:
            obj = resolve(obj, attr)
        setattr(obj, attrs[-1], val)
