"""Lightweight annotators for SAM3/3.1 detection visualization.

Standalone implementations inspired by the supervision library.
Only depends on numpy and cv2 — no external dependencies.

Usage:
    from mlx_vlm.models.sam3.annotators import (
        BoxAnnotator, BoxCornerAnnotator, RoundBoxAnnotator,
        MaskAnnotator, ColorAnnotator, LabelAnnotator,
        EllipseAnnotator, TriangleAnnotator, DotAnnotator,
        CircleAnnotator, BlurAnnotator, PixelateAnnotator,
        HaloAnnotator, PercentageBarAnnotator, BackgroundOverlayAnnotator,
    )

    result = predictor.predict(image, text_prompt="a cat")
    frame = np.array(image)[..., ::-1]  # RGB->BGR

    annotator = MaskAnnotator(opacity=0.4) + BoxAnnotator() + LabelAnnotator()
    out = annotator.annotate(frame, result)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_COLORS = [
    (47, 255, 173),  # bright green
    (255, 100, 50),  # coral orange
    (50, 150, 255),  # sky blue
    (255, 50, 255),  # hot pink
    (80, 255, 80),  # lime
    (255, 220, 50),  # golden yellow
    (180, 80, 255),  # violet
    (50, 255, 255),  # cyan
    (255, 80, 120),  # rose
    (120, 255, 200),  # mint
]


def _get_color(idx: int, colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return colors[idx % len(colors)]


def _color_idx(result, i: int) -> int:
    """Get stable color index: track_ids > class label hash > detection index."""
    if hasattr(result, "track_ids") and result.track_ids is not None:
        return int(result.track_ids[i])
    # Use class label for stable per-class colors (avoids flickering across frames)
    if (
        hasattr(result, "labels")
        and result.labels is not None
        and i < len(result.labels)
    ):
        label = result.labels[i]
        if isinstance(label, str):
            return hash(label) % 1000
        return int(label)
    return i


def _resize_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask.shape[0] != H or mask.shape[1] != W:
        return cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
    return mask.astype(np.uint8) if mask.dtype != np.uint8 else mask


class BaseAnnotator:
    """Base class for annotators. Supports chaining with +."""

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: "BaseAnnotator") -> "ChainAnnotator":
        items = []
        for a in (self, other):
            if isinstance(a, ChainAnnotator):
                items.extend(a.annotators)
            else:
                items.append(a)
        return ChainAnnotator(items)


class ChainAnnotator(BaseAnnotator):
    """Chains multiple annotators: each annotates in sequence."""

    def __init__(self, annotators: List[BaseAnnotator]):
        self.annotators = annotators

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        for ann in self.annotators:
            scene = ann.annotate(scene, result)
        return scene


@dataclass
class BoxAnnotator(BaseAnnotator):
    """Draw bounding boxes."""

    thickness: int = 2
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.thickness)
        return out


@dataclass
class BoxCornerAnnotator(BaseAnnotator):
    """Draw only the corners of bounding boxes."""

    thickness: int = 2
    corner_length: int = 15
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        cl = self.corner_length
        t = self.thickness
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            # Top-left
            cv2.line(out, (x1, y1), (x1 + cl, y1), color, t)
            cv2.line(out, (x1, y1), (x1, y1 + cl), color, t)
            # Top-right
            cv2.line(out, (x2, y1), (x2 - cl, y1), color, t)
            cv2.line(out, (x2, y1), (x2, y1 + cl), color, t)
            # Bottom-left
            cv2.line(out, (x1, y2), (x1 + cl, y2), color, t)
            cv2.line(out, (x1, y2), (x1, y2 - cl), color, t)
            # Bottom-right
            cv2.line(out, (x2, y2), (x2 - cl, y2), color, t)
            cv2.line(out, (x2, y2), (x2, y2 - cl), color, t)
        return out


@dataclass
class RoundBoxAnnotator(BaseAnnotator):
    """Draw bounding boxes with rounded corners."""

    thickness: int = 2
    radius: int = 10
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        r = self.radius
        t = self.thickness
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            # Edges
            cv2.line(out, (x1 + r, y1), (x2 - r, y1), color, t)
            cv2.line(out, (x1 + r, y2), (x2 - r, y2), color, t)
            cv2.line(out, (x1, y1 + r), (x1, y2 - r), color, t)
            cv2.line(out, (x2, y1 + r), (x2, y2 - r), color, t)
            # Corners
            cv2.ellipse(out, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t)
            cv2.ellipse(out, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t)
            cv2.ellipse(out, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t)
            cv2.ellipse(out, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t)
        return out


@dataclass
class MaskAnnotator(BaseAnnotator):
    """Draw semi-transparent segmentation masks with contour outline."""

    opacity: float = 0.6
    contour_thickness: int = 2
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        if (
            not hasattr(result, "masks")
            or result.masks is None
            or len(result.scores) == 0
        ):
            return scene.copy()

        H, W = scene.shape[:2]
        out = scene.copy()

        # Paint all mask colors onto a single overlay, then blend once with cv2
        overlay = out.copy()
        for i in range(len(result.scores)):
            mask = _resize_mask(result.masks[i], H, W)
            color = _get_color(_color_idx(result, i), self.colors)
            overlay[mask > 0] = color

            if self.contour_thickness > 0:
                # Find contours on downscaled mask for speed, scale back
                scale = max(1, min(H, W) // 360)
                if scale > 1:
                    small = cv2.resize(
                        mask, (W // scale, H // scale), interpolation=cv2.INTER_NEAREST
                    )
                    contours, _ = cv2.findContours(
                        small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contours = [c * scale for c in contours]
                else:
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                cv2.drawContours(out, contours, -1, color, self.contour_thickness)

        # Single cv2 blend (C-optimized, ~0.5ms vs ~9ms for numpy)
        cv2.addWeighted(overlay, self.opacity, out, 1 - self.opacity, 0, out)
        return out


@dataclass
class ColorAnnotator(BaseAnnotator):
    """Fill bounding box regions with semi-transparent color."""

    opacity: float = 0.3
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        overlay = out.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, self.opacity, out, 1 - self.opacity, 0, out)
        return out


@dataclass
class EllipseAnnotator(BaseAnnotator):
    """Draw ellipses at the bottom of bounding boxes (sports/tracking style)."""

    thickness: int = 2
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            cx = (x1 + x2) // 2
            w = (x2 - x1) // 2
            cv2.ellipse(
                out, (cx, y2), (w, max(w // 4, 5)), 0, -180, 0, color, self.thickness
            )
        return out


@dataclass
class CircleAnnotator(BaseAnnotator):
    """Draw circles at detection centers."""

    radius: int = 10
    thickness: int = -1
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = _get_color(_color_idx(result, i), self.colors)
            cv2.circle(out, (cx, cy), self.radius, color, self.thickness)
        return out


@dataclass
class DotAnnotator(BaseAnnotator):
    """Draw small dots at detection centers."""

    radius: int = 4
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = _get_color(_color_idx(result, i), self.colors)
            cv2.circle(out, (cx, cy), self.radius, color, -1)
        return out


@dataclass
class TriangleAnnotator(BaseAnnotator):
    """Draw triangle markers above detection centers."""

    size: int = 16
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        s = self.size
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            cx = (x1 + x2) // 2
            color = _get_color(_color_idx(result, i), self.colors)
            pts = np.array([[cx, y1 + s], [cx - s, y1], [cx + s, y1]], dtype=np.int32)
            cv2.fillPoly(out, [pts], color)
        return out


@dataclass
class LabelAnnotator(BaseAnnotator):
    """Draw text labels with background."""

    font_scale: float = 0.6
    thickness: int = 2
    padding: int = 4
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)
    text_color: Tuple[int, int, int] = (255, 255, 255)

    def annotate(
        self,
        scene: np.ndarray,
        result,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            if labels is not None:
                label = labels[i]
            elif hasattr(result, "labels") and result.labels and i < len(result.labels):
                label = f"{result.labels[i]} {result.scores[i]:.2f}"
            else:
                label = f"{result.scores[i]:.2f}"

            x1, y1 = result.boxes[i][:2].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )
            p = self.padding
            cv2.rectangle(
                out, (x1, max(y1 - th - 2 * p, 0)), (x1 + tw + 2 * p, y1), color, -1
            )
            cv2.putText(
                out,
                label,
                (x1 + p, max(y1 - p, th + p)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
            )
        return out


@dataclass
class PercentageBarAnnotator(BaseAnnotator):
    """Draw confidence score as a horizontal bar inside the bounding box."""

    height: int = 12
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)
    bg_color: Tuple[int, int, int] = (50, 50, 50)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = _get_color(_color_idx(result, i), self.colors)
            bar_y = max(y1 - self.height - 2, 0)
            bar_w = x2 - x1
            fill_w = int(bar_w * result.scores[i])
            cv2.rectangle(
                out, (x1, bar_y), (x2, bar_y + self.height), self.bg_color, -1
            )
            cv2.rectangle(
                out, (x1, bar_y), (x1 + fill_w, bar_y + self.height), color, -1
            )
        return out


@dataclass
class BlurAnnotator(BaseAnnotator):
    """Blur detected regions. Uses mask shape when available, falls back to box.

    Set ``background=True`` to blur the background instead (focus mode).
    """

    kernel_size: int = 31
    background: bool = False

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        H, W = out.shape[:2]
        k = self.kernel_size | 1
        has_masks = hasattr(result, "masks") and result.masks is not None
        blurred = cv2.GaussianBlur(out, (k, k), 0)

        if self.background and has_masks:
            fg = np.zeros((H, W), dtype=bool)
            for i in range(len(result.scores)):
                fg |= _resize_mask(result.masks[i], H, W) > 0
            out[~fg] = blurred[~fg]
        else:
            for i in range(len(result.scores)):
                if has_masks:
                    mask = _resize_mask(result.masks[i], H, W) > 0
                    out[mask] = blurred[mask]
                else:
                    x1, y1, x2, y2 = result.boxes[i].astype(int)
                    out[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
        return out


@dataclass
class PixelateAnnotator(BaseAnnotator):
    """Pixelate detected regions. Uses mask shape when available, falls back to box.

    Set ``background=True`` to pixelate the background instead (focus mode).
    """

    pixel_size: int = 12
    background: bool = False

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        H, W = out.shape[:2]
        ps = self.pixel_size
        has_masks = hasattr(result, "masks") and result.masks is not None
        small = cv2.resize(
            out, (max(W // ps, 1), max(H // ps, 1)), interpolation=cv2.INTER_LINEAR
        )
        pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.background and has_masks:
            fg = np.zeros((H, W), dtype=bool)
            for i in range(len(result.scores)):
                fg |= _resize_mask(result.masks[i], H, W) > 0
            out[~fg] = pixelated[~fg]
        else:
            for i in range(len(result.scores)):
                if has_masks:
                    mask = _resize_mask(result.masks[i], H, W) > 0
                    out[mask] = pixelated[mask]
                else:
                    x1, y1, x2, y2 = result.boxes[i].astype(int)
                    out[y1:y2, x1:x2] = pixelated[y1:y2, x1:x2]
        return out


@dataclass
class HaloAnnotator(BaseAnnotator):
    """Draw glowing halo effect around mask edges."""

    opacity: float = 0.4
    kernel_size: int = 21
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        H, W = out.shape[:2]
        k = self.kernel_size | 1
        for i in range(len(result.scores)):
            if not hasattr(result, "masks") or result.masks is None:
                continue
            mask = _resize_mask(result.masks[i], H, W)
            binary = mask > 0
            blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
            halo = (blurred > 0.01) & ~binary
            color = np.array(
                _get_color(_color_idx(result, i), self.colors), dtype=np.float32
            )
            intensity = blurred[halo].clip(0, 1)
            out[halo] = (
                out[halo].astype(np.float32) * (1 - intensity[:, None] * self.opacity)
                + color * intensity[:, None] * self.opacity
            ).astype(np.uint8)
        return out


@dataclass
class BackgroundOverlayAnnotator(BaseAnnotator):
    """Dim/color the background outside detected masks."""

    opacity: float = 0.5
    color: Tuple[int, int, int] = (0, 0, 0)

    def annotate(self, scene: np.ndarray, result) -> np.ndarray:
        out = scene.copy()
        H, W = out.shape[:2]
        fg = np.zeros((H, W), dtype=bool)
        for i in range(len(result.scores)):
            if not hasattr(result, "masks") or result.masks is None:
                continue
            fg |= _resize_mask(result.masks[i], H, W) > 0
        bg = ~fg
        color_layer = np.full_like(out, self.color)
        # Blend only the background region
        cv2.addWeighted(
            color_layer, self.opacity, out, 1 - self.opacity, 0, color_layer
        )
        out[bg] = color_layer[bg]
        return out
