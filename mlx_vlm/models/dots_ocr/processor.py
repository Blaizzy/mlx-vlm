import numpy as np
from PIL import Image
import mlx.core as mx

from .dots_ocr import DotsOCRConfig


class DotsOCRProcessor:
    """
    Minimal, single-image processor:
      - resize to fit within [min_pixels, max_pixels] while preserving aspect ratio
      - normalize by mean/std
      - pad H,W to multiples of patch_size (default 14)
    Returns:
      pixels: mx.array [1,3,H_pad,W_pad]
      grid_thw: [[1, H_pad/patch, W_pad/patch]]
    """

    def __init__(self, cfg: DotsOCRConfig):
        self.cfg = cfg

    @staticmethod
    def _resize_keep_ar(
        im: Image.Image, min_pixels: int, max_pixels: int
    ) -> Image.Image:
        w, h = im.size
        pix = w * h
        scale = 1.0
        if pix < min_pixels:
            scale = (min_pixels / pix) ** 0.5
        elif pix > max_pixels:
            scale = (max_pixels / pix) ** 0.5
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if new_w == w and new_h == h:
            return im
        return im.resize((new_w, new_h), Image.BICUBIC)

    @staticmethod
    def _to_chw_float(arr_hw3: np.ndarray) -> np.ndarray:
        arr = arr_hw3.astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None, ...]

    @staticmethod
    def _pad_to_multiple(x_1chw: np.ndarray, multiple: int) -> np.ndarray:
        _, C, H, W = x_1chw.shape
        Hp = ((H + multiple - 1) // multiple) * multiple
        Wp = ((W + multiple - 1) // multiple) * multiple
        if Hp == H and Wp == W:
            return x_1chw
        out = np.zeros((1, C, Hp, Wp), dtype=x_1chw.dtype)
        out[:, :, :H, :W] = x_1chw
        return out

    def process_one(self, im: Image.Image) -> tuple[mx.array, list[list[int]]]:
        im = im.convert("RGB")
        p = self.cfg.processor
        v = self.cfg.vision

        im = self._resize_keep_ar(im, p.min_pixels, p.max_pixels)
        arr = np.asarray(im, dtype=np.uint8)
        x = self._to_chw_float(arr)

        mean = np.array(p.mean, dtype=np.float32)[:, None, None]
        std = np.array(p.std, dtype=np.float32)[:, None, None]
        x = (x - mean) / std

        x = self._pad_to_multiple(x, v.patch_size)
        H, W = x.shape[-2], x.shape[-1]
        grid_thw = [[1, H // v.patch_size, W // v.patch_size]]

        return mx.array(x), grid_thw

    def process(self, images: list[Image.Image]):
        return [self.process_one(im) for im in images]
