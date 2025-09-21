import os
from collections import defaultdict

import numpy as np
from PIL import Image
import mlx.core as mx

from .dots_ocr import DotsOCRConfig

MAX_DOTS_PIXELS = 11_289_600  # ~11.29M pixels cap from dots.ocr docs
RECOMMENDED_DPI = 200


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


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
        w0, h0 = im.size
        if w0 * h0 > MAX_DOTS_PIXELS * 4:
            scale = (MAX_DOTS_PIXELS * 4 / (w0 * h0)) ** 0.5
            new_w = max(1, int(w0 * scale))
            new_h = max(1, int(h0 * scale))
            if new_w != w0 or new_h != h0:
                im = im.resize((new_w, new_h), Image.BICUBIC)
        p = self.cfg.processor
        v = self.cfg.vision

        im = self._resize_keep_ar(im, p.min_pixels, p.max_pixels)
        arr = np.asarray(im, dtype=np.uint8)
        x = self._to_chw_float(arr)

        mean = np.array(p.mean, dtype=np.float32)[:, None, None]
        std = np.array(p.std, dtype=np.float32)[:, None, None]
        x = (x - mean) / std

        multiple = v.patch_size * v.merge_size
        x = self._pad_to_multiple(x, multiple)
        H, W = x.shape[-2], x.shape[-1]
        assert (
            H * W <= MAX_DOTS_PIXELS
        ), f"pixels after pad exceed cap ({H*W} > {MAX_DOTS_PIXELS}); lower DPI or shrink"
        grid_thw = [[1, (H // v.patch_size), (W // v.patch_size)]]

        return mx.array(x), grid_thw

    def process(self, images: list[Image.Image]):
        return [self.process_one(im) for im in images]


def build_cu_seqlens(grid_thw: list[list[int]]):
    """
    grid_thw: [[1,H',W'], ...]
    returns: mx.array[int32] of shape [B+1], cumulative sums of H'*W'
    """

    cu = [0]
    for _, H, W in grid_thw:
        cu.append(cu[-1] + int(H) * int(W))
    return mx.array(cu, dtype=mx.int32)


class MicroBatchPacker:
    """
    Conservative packer: one image per batch.
    Extend later to group by total tokens if desired.
    """

    def __init__(self, max_tokens_per_batch: int | None = None):
        if max_tokens_per_batch is None:
            max_tokens_per_batch = getenv_int("MLX_MAX_TOKENS", 1_000_000)
        self.max_tokens = max_tokens_per_batch

    def pack(self, processed: list[tuple[mx.array, list[list[int]]]]):
        for pixels, grid in processed:
            HtW = grid[0][1] * grid[0][2]
            if HtW > self.max_tokens:
                raise ValueError(
                    "single image tokens {} exceed max_tokens_per_batch {}".format(
                        HtW, self.max_tokens
                    )
                )
            yield pixels, grid


class OOMBackoffRunner:
    """
    Wrapper to execute a callable on micro-batches; on OOM (RuntimeError),
    reduces max_tokens and retries up to N times.
    """

    def __init__(self, retries: int = 2, factor: float = 0.5):
        self.retries = retries
        self.factor = factor

    def run(self, packer: MicroBatchPacker, processed, fn):
        mt = packer.max_tokens
        for _ in range(self.retries + 1):
            try:
                for px, gr in packer.pack(processed):
                    fn(px, gr)
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and mt > 1024:
                    mt = int(max(1024, mt * self.factor))
                    packer.max_tokens = mt
                else:
                    raise
        return False


class GroupedBatchPacker:
    """
    Packer that groups samples by identical patch grids and batches them together.
    """

    def __init__(self, max_tokens_per_batch: int | None = None):
        if max_tokens_per_batch is None:
            max_tokens_per_batch = getenv_int("MLX_MAX_TOKENS", 1_000_000)
        self.max_tokens = max_tokens_per_batch

    def pack(self, processed: list[tuple[mx.array, list[list[int]]]]):
        buckets: dict[tuple[int, int], list[tuple[mx.array, list[list[int]]]]] = defaultdict(list)
        for pixels, grid in processed:
            Hp, Wp = grid[0][1], grid[0][2]
            buckets[(Hp, Wp)].append((pixels, grid))

        for (Hp, Wp), items in buckets.items():
            cur_pixels: list[mx.array] = []
            cur_grids: list[list[int]] = []
            cur_tokens = 0
            token_per_sample = Hp * Wp

            for pixels, grid in items:
                if cur_tokens > 0 and cur_tokens + token_per_sample > self.max_tokens:
                    yield self._stack(cur_pixels), cur_grids
                    cur_pixels, cur_grids, cur_tokens = [], [], 0

                cur_pixels.append(pixels)
                cur_grids.extend(grid)
                cur_tokens += token_per_sample

            if cur_pixels:
                yield self._stack(cur_pixels), cur_grids

    @staticmethod
    def _stack(pixels_list: list[mx.array]) -> mx.array:
        arrays = [np.array(px) for px in pixels_list]
        stacked = np.concatenate(arrays, axis=0)
        return mx.array(stacked)
