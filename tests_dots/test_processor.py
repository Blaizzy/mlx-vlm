import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.processor import (
    DotsOCRProcessor,
    GroupedBatchPacker,
    MicroBatchPacker,
    OOMBackoffRunner,
    build_cu_seqlens,
)
from mlx_vlm.models.dots_ocr.dots_vision import DotsVisionTransformer_MLX


def test_processor_single_image_shapes():
    cfg = DotsOCRConfig({})
    proc = DotsOCRProcessor(cfg)
    im = Image.fromarray((np.random.rand(333, 517, 3) * 255).astype("uint8"))
    pixels, grid = proc.process_one(im)
    assert pixels.shape[0] == 1 and pixels.shape[1] == 3
    H, W = pixels.shape[-2], pixels.shape[-1]
    assert H % 14 == 0 and W % 14 == 0
    assert grid == [[1, H // 14, W // 14]]

def test_cu_seqlens_and_encode_integration():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    proc = DotsOCRProcessor(cfg)
    im = Image.fromarray((np.random.rand(320, 480, 3) * 255).astype("uint8"))
    pixels, grid = proc.process_one(im)
    cu = build_cu_seqlens(grid)
    assert cu.shape[0] == 2 and int(cu[-1]) == grid[0][1] * grid[0][2]
    model = DotsVisionTransformer_MLX(cfg)
    y = model(pixels, grid)
    expected = (grid[0][1] * grid[0][2]) // (cfg.vision.merge_size**2)
    assert y.shape[0] == expected
    assert y.shape[1] == 1536


def test_microbatch_and_runner_happy_path():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 1}})
    proc = DotsOCRProcessor(cfg)
    ims = [
        Image.fromarray((np.random.rand(200, 300, 3) * 255).astype("uint8")),
        Image.fromarray((np.random.rand(180, 260, 3) * 255).astype("uint8")),
    ]
    processed = proc.process(ims)
    packer = MicroBatchPacker(max_tokens_per_batch=10_000)
    model = DotsVisionTransformer_MLX(cfg)

    def fn(px, gr):
        y = model(px, gr)
        assert y.shape[1] == 1536

    runner = OOMBackoffRunner()
    ok = runner.run(packer, processed, fn)
    assert ok is True


def test_grouped_packer_batches_by_grid():
    cfg = DotsOCRConfig({})
    proc = DotsOCRProcessor(cfg)
    images = [
        Image.fromarray((np.random.rand(300, 420, 3) * 255).astype("uint8")),
        Image.fromarray((np.random.rand(302, 421, 3) * 255).astype("uint8")),
        Image.fromarray((np.random.rand(256, 400, 3) * 255).astype("uint8")),
    ]
    processed = proc.process(images)
    packer = GroupedBatchPacker(max_tokens_per_batch=10_000)
    batches = list(packer.pack(processed))

    assert len(batches) >= 2
    for pixels, grids in batches:
        assert pixels.shape[0] == len(grids)
