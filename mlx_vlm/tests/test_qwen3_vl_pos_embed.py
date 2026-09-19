"""A quantized Qwen3-VL position table has to interpolate like a float one.

`fast_pos_embed_interpolate` weights the four corner rows of the position table
bilinearly. It built those weights in `pos_embed.weight.dtype`. When a checkpoint
ships `pos_embed` quantized, the loader makes it an `nn.QuantizedEmbedding`, whose
`weight` is the PACKED uint32 array, so every fractional weight became 0 and only
patches that land exactly on a table row kept a position.

The oracle is the same table dequantized to float: a QuantizedEmbedding returns
exactly those rows, so a correct interpolation cannot tell the two apart.
"""

import importlib
import unittest

import mlx.core as mx
import mlx.nn as nn

TOWERS = ["qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe"]


def tiny_tower(name):
    config = importlib.import_module(f"mlx_vlm.models.{name}.config")
    vision = importlib.import_module(f"mlx_vlm.models.{name}.vision")
    # An 8 x 8 position table, 64 wide: two quantization groups of 32 per row.
    cfg = config.VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        out_hidden_size=32,
        num_heads=2,
        patch_size=2,
        spatial_patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    return vision.VisionModel(cfg)


def float_table(weight):
    table = nn.Embedding(weight.shape[0], weight.shape[1])
    table.weight = weight
    return table


class TestQuantizedPosEmbed(unittest.TestCase):
    # 6 x 10 patches on the 8 x 8 table: both axes land between table rows
    # everywhere except at their ends, so only the four corner patches sit
    # exactly on a row.
    grid = mx.array([[1, 6, 10]])

    def test_quantized_table_interpolates_like_its_float_decoding(self):
        for name in TOWERS:
            for dtype in (mx.float32, mx.bfloat16):
                with self.subTest(tower=name, dtype=dtype):
                    model = tiny_tower(name)
                    weight = mx.random.normal((64, 64), key=mx.random.key(164))
                    quantized = nn.QuantizedEmbedding.from_embedding(
                        float_table(weight.astype(dtype)), group_size=32, bits=4
                    )
                    self.assertEqual(quantized.weight.dtype, mx.uint32)
                    decoded = mx.dequantize(
                        quantized.weight,
                        quantized.scales,
                        quantized.biases,
                        group_size=32,
                        bits=4,
                    )

                    model.pos_embed = float_table(decoded)
                    expected = model.fast_pos_embed_interpolate(self.grid)
                    model.pos_embed = quantized
                    actual = model.fast_pos_embed_interpolate(self.grid)

                    self.assertEqual(actual.shape, (60, 64))
                    unplaced = (mx.abs(actual).max(axis=-1) == 0).sum().item()
                    self.assertEqual(
                        unplaced, 0, f"{unplaced} of 60 patches have no position"
                    )
                    worst = mx.abs(
                        actual.astype(mx.float32) - expected.astype(mx.float32)
                    ).max()
                    self.assertLessEqual(worst.item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
