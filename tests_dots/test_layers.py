import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_vision import (
    RMSNorm,
    SwiGLU,
    PatchEmbed,
    VisionAttention,
    apply_rotary,
    build_2d_rotary_cos_sin,
    build_block_mask_from_cu,
)


def test_rmsnorm_shapes_and_numerics():
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    y = RMSNorm(4)(x)
    assert y.shape == (2, 4)
    assert float(mx.max(mx.abs(y))) > 0.0


def test_swish_glu_shapes():
    x = mx.random.uniform(shape=(10, 1536))
    y = SwiGLU()(x)
    assert y.shape == (10, 1536)


def test_patch_embed_shapes_224():
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    tok, Hp, Wp = PatchEmbed()(x)
    assert (Hp, Wp) == (16, 16)
    assert tok.shape == (16 * 16, 1536)


def test_rotary_build_and_apply_shapes():
    H, W = 8, 8
    heads, head_dim = 12, 128
    seq = H * W
    cos, sin = build_2d_rotary_cos_sin(H, W, head_dim // 2)
    assert cos.shape == (seq, head_dim)
    assert sin.shape == (seq, head_dim)

    q = mx.random.uniform(shape=(seq, heads, head_dim))
    k = mx.random.uniform(shape=(seq, heads, head_dim))
    q2, k2 = apply_rotary(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape


def test_attention_shapes_all_allow():
    H, W = 8, 8
    seq = H * W
    heads, dim = 12, 1536
    x = mx.random.uniform(shape=(seq, dim))
    cos, sin = build_2d_rotary_cos_sin(H, W, (dim // heads) // 2)
    mask = mx.ones((seq, seq), dtype=mx.bool_)
    y = VisionAttention(dim, heads)(x, mask, cos, sin)
    assert y.shape == (seq, dim)


def test_attention_block_mask_basic():
    H1, W1 = 8, 8
    H2, W2 = 4, 4
    seq1, seq2 = H1 * W1, H2 * W2
    total = seq1 + seq2
    cu = mx.array([0, seq1, total], dtype=mx.int32)
    mask = build_block_mask_from_cu(cu)
    assert bool(mask[0, 0]) is True
    assert bool(mask[seq1 - 1, seq1]) is False
from mlx_vlm.models.dots_ocr.dots_vision import VisionBlock


def test_vision_block_forward_shape():
    H, W = 8, 8
    seq, dim, heads = H * W, 1536, 12
    x = mx.random.uniform(shape=(seq, dim))
    cos, sin = build_2d_rotary_cos_sin(H, W, (dim // heads) // 2)
    mask = mx.ones((seq, seq), dtype=mx.bool_)
    y = VisionBlock(dim, heads)(x, mask, cos, sin)
    assert y.shape == (seq, dim)
from mlx_vlm.models.dots_ocr.dots_vision import PatchMerger


def test_patch_merger_shapes():
    H, W, D = 16, 16, 1536
    x = mx.random.uniform(shape=(H * W, D))
    y = PatchMerger(D)(x, H, W)
    assert y.shape == ((H // 2) * (W // 2), D)
from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.dots_vision import DotsVisionTransformer_MLX


def test_vision_wrapper_single_image_224():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    model = DotsVisionTransformer_MLX(cfg)
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    grid = [[1, 16, 16]]
    y = model(x, grid)
    assert y.shape == (64, 1536)
