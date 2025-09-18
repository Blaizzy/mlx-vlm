import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_vision import RMSNorm


def test_rmsnorm_shapes_and_numerics():
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    y = RMSNorm(4)(x)
    assert y.shape == (2, 4)
    # Check finite and roughly scaled
    assert float(mx.max(mx.abs(y))) > 0.0
from mlx_vlm.models.dots_ocr.dots_vision import SwiGLU


def test_swish_glu_shapes():
    x = mx.random.uniform(shape=(10, 1536))
    y = SwiGLU()(x)
    assert y.shape == (10, 1536)
from mlx_vlm.models.dots_ocr.dots_vision import PatchEmbed


def test_patch_embed_shapes_224():
    x = mx.random.uniform(shape=(1, 3, 224, 224))  # divisible by 14
    tok, Hp, Wp = PatchEmbed()(x)
    assert (Hp, Wp) == (16, 16)
    assert tok.shape == (16 * 16, 1536)
from mlx_vlm.models.dots_ocr.dots_vision import build_2d_rotary_cos_sin, apply_rotary


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
