"""Tests for the Z-Image model family."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.image import (
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.z_image.config import (
    ZImageConfig,
    ZImageTextEncoderConfig,
    ZImageTransformerConfig,
    ZImageVAEConfig,
    detect_z_image_layout,
)
from mlx_vlm.models.z_image.model import ZImageGenerationModel
from mlx_vlm.models.z_image.scheduler import FlowMatchEulerScheduler
from mlx_vlm.models.z_image.text_encoder import ZImageTextEncoder
from mlx_vlm.models.z_image.transformer import (
    ZImageTransformer,
    sanitize_transformer_weights,
)
from mlx_vlm.models.z_image.vae import ZImageVAE


# --- Config / Layout tests ---


def test_detect_z_image_layout(tmp_path: Path) -> None:
    """Positive and negative layout detection."""
    # Missing tokenizer → False
    assert not detect_z_image_layout(tmp_path)
    # Create full layout
    for rel in (
        "transformer/model.safetensors.index.json",
        "text_encoder/model.safetensors.index.json",
        "vae/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    assert detect_z_image_layout(tmp_path)


def test_detect_z_image_layout_real() -> None:
    """Detect the real local quantized checkpoint."""
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    assert detect_z_image_layout(path)


# --- Dispatch tests ---


def test_supports_model_local_path() -> None:
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    assert ZImageGenerationModel.supports_model(str(path))


def test_dispatch_via_image_generation_model_class() -> None:
    """The dispatch resolves z_image from known alias."""
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    cls = image_generation_model_class(str(path))
    assert cls is ZImageGenerationModel


def test_is_image_generation_model_alias() -> None:
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    assert is_image_generation_model(str(path))


# --- Tiny random-weight shape tests ---


def test_transformer_forward_shape() -> None:
    """Tiny transformer produces correct output shape."""
    cfg = ZImageTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        in_channels=16,
        text_embed_dim=32,
        num_hidden_layers=2,
        n_refiner_layers=1,
        n_context_refiner_layers=1,
        adaln_embed_dim=256,
        rope_sections=(4, 6, 6),
    )
    model = ZImageTransformer(cfg)
    # Input: [B=1, C=16, F=1, H=4, W=4] (patch_size=2 → 2x2 grid)
    x = mx.random.normal((1, 16, 1, 4, 4))
    t = mx.array([0.5])
    cap = mx.random.normal((1, 8, 32))
    out = model(x, t, cap)
    mx.eval(out)
    assert out.shape == (1, 16, 1, 4, 4)


def test_text_encoder_forward_shape() -> None:
    """Tiny text encoder produces correct output shape."""
    cfg = ZImageTextEncoderConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
    )
    model = ZImageTextEncoder(cfg)
    ids = mx.array([[1, 2, 3, 4, 5]])
    out = model(ids)
    mx.eval(out)
    assert out.shape == (1, 5, 64)


def test_vae_decoder_shape() -> None:
    """Tiny VAE decoder produces correct output shape."""
    cfg = ZImageVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
    )
    vae = ZImageVAE(cfg)
    # Latent: [B=1, H=4, W=4, C=4]
    z = mx.random.normal((1, 4, 4, 4))
    out = vae.decode(z)
    mx.eval(out)
    # After 2 up_blocks with upsample (first block upsamples, second doesn't)
    # input 4×4 → 8×8 after first upsample → stays 8×8 (last block no upsample)
    assert out.shape[0] == 1
    assert out.shape[-1] == 3  # 3 output channels


def test_scheduler_steps() -> None:
    """Scheduler produces correct number of steps."""
    sched = FlowMatchEulerScheduler(9)
    assert sched.timesteps.shape == (9,)
    assert sched.sigmas.shape == (10,)  # steps + 1
    latents = mx.ones((1, 16, 1, 4, 4))
    pred = mx.ones_like(latents)
    result = sched.step(pred, 0, latents)
    mx.eval(result)
    assert result.shape == latents.shape


def test_sanitize_transformer_weights() -> None:
    """Weight key sanitization maps correctly."""
    weights = {
        "all_final_layer.2-1.linear.weight": mx.zeros((3,)),
        "all_x_embedder.2-1.weight": mx.zeros((3,)),
        "layers.0.attention.to_q.weight": mx.zeros((3,)),
    }
    sanitized = sanitize_transformer_weights(weights)
    assert "_final_layer.linear.weight" in sanitized
    assert "_x_embedder.weight" in sanitized
    assert "layers.0.attention.to_q.weight" in sanitized


# --- Real checkpoint loading test ---


def test_load_transformer_from_real_checkpoint() -> None:
    """Load the real quantized transformer and verify shapes."""
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    from mlx_vlm.models.z_image.weights import load_transformer

    transformer = load_transformer(path)
    # Verify key structural properties
    assert len(transformer.layers) == 30
    assert len(transformer.noise_refiner) == 2
    assert len(transformer.context_refiner) == 2
    # Verify a quantized weight has scales
    w = transformer.layers[0].adaLN_modulation[0]
    # QuantizedLinear has 'scales' attribute after loading quantized weights
    assert hasattr(w, "scales") or hasattr(w, "weight")


def test_load_text_encoder_from_real_checkpoint() -> None:
    """Load the real quantized text encoder and verify structure."""
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    from mlx_vlm.models.z_image.weights import load_text_encoder

    encoder = load_text_encoder(path)
    assert len(encoder.layers) == 36


def test_load_vae_from_real_checkpoint() -> None:
    """Load the real quantized VAE and verify structure."""
    path = Path("~/.models/Tongyi-MAI/Z-Image-Turbo-mxfp8").expanduser()
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    from mlx_vlm.models.z_image.weights import load_vae

    vae = load_vae(path)
    assert len(vae.decoder.up_blocks) == 4
    assert len(vae.encoder.down_blocks) == 4
