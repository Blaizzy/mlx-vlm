"""End-to-end correctness tests for APC exact-mode store/restore with hybrid
attention models (Gemma 4, Qwen 3.5).

These tests exercise the full path: instantiate a hybrid-cache model, prefill,
store via APCManager.store_exact_cache(), restore via lookup_exact_cache(), and
verify the restored cache state is correct.

Unlike test_apc.py (which tests APC mechanics with synthetic arrays), these
tests run real model forward passes to validate the behavioral contract.
"""

from __future__ import annotations

import pytest

from mlx_vlm.apc import model_apc_mode

# ============================================================================
# Model factories — tiny random-weight instances, no downloads needed
# ============================================================================


def _make_tiny_gemma4():
    """Create a tiny Gemma 4 language model with mixed cache types.

    Config adapted from test_models.py::TestModels::test_gemma4 with
    num_hidden_layers bumped to 6 for full sliding_window_pattern coverage.

    sliding_window_pattern=3 → pattern: [sliding, sliding, full] repeated
    With 6 layers: 4 RotatingKVCache + 2 KVCache → triggers exact mode.
    """
    from mlx_vlm.models import gemma4

    text_config = gemma4.TextConfig(
        model_type="gemma4_text",
        hidden_size=32,
        num_hidden_layers=6,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=0,
        sliding_window=32,
        sliding_window_pattern=3,
        final_logit_softcapping=30.0,
    )
    vision_config = gemma4.VisionConfig(
        model_type="gemma4_vision",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        patch_size=16,
        pooling_kernel_size=2,
        default_output_length=4,
        position_embedding_size=64,
        use_clipped_linears=False,
    )
    config = gemma4.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        model_type="gemma4",
        vocab_size=64,
        image_token_id=63,
    )
    model = gemma4.Model(config)
    return model.language_model


def _make_tiny_qwen35():
    """Create a tiny Qwen 3.5 language model with mixed cache types.

    Config adapted from test_models.py::TestModels::test_qwen3_5_decode_uses_rope_deltas_kwarg
    with num_hidden_layers bumped to 4 for full_attention_interval coverage.

    full_attention_interval=4 → 3 out of 4 layers use ArraysCache (linear/SSM),
    1 out of 4 uses KVCache (full attention) → triggers exact mode.
    """
    from mlx_vlm.models import qwen3_5

    text_config = qwen3_5.TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=8,
        full_attention_interval=4,
    )
    config = qwen3_5.ModelConfig(
        text_config=text_config,
        vision_config=qwen3_5.VisionConfig(
            model_type="qwen3_5",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
        ),
        model_type="qwen3_5",
    )
    model = qwen3_5.LanguageModel(text_config, config)
    return model


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_mode_detected_for_hybrid_models(model_factory):
    """Hybrid models must route to exact mode, not block mode."""
    lm = model_factory()
    assert model_apc_mode(lm) == "exact"


# ============================================================================
# Structural invariants — foundations for per-layer hybrid APC
# ============================================================================
