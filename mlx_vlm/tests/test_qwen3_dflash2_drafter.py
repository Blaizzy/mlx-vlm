import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.speculative.drafters.qwen3_dflash import Model, ModelConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
    CandidateSelector,
    DFlash2DecoderLayer,
    DFlashDecoderLayer,
)

HIDDEN = 64
LAYERS = 2
VOCAB = 128
GROUP = 16
RANK = 8
TOP_K = 4


def _config_dict(dflash_extra=None):
    dflash_config = {
        "block_size": 8,
        "mask_token_id": 120,
        "target_layer_ids": [0, 1, 2, 3, 4],
    }
    dflash_config.update(dflash_extra or {})
    return {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "num_target_layers": 8,
        "max_position_embeddings": 4096,
        "layer_types": ["sliding_attention"] * LAYERS,
        "sliding_window": 32,
        "rope_parameters": {"rope_theta": 1234567.0, "rope_type": "default"},
        "dflash_config": dflash_config,
    }


def _v2_config_dict():
    return _config_dict(
        {
            "conv_kernel_size": 2,
            "conv_group_size": GROUP,
            "selector_rank": RANK,
            "selector_top_k": TOP_K,
        }
    )


def test_v1_config_is_unchanged_and_disables_the_v2_modules():
    config = ModelConfig.from_dict(_config_dict())

    assert config.conv_kernel_size == 0
    assert config.conv_group_size == 0
    assert config.selector_rank == 0
    assert config.selector_top_k == 0

    model = Model(config)
    assert model.candidate_selector is None
    assert all(isinstance(layer, DFlashDecoderLayer) for layer in model.layers)
    assert not any(isinstance(layer, DFlash2DecoderLayer) for layer in model.layers)


def test_v2_config_reads_the_dflash_block():
    config = ModelConfig.from_dict(_v2_config_dict())

    assert config.block_size == 8  # DFlash 2 nests this inside dflash_config
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == GROUP
    assert config.selector_rank == RANK
    assert config.selector_top_k == TOP_K
    # transformers 5.x nests RoPE settings, which must not fall back to the default
    assert config.rope_theta == 1234567.0


def test_v2_model_parameters_match_the_checkpoint_layout():
    model = Model(ModelConfig.from_dict(_v2_config_dict()))
    params = dict(tree_flatten(model.parameters()))

    groups = HIDDEN // GROUP
    expected = {
        "candidate_selector.hidden_projection.weight": (RANK, HIDDEN),
        "candidate_selector.predecessor_codebook.weight": (VOCAB, RANK),
        "candidate_selector.successor_codebook.weight": (VOCAB, RANK),
    }
    for layer in range(LAYERS):
        for conv in ("attention_conv", "mlp_conv"):
            expected[f"layers.{layer}.{conv}.base_kernel"] = (2, 2, HIDDEN)
            expected[f"layers.{layer}.{conv}.kernel_projection.weight"] = (
                2 * 2 * groups,
                HIDDEN,
            )

    for name, shape in expected.items():
        assert name in params, f"missing parameter {name}"
        assert tuple(params[name].shape) == shape, name


def test_sanitize_adds_the_weight_suffix_to_the_codebooks():
    model = Model(ModelConfig.from_dict(_v2_config_dict()))
    raw = {
        "model.candidate_selector.predecessor_codebook": mx.zeros((VOCAB, RANK)),
        "candidate_selector.successor_codebook": mx.zeros((VOCAB, RANK)),
        "candidate_selector.hidden_projection.weight": mx.zeros((RANK, HIDDEN)),
    }
    out = model.sanitize(raw)

    assert "candidate_selector.predecessor_codebook.weight" in out
    assert "candidate_selector.successor_codebook.weight" in out
    assert "candidate_selector.hidden_projection.weight" in out


def test_selector_falls_back_to_argmax_without_edge_scores():
    config = ModelConfig.from_dict(_v2_config_dict())
    selector = CandidateSelector(config)
    # Zeroed codebooks remove the pairwise term, so the path must follow the
    # per-position argmax of the logits.
    selector.predecessor_codebook.weight = mx.zeros((VOCAB, RANK))
    selector.successor_codebook.weight = mx.zeros((VOCAB, RANK))

    block = 3
    logits = mx.random.normal((1, block, VOCAB))
    hidden = mx.random.normal((1, block, HIDDEN))
    path = selector.select(hidden, logits, mx.array([0]))

    assert path.shape == (1, block)
    assert mx.array_equal(path, mx.argmax(logits, axis=-1))


@pytest.mark.parametrize("kernel_size", [1, 2])
def test_grouped_dynamic_conv_is_causal(kernel_size):
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
        GroupedDynamicCausalConv,
    )

    conv = GroupedDynamicCausalConv(HIDDEN, kernel_size, GROUP)
    conv.base_kernel = mx.random.normal((2, kernel_size, HIDDEN)) * 0.1

    x = mx.random.normal((1, 6, HIDDEN))
    out, _ = conv.prepare(x)

    # Changing a later position must not alter earlier outputs.
    x2 = mx.array(x)
    x2[:, 4:] = mx.random.normal((1, 2, HIDDEN))
    out2, _ = conv.prepare(x2)

    assert mx.allclose(out[:, :4], out2[:, :4], atol=1e-5)
