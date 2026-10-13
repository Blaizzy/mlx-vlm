"""Numerical regression tests for LFM2.5 MoE routing and cached inference."""

from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.lfm2.language import Lfm2MoeSparseMoeBlock
from mlx_vlm.models.lfm2.speculative_verifier import Lfm2ExactSpeculativeVerifier
from mlx_vlm.models.lfm2_moe import Model, ModelConfig


def tiny_config(**kwargs):
    config = dict(
        model_type="lfm2_moe",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128000,
        use_expert_bias=True,
        num_dense_layers=1,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
    )
    return ModelConfig(**(config | kwargs))


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_router_matches_explicit_expert_sum(use_bias, normalize):
    mx.random.seed(7)
    config = tiny_config(
        use_expert_bias=use_bias,
        norm_topk_prob=normalize,
        routed_scaling_factor=1.7,
    )
    block = Lfm2MoeSparseMoeBlock(config)
    if use_bias:
        block.expert_bias = mx.array([0.7, -0.4, 0.2, -0.1])
    x = mx.random.normal((2, 3, config.hidden_size))
    actual = np.array(block(x))
    values = np.array(x)
    logits = values @ np.array(block.gate.weight).T
    probabilities = 1 / (1 + np.exp(-logits))
    selection = probabilities + (np.array(block.expert_bias) if use_bias else 0)
    indices = np.argsort(selection, axis=-1)[..., -config.num_experts_per_tok :]
    scores = np.take_along_axis(probabilities, indices, axis=-1)
    if normalize:
        scores /= scores.sum(axis=-1, keepdims=True) + 1e-6
    scores *= config.routed_scaling_factor
    weights = block.switch_mlp
    gate, up, down = (
        np.array(getattr(weights, name).weight)
        for name in ("gate_proj", "up_proj", "down_proj")
    )
    expected = np.zeros_like(values)
    for batch in range(values.shape[0]):
        for token in range(values.shape[1]):
            for slot, expert in enumerate(indices[batch, token]):
                v = values[batch, token]
                g = gate[expert] @ v
                result = down[expert] @ ((g / (1 + np.exp(-g))) * (up[expert] @ v))
                expected[batch, token] += scores[batch, token, slot] * result
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    verified = Lfm2ExactSpeculativeVerifier()._feed_forward(block, x)
    np.testing.assert_allclose(np.array(verified), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_chunked_prefill_and_decode_match_full_forward(batch_size):
    mx.random.seed(13)
    model = Model(tiny_config())
    model.eval()
    tokens = mx.random.randint(0, 128, (batch_size, 13))
    full = model(tokens).logits
    cache = model.make_cache()
    chunks = []
    for start, end in ((0, 5), (5, 9), (9, 12), (12, 13)):
        chunks.append(model(tokens[:, start:end], cache=cache).logits)
        mx.eval(chunks[-1], [c.state for c in cache])
    np.testing.assert_allclose(
        np.array(mx.concatenate(chunks, axis=1)), np.array(full), atol=1e-5, rtol=1e-4
    )
    last = model(tokens, logits_to_keep=1).logits
    np.testing.assert_allclose(np.array(last), np.array(full[:, -1:]), atol=1e-5)


def test_transformers_numerical_parity():
    torch = pytest.importorskip("torch")
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    torch.manual_seed(42)
    config = tiny_config(routed_scaling_factor=1.3)
    reference = Lfm2MoeForCausalLM(Lfm2MoeConfig(**asdict(config))).eval()
    # Nonzero balancing biases catch the old use of biased output weights.
    with torch.no_grad():
        for layer in reference.model.layers[1:]:
            layer.feed_forward.expert_bias.copy_(torch.tensor([0.4, -0.3, 0.2, -0.1]))
    model = Model(config)
    weights = {
        k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    tokens = torch.tensor([[1, 7, 11, 2, 3, 9], [4, 5, 13, 6, 2, 1]])
    with torch.no_grad():
        expected = reference(tokens, use_cache=False).logits.numpy()
    actual = np.array(model(mx.array(tokens.numpy())).logits)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
