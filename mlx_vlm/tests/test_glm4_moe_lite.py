import mlx.core as mx

from mlx_vlm.models import base
from mlx_vlm.models.glm4_moe_lite import Model, ModelConfig, language
from mlx_vlm.turboquant import BatchTurboQuantKVCache
from mlx_vlm.utils import get_model_and_args


def _tiny_config():
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=None,
        n_routed_experts=None,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        num_nextn_predict_layers=0,
    )


def test_glm4_moe_lite_routes_to_native_mlx_vlm_model():
    architecture, model_type = get_model_and_args({"model_type": "glm4_moe_lite"})

    assert model_type == "glm4_moe_lite"
    assert architecture.Model is Model
    assert language.scaled_dot_product_attention is base.scaled_dot_product_attention


def test_glm4_moe_lite_batch_turboquant_prefill_and_decode():
    model = Model(_tiny_config())
    cache = [BatchTurboQuantKVCache([0], bits=3.5)]

    prefill = model(mx.array([[1, 2, 3]]), cache=cache)
    decode = model(mx.array([[4]]), cache=cache)
    mx.eval(prefill.logits, decode.logits, cache[0].keys, cache[0].values)

    assert prefill.logits.shape == (1, 3, 32)
    assert decode.logits.shape == (1, 1, 32)
    assert cache[0].key_codec.dim == 8
    assert cache[0].value_codec.dim == 4
    assert cache[0]._idx == 4
