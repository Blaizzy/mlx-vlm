import importlib
import inspect
import threading
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map


class TestModels(unittest.TestCase):
    def language_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(model.model_type, model_type)
        self.assertEqual(len(model.layers), num_layers)

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 2, vocab_size))
            self.assertEqual(logits.dtype, t)

            outputs = model(mx.argmax(logits[0, -1:, :], keepdims=True), cache=None)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 1, vocab_size))
            self.assertEqual(logits.dtype, t)

    def mm_projector_test_runner(
        self, mm_projector, vision_hidden_size, text_hidden_size
    ):

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            mm_projector.update(
                tree_map(lambda p: p.astype(t), mm_projector.parameters())
            )

            vision_features = mx.random.uniform(
                shape=(batch_size, vision_hidden_size), dtype=t
            )
            input_tensor = mx.array(vision_features)

            outputs = mm_projector(input_tensor)
            self.assertEqual(outputs.shape, (batch_size, text_hidden_size))
            self.assertEqual(outputs.dtype, t)

    def vision_test_runner(
        self,
        vision_tower,
        model_type,
        vision_hidden_size,
        num_channels,
        image_size: tuple,
        vision_feature_layer=-2,
        channel_first=False,
        **kwargs,
    ):
        for t in [mx.float32, mx.float16]:
            vision_tower.update(
                tree_map(lambda p: p.astype(t), vision_tower.parameters())
            )
            self.assertEqual(vision_tower.model_type, model_type)

            if model_type == "llama4_vision_model":
                vision_hidden_size = kwargs.pop(
                    "projector_output_dim", vision_hidden_size
                )

            batch_size = kwargs.pop("batch_size", 1)
            if model_type in [
                "qwen2_5_vl",
                "glm4v_moe",
                "glm4v",
                "hunyuan_vl",
                "siglip2_vision_model",
            ]:
                input_tensor = mx.random.uniform(shape=(image_size[0], image_size[1]))
            else:
                shape = (
                    (batch_size, num_channels, image_size[0], image_size[1])
                    if channel_first
                    else (batch_size, image_size[0], image_size[1], num_channels)
                )
                input_tensor = mx.random.uniform(shape=shape)

            if "image_masks" in inspect.signature(vision_tower.__call__).parameters:
                input_tensor = input_tensor.transpose(0, 3, 1, 2)
                image_masks = mx.ones((batch_size, num_channels, image_size[0]))
                kwargs["image_masks"] = image_masks

            input_tensor = input_tensor.astype(t)

            if (
                "output_hidden_states"
                in inspect.signature(vision_tower.__call__).parameters
            ):
                hidden_states = vision_tower(
                    input_tensor, output_hidden_states=True, **kwargs
                )
            else:
                hidden_states = vision_tower(input_tensor, **kwargs)

            hidden_states = hidden_states[vision_feature_layer]

            # Check vision hidden feature layer's shape matches the expected hidden size
            if channel_first:
                if model_type == "llama4_vision_model":
                    self.assertEqual(hidden_states.shape[1], vision_hidden_size)
                else:
                    self.assertEqual(hidden_states.shape[1], vision_hidden_size)
            else:
                self.assertEqual(hidden_states.shape[-1], vision_hidden_size)

            self.assertEqual(hidden_states.dtype, t)

    def test_laguna_language_model(self):
        from mlx_vlm.models import laguna

        config = laguna.ModelConfig(
            model_type="laguna",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            layer_types=["full_attention", "sliding_attention"],
            num_attention_heads_per_layer=[4, 4],
            sliding_window=8,
            mlp_layer_types=["dense", "sparse"],
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        )

        model = laguna.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "KVCache")
        self.assertEqual(type(cache[1]).__name__, "RotatingKVCache")

    def test_hrm_text_language_model(self):
        from mlx_vlm.models import hrm_text

        config = hrm_text.ModelConfig(
            model_type="hrm_text",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=128,
            H_cycles=2,
            L_cycles=2,
            embedding_scale=1.0,
        )
        model = hrm_text.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(len(cache), config.num_hidden_layers)
        self.assertEqual(type(cache[0]).__name__, "KVCache")

        fused = {
            "model.L_module.layers.0.attn.gqkv_proj.weight": mx.zeros((128, 32)),
            "model.L_module.layers.0.mlp.gate_up_proj.weight": mx.zeros((128, 32)),
        }
        sanitized = model.sanitize(fused)
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.self_attn.gate_proj.weight"
            ].shape,
            (32, 32),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.self_attn.q_proj.weight"
            ].shape,
            (32, 32),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.self_attn.k_proj.weight"
            ].shape,
            (32, 32),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.self_attn.v_proj.weight"
            ].shape,
            (32, 32),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.mlp.gate_proj.weight"
            ].shape,
            (64, 32),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.L_module.layers.0.mlp.up_proj.weight"
            ].shape,
            (64, 32),
        )

    def test_lfm2_language_model(self):
        from mlx_vlm.models import lfm2

        config = lfm2.ModelConfig(
            model_type="lfm2",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            block_dim=64,
            block_ff_dim=128,
            block_multiple_of=16,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=True,
            layer_types=["conv", "full_attention"],
            tie_word_embeddings=True,
        )

        model = lfm2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "ArraysCache")
        self.assertEqual(type(cache[1]).__name__, "KVCache")

    def test_lfm2_moe_language_model(self):
        from mlx_vlm.models import lfm2_moe

        config = lfm2_moe.ModelConfig(
            model_type="lfm2_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_expert_bias=True,
            num_dense_layers=1,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            layer_types=["conv", "full_attention"],
            tie_word_embeddings=True,
        )

        model = lfm2_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "ArraysCache")
        self.assertEqual(type(cache[1]).__name__, "KVCache")

    def test_cohere2_moe_language_model(self):
        from mlx_vlm.models import cohere2_moe

        config = cohere2_moe.ModelConfig(
            model_type="cohere2_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_norm_eps=1e-5,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            moe_num_shared_experts=1,
            logit_scale=1.0,
        )

        model = cohere2_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "RotatingKVCache")
        self.assertEqual(type(cache[1]).__name__, "KVCache")

    def test_lfm2_moe_sanitize_stacks_experts(self):
        from mlx_vlm.models import lfm2_moe

        config = lfm2_moe.ModelConfig(
            model_type="lfm2_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=True,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_expert_bias=True,
            num_dense_layers=1,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            layer_types=["conv", "full_attention"],
        )
        model = lfm2_moe.Model(config)
        prefix = "model.layers.1.feed_forward"
        weights = {
            f"{prefix}.experts.0.w1.weight": mx.ones((32, 64)),
            f"{prefix}.experts.1.w1.weight": mx.zeros((32, 64)),
            f"{prefix}.experts.0.w2.weight": mx.ones((64, 32)),
            f"{prefix}.experts.1.w2.weight": mx.zeros((64, 32)),
            f"{prefix}.experts.0.w3.weight": mx.ones((32, 64)),
            f"{prefix}.experts.1.w3.weight": mx.zeros((32, 64)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn(f"language_model.{prefix}.switch_mlp.gate_proj.weight", sanitized)
        self.assertIn(f"language_model.{prefix}.switch_mlp.down_proj.weight", sanitized)
        self.assertIn(f"language_model.{prefix}.switch_mlp.up_proj.weight", sanitized)
        self.assertEqual(
            sanitized[f"language_model.{prefix}.switch_mlp.gate_proj.weight"].shape,
            (2, 32, 64),
        )
        self.assertNotIn(f"language_model.{prefix}.experts.0.w1.weight", sanitized)

    def test_lfm2_vl_projector_creates_layernorm_only_when_enabled(self):
        from mlx_vlm.models import lfm2_vl
        from mlx_vlm.models.lfm2_vl.lfm2_vl import Lfm2VlMultiModalProjector

        def make_projector(use_layernorm):
            config = lfm2_vl.ModelConfig(
                model_type="lfm2-vl",
                text_config=lfm2_vl.TextConfig(layer_types=["full_attention"]),
                vision_config=lfm2_vl.VisionConfig(),
                projector_use_layernorm=use_layernorm,
            )
            return config, Lfm2VlMultiModalProjector(config)

        in_channels = 768 * (2**2)
        x = mx.zeros((1, 4, in_channels))

        # LFM2-VL checkpoints omit the flag (defaults True) and ship
        # layer_norm weights, so the module must exist.
        config, projector = make_projector(True)
        self.assertIn("layer_norm", projector.parameters())
        self.assertEqual(projector(x).shape, (1, 4, config.text_config.hidden_size))

        # LFM2.5-VL checkpoints set the flag to False and ship no
        # layer_norm weights; strict loading must not require them.
        config, projector = make_projector(False)
        self.assertNotIn("layer_norm", projector.parameters())
        self.assertEqual(projector(x).shape, (1, 4, config.text_config.hidden_size))

    def test_deepseek_v4_language_model(self):
        from mlx_vlm.models import deepseek_v4
        from mlx_vlm.models.deepseek_v4.hyper_connection import (
            _hc_split_sinkhorn_ops,
            hc_expand,
        )
        from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE

        rope = DeepseekV4RoPE(4, 10000)
        x = mx.random.uniform(shape=(1, 2, 3, 4))
        y = rope(x, offset=1)
        y_inv = rope(y, offset=1, inverse=True)
        self.assertTrue(mx.allclose(y_inv, x, rtol=1e-5, atol=1e-5))

        mixes = mx.random.normal((2, 3, 8), dtype=mx.float32)
        scale = mx.array([1.2, 0.7, 1.1], dtype=mx.float32)
        base = mx.random.normal((8,), dtype=mx.float32)
        _, _, comb = _hc_split_sinkhorn_ops(mixes, scale, base, 2, 20, 1e-6)
        self.assertTrue(mx.allclose(comb.sum(-1), mx.ones_like(comb.sum(-1)), atol=0.1))

        post = mx.random.normal((2, 3, 2), dtype=mx.float32)
        block_out = mx.random.normal((2, 3, 8), dtype=mx.bfloat16)
        comb = mx.random.normal((2, 3, 2, 2), dtype=mx.float32)
        residual = mx.random.normal((2, 3, 2, 8), dtype=mx.bfloat16)
        expected = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
        expected = expected + mx.matmul(
            comb.swapaxes(-1, -2), residual.astype(mx.float32)
        )
        actual = hc_expand(block_out, residual, post, comb)
        self.assertTrue(mx.allclose(actual, expected.astype(block_out.dtype)))

        config = deepseek_v4.ModelConfig(
            model_type="deepseek_v4",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            q_lora_rank=16,
            o_lora_rank=8,
            o_groups=2,
            head_dim=16,
            qk_rope_head_dim=4,
            sliding_window=16,
            compress_ratios=[0, 0, 4, 0],
            index_n_heads=4,
            index_head_dim=8,
            index_topk=4,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=1,
            hc_mult=2,
            hc_sinkhorn_iters=2,
        )
        loaded_config = deepseek_v4.ModelConfig.from_dict(
            {"model_type": "deepseek_v4", "eos_token_id": 1}
        )
        self.assertEqual(loaded_config.eos_token_id, 1)

        model = deepseek_v4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )
        inputs = mx.array([[1, 2, 3]])
        inputs_embeds = model.language_model.model.embed_tokens(inputs)
        out = model.language_model(inputs, inputs_embeds=inputs_embeds)
        self.assertEqual(out.logits.shape, (1, 3, config.vocab_size))

        cache = model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "RotatingKVCache")
        self.assertEqual(type(cache[2]).__name__, "CacheList")

        weight = mx.to_fp8(mx.ones((128, 128), dtype=mx.float32))
        converted = model.sanitize(
            {
                "layers.0.attn.wkv.weight": weight,
                "layers.0.attn.wkv.scale": mx.full((1, 1), 127, dtype=mx.uint8),
            }
        )
        wkey = "language_model.model.layers.0.attn.wkv.weight"
        skey = "language_model.model.layers.0.attn.wkv.scales"
        self.assertIn(wkey, converted)
        self.assertIn(skey, converted)
        self.assertTrue(mx.all(converted[wkey] == weight.view(mx.uint32)))
        self.assertEqual(converted[skey].shape, (128, 4))

    def test_glm_moe_dsa_language_model(self):
        from mlx_vlm.models import glm_moe_dsa

        config = glm_moe_dsa.ModelConfig(
            model_type="glm_moe_dsa",
            vocab_size=1024,
            hidden_size=128,
            index_head_dim=16,
            index_n_heads=4,
            index_topk=4,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=24,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            moe_layer_freq=1,
            first_k_dense_replace=1,
            max_position_embeddings=1024,
            rms_norm_eps=1e-5,
            rope_parameters={"rope_theta": 10000.0},
            attention_bias=False,
            index_topk_pattern="FSFSFS",
        )
        self.assertEqual(
            config.indexer_types,
            ["full", "shared", "full", "shared", "full", "shared"],
        )

        model = glm_moe_dsa.Model(config)
        has_indexer = [
            layer.self_attn.indexer is not None
            for layer in model.language_model.model.layers
        ]
        self.assertEqual(has_indexer, [True, False, True, False, True, False])

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        cache = model.make_cache()
        self.assertEqual(len(cache[0].caches), 2)
        self.assertEqual(len(cache[1].caches), 1)

        sanitized = model.sanitize(
            {
                "model.embed_tokens.weight": mx.zeros((config.vocab_size, 128)),
                "lm_head.weight": mx.zeros((config.vocab_size, 128)),
            }
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)

        prefixed = {"language_model.lm_head.weight": mx.zeros((config.vocab_size, 128))}
        self.assertIs(model.sanitize(prefixed), prefixed)

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits = model(prompt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 8, config.vocab_size))

        nxt = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(nxt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 1, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())
        mx.eval([c.state for c in cache])

    def test_nemotron_h_language_model(self):
        from mlx_vlm.models import nemotron_h

        config = nemotron_h.ModelConfig(
            model_type="nemotron_h",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            max_position_embeddings=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
            mamba_num_heads=4,
            mamba_head_dim=16,
            mamba_proj_bias=False,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=2,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hybrid_override_pattern=["M", "*", "-", "E"],
            moe_intermediate_size=32,
            n_group=1,
            n_routed_experts=4,
            n_shared_experts=1,
            moe_shared_expert_intermediate_size=32,
            topk_group=1,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )

        model = nemotron_h.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        cache = model.make_cache()
        self.assertEqual(len(cache), 2)
        self.assertEqual(type(cache[0]).__name__, "ArraysCache")
        self.assertEqual(type(cache[1]).__name__, "KVCache")

        sanitized = model.sanitize(
            {"backbone.embeddings.weight": mx.zeros((config.vocab_size, 64))}
        )
        self.assertIn("language_model.backbone.embeddings.weight", sanitized)

        prefixed = {
            "language_model.backbone.embeddings.weight": mx.zeros(
                (config.vocab_size, 64)
            )
        }
        self.assertIs(model.sanitize(prefixed), prefixed)

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits = model(prompt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 8, config.vocab_size))

        nxt = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(nxt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 1, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_deepseek_v3_language_model(self):
        from mlx_vlm.models import deepseek_v3

        config = deepseek_v3.ModelConfig(
            model_type="deepseek_v3",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=24,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            moe_layer_freq=1,
            first_k_dense_replace=1,
            max_position_embeddings=1024,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_bias=False,
        )

        model = deepseek_v3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        sanitized = model.sanitize(
            {
                "model.embed_tokens.weight": mx.zeros((config.vocab_size, 128)),
                "lm_head.weight": mx.zeros((config.vocab_size, 128)),
            }
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)

        prefixed = {"language_model.lm_head.weight": mx.zeros((config.vocab_size, 128))}
        self.assertIs(model.sanitize(prefixed), prefixed)

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits = model(prompt).logits
        self.assertEqual(logits.shape, (1, 8, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_deepseek_v32_language_model(self):
        from mlx_vlm.models import deepseek_v32

        config = deepseek_v32.ModelConfig(
            model_type="deepseek_v32",
            vocab_size=1024,
            hidden_size=128,
            index_head_dim=16,
            index_n_heads=4,
            index_topk=4,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=24,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            moe_layer_freq=1,
            first_k_dense_replace=1,
            max_position_embeddings=1024,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_bias=False,
        )

        model = deepseek_v32.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        cache = model.make_cache()
        self.assertEqual(len(cache), config.num_hidden_layers)
        self.assertEqual(len(cache[0].caches), 2)

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits = model(prompt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 8, config.vocab_size))

        nxt = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(nxt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 1, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_qwen2_language_model(self):
        from mlx_vlm.models import qwen2

        config = qwen2.ModelConfig(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )

        model = qwen2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        sanitized = model.sanitize(
            {"model.embed_tokens.weight": mx.zeros((config.vocab_size, 64))}
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)

        prefixed = {
            "language_model.model.embed_tokens.weight": mx.zeros(
                (config.vocab_size, 64)
            )
        }
        self.assertIs(model.sanitize(prefixed), prefixed)

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_gemma3_text_language_model(self):
        from mlx_vlm.models import gemma3_text

        config = gemma3_text.ModelConfig.from_dict(
            {
                "model_type": "gemma3_text",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "num_key_value_heads": 2,
                "rope_global_base_freq": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "query_pre_attn_scalar": 16,
                "sliding_window": 8,
                "sliding_window_pattern": 3,
                "max_position_embeddings": 256,
            }
        )

        model = gemma3_text.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        cache = model.language_model.make_cache()
        self.assertEqual(len(cache), config.num_hidden_layers)
        self.assertEqual(type(cache[0]).__name__, "RotatingKVCache")
        self.assertEqual(type(cache[2]).__name__, "KVCache")
        self.assertEqual(type(cache[5]).__name__, "KVCache")

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        sanitized = model.sanitize(
            {"model.embed_tokens.weight": mx.zeros((config.vocab_size, 64))}
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)

        resanitized = model.sanitize(dict(sanitized))
        self.assertEqual(set(resanitized), set(sanitized))

        logits = model(mx.array([[1, 2, 3, 4] * 4]), cache=cache).logits
        self.assertEqual(logits.shape, (1, 16, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

        nxt = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(nxt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 1, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_llama_language_model(self):
        from mlx_vlm.models import llama

        config = llama.ModelConfig(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rope_theta=500000,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 128,
            },
            tie_word_embeddings=True,
        )

        model = llama.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        sanitized = model.sanitize(
            {
                "model.embed_tokens.weight": mx.zeros((config.vocab_size, 64)),
                "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros(8),
            }
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertNotIn(
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq", sanitized
        )

        prefixed = {
            "language_model.model.embed_tokens.weight": mx.zeros(
                (config.vocab_size, 64)
            )
        }
        self.assertIs(model.sanitize(prefixed), prefixed)

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

        sliding_config = llama.ModelConfig(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=128,
            num_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16,
            tie_word_embeddings=True,
        )
        sliding_model = llama.Model(sliding_config)
        cache = sliding_model.language_model.make_cache()
        self.assertEqual(type(cache[0]).__name__, "RotatingKVCache")
        self.assertEqual(type(cache[1]).__name__, "KVCache")

        cached_logits = sliding_model(mx.array([[1, 2, 3, 4] * 8]), cache=cache).logits
        self.assertEqual(cached_logits.shape, (1, 32, sliding_config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(cached_logits)).item())

    def test_mistral_language_model(self):
        from mlx_vlm.models import llama

        config = llama.ModelConfig.from_dict(
            {
                "model_type": "mistral",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 128,
                "num_key_value_heads": 2,
                "max_position_embeddings": 256,
                "rope_theta": 10000.0,
                "sliding_window": 4096,
                "head_dim": 16,
                "tie_word_embeddings": False,
            }
        )

        self.assertEqual(config.layer_types, ["full_attention"] * 2)

        model = llama.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        kept = model.sanitize({"lm_head.weight": mx.zeros((config.vocab_size, 64))})
        self.assertIn("language_model.lm_head.weight", kept)

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_qwen3_language_model(self):
        from mlx_vlm.models import qwen3

        config = qwen3.ModelConfig(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rope_theta=1000000,
            head_dim=32,
            tie_word_embeddings=True,
        )

        model = qwen3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        attn = model.language_model.model.layers[0].self_attn
        self.assertEqual(attn.q_norm.weight.shape, (config.head_dim,))
        self.assertEqual(attn.k_norm.weight.shape, (config.head_dim,))
        self.assertEqual(attn.q_proj.weight.shape[0], config.head_dim * 4)

        inputs = mx.array([[1, 2, 3]])
        embeddings = model.get_input_embeddings(inputs)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, config.hidden_size))

        sanitized = model.sanitize(
            {
                "model.embed_tokens.weight": mx.zeros((config.vocab_size, 64)),
                "lm_head.weight": mx.zeros((config.vocab_size, 64)),
            }
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertNotIn("language_model.lm_head.weight", sanitized)

        prefixed = {
            "language_model.model.embed_tokens.weight": mx.zeros(
                (config.vocab_size, 64)
            )
        }
        self.assertIs(model.sanitize(prefixed), prefixed)

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

        untied = qwen3.ModelConfig(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rope_theta=1000000,
            head_dim=32,
            tie_word_embeddings=False,
        )
        untied_model = qwen3.Model(untied)
        kept = untied_model.sanitize(
            {"lm_head.weight": mx.zeros((untied.vocab_size, 64))}
        )
        self.assertIn("language_model.lm_head.weight", kept)
        logits = untied_model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, untied.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_llava_bunny(self):
        from mlx_vlm.models import llava_bunny

        text_config = llava_bunny.TextConfig(
            model_type="qwen2",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava_bunny.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava_bunny.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava-qwen2",
            auto_map={
                "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
            },
            hidden_size=1024,
            mm_hidden_size=1152,
            mm_projector_type="mlp2x_gelu",
            ignore_index=-100,
            image_token_index=-200,
            vocab_size=151936,
        )

        model = llava_bunny.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.mm_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llava_next(self):
        from mlx_vlm.models import llava_next

        text_config = llava_next.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava_next.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava_next.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava_next.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llava(self):
        from mlx_vlm.models import llava

        text_config = llava.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_idefics2(self):
        from mlx_vlm.models import idefics2

        text_config = idefics2.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=8,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = idefics2.VisionConfig(
            model_type="idefics2",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=980,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        perceiver_config = idefics2.PerceiverConfig(
            model_type="idefics2Perceiver",
            resampler_n_latents=64,
            resampler_depth=3,
            resampler_n_heads=16,
            resampler_head_dim=96,
            num_key_value_heads=4,
        )

        config = idefics2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            perceiver_config=perceiver_config,
            model_type="idefics2",
            ignore_index=-100,
            image_token_index=32001,
        )

        model = idefics2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_idefics3(self):
        from mlx_vlm.models import idefics3

        text_config = idefics3.TextConfig(
            model_type="idefics3",
            hidden_size=2048,
            num_hidden_layers=24,
            intermediate_size=8192,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=49155,
            num_key_value_heads=8,
            rope_theta=273768.0,
            rope_traditional=False,
        )

        vision_config = idefics3.VisionConfig(
            model_type="idefics3",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = idefics3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="idefics3",
            ignore_index=-100,
            image_token_id=49153,
        )

        model = idefics3.Model(config)

        head_dim = (
            config.text_config.hidden_size // config.text_config.num_attention_heads
        )
        expected_kv_width = config.text_config.num_key_value_heads * head_dim
        self.assertEqual(
            model.language_model.layers[0].self_attn.k_proj.weight.shape,
            (expected_kv_width, config.text_config.hidden_size),
        )
        self.assertEqual(
            model.language_model.layers[0].self_attn.v_proj.weight.shape,
            (expected_kv_width, config.text_config.hidden_size),
        )

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_locateanything(self):
        from mlx_vlm.models import locateanything

        text_config = locateanything.TextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            max_position_embeddings=512,
            tie_word_embeddings=True,
        )

        vision_config = locateanything.VisionConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            patch_size=14,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            num_channels=3,
            merge_kernel_size=[2, 2],
        )

        config = locateanything.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=5,
            vocab_size=128,
        )

        model = locateanything.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        model = locateanything.Model(config)
        pixels = mx.random.uniform(shape=(16, 14, 14, 3))
        image_features = model.vision_tower(
            pixels,
            grid_thw=mx.array([[4, 4]]),
            grid_shapes=[(4, 4)],
        )
        self.assertEqual(len(image_features), 1)
        self.assertEqual(image_features[0].shape, (4, 4, 32))

        projected = model.multi_modal_projector(image_features)
        self.assertEqual(projected.shape, (4, 64))

        input_ids = mx.array([[5, 5, 5, 5, 1, 2, 3]])
        pixel_values = mx.random.uniform(shape=(16, 3, 14, 14))
        embeddings = model.get_input_embeddings(
            input_ids,
            pixel_values=pixel_values,
            image_grid_hws=mx.array([[4, 4]]),
            _grid_shapes=[(4, 4)],
        )
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 7, 64))

        out = model(
            input_ids,
            pixel_values=pixel_values,
            image_grid_hws=mx.array([[4, 4]]),
            _grid_shapes=[(4, 4)],
        )
        self.assertEqual(out.logits.shape, (1, 7, 128))

    def test_smolvlm_text_config_infers_heads_from_head_dim(self):
        from mlx_vlm.models import smolvlm

        text_config = smolvlm.TextConfig.from_dict(
            {
                "hidden_size": 2048,
                "head_dim": 64,
                "intermediate_size": 8192,
                "num_hidden_layers": 24,
                "rms_norm_eps": 1e-5,
                "vocab_size": 49280,
            }
        )

        self.assertEqual(text_config.num_attention_heads, 32)
        self.assertEqual(text_config.num_key_value_heads, 32)

    def test_smolvlm_vision_config_infers_500m_defaults(self):
        from mlx_vlm.models import smolvlm

        vision_config = smolvlm.VisionConfig.from_dict(
            {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "patch_size": 16,
                "image_size": 512,
                "model_type": "smolvlm_vision",
            }
        )

        self.assertEqual(vision_config.num_hidden_layers, 12)
        self.assertEqual(vision_config.intermediate_size, 3072)

    def test_internvl_chat(self):
        from mlx_vlm.models import internvl_chat

        test_config = internvl_chat.TextConfig(
            model_type="qwen2",
            hidden_size=3584,
            num_hidden_layers=5,
            intermediate_size=18944,
            num_attention_heads=28,
            rms_norm_eps=1e-6,
            max_window_layers=16,
            use_sliding_window=False,
            vocab_size=151674,
            num_key_value_heads=4,
            rope_theta=1000000.0,
            rope_scaling={"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"},
            hidden_act="silu",
            max_position_embeddings=32768,
        )

        vision_config = internvl_chat.VisionConfig(
            model_type="intern_vit_6b",
            num_hidden_layers=5,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = internvl_chat.ModelConfig(
            text_config=test_config,
            vision_config=vision_config,
            model_type="internvl_chat",
            ignore_index=-100,
            image_token_index=151667,
        )

        model = internvl_chat.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_paligemma(self):
        from mlx_vlm.models import paligemma

        text_config = paligemma.TextConfig(
            model_type="gemma",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
            num_key_value_heads=1,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = paligemma.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=2048,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = paligemma.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="paligemma",
            ignore_index=-100,
            image_token_index=257152,
            hidden_size=2048,
            vocab_size=257216,
        )

        model = paligemma.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_paligemma_from_dict_defaults_bidirectional_attention(self):
        from mlx_vlm.models import paligemma

        config = paligemma.ModelConfig.from_dict(
            {
                "model_type": "paligemma",
                "hidden_size": 2048,
                "projection_dim": 2048,
                "text_config": {
                    "model_type": "gemma2",
                    "hidden_size": 2048,
                    "num_hidden_layers": 2,
                    "intermediate_size": 8192,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "vocab_size": 256000,
                    "head_dim": 256,
                    "query_pre_attn_scalar": 256,
                    "attn_logit_softcapping": 50.0,
                    "final_logit_softcapping": 30.0,
                    "hidden_act": "gelu_pytorch_tanh",
                },
                "vision_config": {
                    "model_type": "siglip_vision_model",
                    "num_hidden_layers": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_attention_heads": 16,
                    "image_size": 896,
                    "patch_size": 14,
                },
            }
        )

        self.assertTrue(config.text_config.use_bidirectional_attention)
        self.assertEqual(config.text_config.hidden_activation, "gelu_pytorch_tanh")
        self.assertEqual(config.text_config.num_image_tokens, 4096)
        self.assertEqual(config.vision_config.projection_dim, 2048)

    def test_multi_modality(self):
        from mlx_vlm.models import multi_modality

        text_config = multi_modality.TextConfig(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=24,
            intermediate_size=5632,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = multi_modality.VisionConfig(
            model_type="vision",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-5,
            params={},
        )

        projector_config = multi_modality.ProjectorConfig(
            cls="MlpProjector",
            model_type="projector",
            params={
                "depth": 2,
                "input_dim": 1024,
                "n_embed": 2048,
                "projector_type": "mlp_gelu",
            },
        )

        config = multi_modality.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="multi_modality",
            ignore_index=-100,
            image_token_index=100015,
            vocab_size=32000,
        )

        model = multi_modality.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.aligner,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_phi3_v(self):
        from mlx_vlm.models import phi3_v

        text_config = phi3_v.TextConfig()

        vision_config = phi3_v.VisionConfig(
            model_type="phi3_v",
            image_dim_out=1024,
            model_name="openai/clip-vit-large-patch14-336",
            name="clip_vision_model",
            num_img_tokens=144,
        )

        # Use smaller model dimensions for CI memory constraints
        config = phi3_v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            **{
                "hidden_size": 256,
                "intermediate_size": 512,
                "max_position_embeddings": 4096,
                "model_type": "phi3_v",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "original_max_position_embeddings": 4096,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "long_factor": [1.0] * 32,
                    "short_factor": [1.0] * 32,
                    "type": "su",
                },
                "rope_theta": 10000.0,
                "vocab_size": 1000,
            },
        )

        model = phi3_v.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_mistral3(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            head_dim=128,
            hidden_size=5120,
            intermediate_size=32768,
            max_position_embeddings=131072,
            model_type="mistral",
            num_attention_heads=32,
            num_hidden_layers=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            rope_theta=1000000000.0,
            vocab_size=131072,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=False,
            layer_types=["full_attention"] * 40,
            use_qk_norm=False,
        )

        vision_config = mistral3.VisionConfig(
            model_type="pixtral",
            hidden_size=1024,
            num_hidden_layers=24,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            rms_norm_eps=1e-6,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mistral3",
        )

        model = mistral3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_ministral3(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            head_dim=128,
            hidden_size=3072,
            intermediate_size=9216,
            max_position_embeddings=262144,
            model_type="ministral3",
            num_attention_heads=32,
            num_hidden_layers=26,
            rms_norm_eps=1e-05,
            rope_parameters={
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 16.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "type": "yarn",
            },
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=True,
            vocab_size=131072,
        )

        vision_config = mistral3.VisionConfig(
            head_dim=64,
            hidden_size=1024,
            image_size=1540,
            intermediate_size=4096,
            model_type="pixtral",
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=24,
            patch_size=14,
            rope_theta=10000.0,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="pixtral",
        )
        model = mistral3.Model(config)
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )
        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_pixtral(self):
        from mlx_vlm.models import pixtral

        text_config = pixtral.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = pixtral.VisionConfig(
            model_type="pixtral",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            rms_norm_eps=1e-6,
        )

        config = pixtral.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="pixtral",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = pixtral.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        pixel_values = mx.random.uniform(shape=(2, 56, 56, 3))
        image_sizes = mx.array([[28, 42], [56, 56]])

        full_hidden, _ = model.vision_tower(pixel_values, output_hidden_states=True)
        sized_hidden, _ = model.vision_tower(
            pixel_values, output_hidden_states=True, image_sizes=image_sizes
        )

        expected_full_tokens = 2 * (56 // 14) * (56 // 14)
        expected_sized_tokens = (28 // 14) * (42 // 14) + (56 // 14) * (56 // 14)

        self.assertEqual(full_hidden.shape[1], expected_full_tokens)
        self.assertEqual(sized_hidden.shape[1], expected_sized_tokens)

    def test_qwen2_vl(self):
        from mlx_vlm.models import qwen2_vl

        text_config = qwen2_vl.TextConfig(
            model_type="qwen2_vl",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=37,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=152064,
            num_key_value_heads=4,
            max_position_embeddings=512,
            rope_theta=10000,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
            tie_word_embeddings=False,
        )

        vision_config = qwen2_vl.VisionConfig(
            model_type="qwen2_vl",
            depth=2,
            embed_dim=32,
            hidden_size=32,
            image_size=224,
            num_heads=4,
            patch_size=14,
            mlp_ratio=4,
            in_channels=3,
            spatial_merge_size=1,
            temporal_patch_size=2,
        )

        config = qwen2_vl.ModelConfig(
            model_type="qwen2_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=151655,
            vocab_size=32000,
        )

        model = qwen2_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.in_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=-1,
            grid_thw=mx.ones((1, 3)),  # image temporals shape (num_images, 3)
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )
        self._assert_mrope_decode_uses_rope_deltas_kwarg(
            model.language_model, config.text_config.hidden_size
        )

    def test_qwen2_5_vl(self):
        from mlx_vlm.models import qwen2_5_vl

        text_config = qwen2_5_vl.TextConfig(
            model_type="qwen2_5_vl",
            hidden_size=1280,
            num_hidden_layers=32,
            intermediate_size=3420,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=16,
            max_position_embeddings=128000,
            rope_theta=1000000.0,
            rope_traditional=False,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
            tie_word_embeddings=True,
        )

        vision_config = qwen2_5_vl.VisionConfig(
            model_type="qwen2_5_vl",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            image_size=384,
            vocab_size=32000,
            mlp_ratio=4.0,
            in_channels=3,
            layer_norm_eps=1e-6,
            spatial_patch_size=14,
            spatial_merge_size=2,
            tokens_per_second=2,
            temporal_patch_size=2,
            window_size=112,
            patch_size=14,
            fullatt_block_indexes=[7, 15, 23, 31],
        )

        config = qwen2_5_vl.ModelConfig(
            model_type="qwen2_5_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=32000,
        )

        model = qwen2_5_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )
        self._assert_mrope_decode_uses_rope_deltas_kwarg(
            model.language_model, config.text_config.hidden_size
        )

    def test_dots_ocr(self):
        from mlx_vlm.models import dots_ocr

        text_config = dots_ocr.TextConfig(
            model_type="dots_ocr",
            vocab_size=256,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            attention_bias=True,
            tie_word_embeddings=False,
        )

        vision_config = dots_ocr.VisionConfig(
            model_type="dots_vit",
            embed_dim=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=1,
            use_bias=False,
        )

        config = dots_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="dots_ocr",
            image_token_id=10,
            video_token_id=11,
            vocab_size=256,
        )

        model = dots_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        pixel_values = mx.random.uniform(shape=(4, 3 * 14 * 14), dtype=mx.float32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        vision_features = model.vision_tower(pixel_values, image_grid_thw)
        self.assertEqual(vision_features.shape, (1, 64))

        input_ids = mx.array([[1, config.image_token_id, 2]], dtype=mx.int32)
        embeddings = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, 64))

    def test_qwen3_vl(self):
        from mlx_vlm.models import qwen3_vl

        text_config = qwen3_vl.TextConfig(
            model_type="qwen3_vl_text",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            head_dim=32,
            vocab_size=10_000,
            rope_theta=1000,
            max_position_embeddings=1000,
            tie_word_embeddings=False,
            norm_topk_prob=True,
            rope_scaling={"rope_type": "mrope", "mrope_section": [8, 6, 6]},
        )

        vision_config = qwen3_vl.VisionConfig(
            model_type="qwen3_vl",
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            out_hidden_size=128,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=144,
            deepstack_visual_indexes=[],
        )

        config = qwen3_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="qwen3_vl",
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=10_000,
        )

        model = qwen3_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Test vision model with proper input format
        # Input shape: (total_patches, channels, temporal_patch_size, patch_size, patch_size)
        # For grid_thw = [1, 28, 28], we have 1*28*28 = 784 patches
        grid_thw = mx.array([[1, 28, 28]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor
        pixel_values = mx.random.uniform(
            shape=(
                num_patches,
                config.vision_config.in_channels,
                config.vision_config.temporal_patch_size,
                config.vision_config.patch_size,
                config.vision_config.patch_size,
            )
        )

        # Forward pass
        hidden_states, _ = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have 28/2 * 28/2 = 14*14 = 196 patches
        expected_patches = (
            grid_thw[0, 1] // config.vision_config.spatial_merge_size
        ) * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

        # Multi-image batch: per-sample slicing in `_deepstack_process` must
        # avoid the (N,D)/(M,D) broadcast crash and write through to output.
        self._run_deepstack_multi_image_assertions(
            model.language_model.model._deepstack_process
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )
        self._assert_mrope_decode_uses_rope_deltas_kwarg(
            model.language_model, config.text_config.hidden_size
        )

    def test_qwen3_5_model_config(self):
        from mlx_vlm.models import qwen3_5, qwen3_5_moe

        quantization = {
            "group_size": 128,
            "bits": 4,
            "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                "group_size": 128,
                "bits": 6,
            },
            "model.visual.blocks.0.attn.qkv": False,
            "lm_head": False,
        }

        for model_module in (qwen3_5, qwen3_5_moe):
            with self.subTest(model_type=model_module.__name__):
                config = model_module.ModelConfig.from_dict(
                    {
                        "model_type": model_module.__name__.rsplit(".", 1)[-1],
                        "text_config": {},
                        "vision_config": {"patch_size": 16},
                        "quantization": quantization,
                        "quantization_config": quantization,
                    }
                )

                self.assertEqual(config.vision_config.patch_size, 16)
                self.assertIn(
                    "language_model.model.layers.0.linear_attn.in_proj_qkv",
                    config.quantization,
                )
                self.assertEqual(
                    config.quantization[
                        "language_model.model.layers.0.linear_attn.in_proj_qkv"
                    ],
                    {"group_size": 128, "bits": 6},
                )
                self.assertNotIn(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv",
                    config.quantization,
                )
                self.assertIn("vision_tower.blocks.0.attn.qkv", config.quantization)
                self.assertIn("language_model.lm_head", config.quantization)
                self.assertIs(config.quantization, config.quantization_config)

    def test_qwen3_5_decode_uses_rope_deltas_kwarg(self):
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
            num_hidden_layers=1,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=128,
            head_dim=8,
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
        language_model = qwen3_5.LanguageModel(text_config, config)

        self._assert_mrope_decode_uses_rope_deltas_kwarg(language_model, 16)

    def test_qwen3_5_sanitize_key_routes_nested_visual_weights(self):
        from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key

        self.assertEqual(
            sanitize_key("model.language_model.visual.blocks.0.attn.qkv.weight"),
            "vision_tower.blocks.0.attn.qkv.weight",
        )
        self.assertEqual(
            sanitize_key(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            ),
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        )
        self.assertEqual(
            sanitize_key("model.visual.blocks.0.attn.qkv.weight"),
            "vision_tower.blocks.0.attn.qkv.weight",
        )
        self.assertEqual(
            sanitize_key("lm_head.weight"),
            "language_model.lm_head.weight",
        )

    def test_qwen3_5_rotary_inv_freq_is_thread_safe(self):
        if not mx.metal.is_available():
            self.skipTest("requires Metal streams")

        from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

        rotary = Qwen3_5RotaryEmbedding(dim=8)
        errors = []

        def worker():
            try:
                y = mx.ones((rotary.inv_freq.shape[0],), dtype=mx.float32)
                y = y * rotary.inv_freq
                mx.eval(y)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertEqual(errors, [])

    def test_qwen3_5_model_config_promotes_text_eos_token_id(self):
        from mlx_vlm.models import qwen3_5, qwen3_5_moe

        text_configs = {
            qwen3_5: qwen3_5.TextConfig(
                model_type="qwen3_5",
                hidden_size=128,
                intermediate_size=256,
                linear_num_value_heads=2,
                linear_num_key_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                num_hidden_layers=4,
                num_attention_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=1024,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                eos_token_id=248044,
            ),
            qwen3_5_moe: qwen3_5_moe.TextConfig(
                model_type="qwen3_5_moe",
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                linear_num_value_heads=2,
                linear_num_key_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                num_experts=4,
                num_experts_per_tok=2,
                shared_expert_intermediate_size=128,
                moe_intermediate_size=128,
                rms_norm_eps=1e-5,
                vocab_size=1024,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                eos_token_id=248044,
            ),
        }

        for model_module, text_config in text_configs.items():
            with self.subTest(model_type=model_module.__name__):
                config = model_module.ModelConfig(
                    text_config=text_config,
                    vision_config=SimpleNamespace(),
                    model_type=model_module.__name__.rsplit(".", 1)[-1],
                )
                self.assertEqual(config.eos_token_id, [248044, 248046])

                config_from_dict = model_module.ModelConfig.from_dict(
                    {
                        "model_type": model_module.__name__.rsplit(".", 1)[-1],
                        "text_config": {"eos_token_id": 248044},
                        "vision_config": {"patch_size": 16},
                    }
                )
                self.assertEqual(config_from_dict.vision_config.patch_size, 16)
                self.assertEqual(config_from_dict.eos_token_id, [248044, 248046])

                explicit = model_module.ModelConfig(
                    text_config=text_config,
                    vision_config=SimpleNamespace(),
                    model_type=model_module.__name__.rsplit(".", 1)[-1],
                    eos_token_id=248046,
                )
                self.assertEqual(explicit.eos_token_id, 248046)

    def test_phi3_v_model_config_uses_chat_control_eos_token_ids(self):
        from mlx_vlm.models import phi3_v

        config = phi3_v.ModelConfig.from_dict(
            {
                "model_type": "phi3_v",
                "vocab_size": 32064,
                "eos_token_id": 2,
            }
        )
        self.assertEqual(config.eos_token_id, [2, 32000, 32007])

        explicit = phi3_v.ModelConfig(
            model_type="phi3_v",
            vocab_size=32064,
            eos_token_id=[2, 32007, 32000],
        )
        self.assertEqual(explicit.eos_token_id, [2, 32007, 32000])

        tiny = phi3_v.ModelConfig(model_type="phi3_v", vocab_size=1000)
        self.assertIsNone(tiny.eos_token_id)

    def test_qwen3_vl_moe(self):
        from mlx_vlm.models import qwen3_vl_moe

        text_config = qwen3_vl_moe.TextConfig(
            model_type="qwen3_vl_moe_text",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            head_dim=32,
            vocab_size=10_000,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            num_experts_per_tok=2,
            num_experts=4,
            moe_intermediate_size=128,
            rope_theta=1000,
            max_position_embeddings=1000,
            tie_word_embeddings=False,
            norm_topk_prob=True,
            rope_scaling={"rope_type": "mrope", "mrope_section": [8, 6, 6]},
        )

        vision_config = qwen3_vl_moe.VisionConfig(
            model_type="qwen3_vl_moe",
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            out_hidden_size=128,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=144,
            deepstack_visual_indexes=[],
        )

        config_from_dict = qwen3_vl_moe.ModelConfig.from_dict(
            {
                "model_type": "qwen3_vl_moe",
                "text_config": vars(text_config).copy(),
                "vision_config": {**vars(vision_config), "patch_size": 16},
                "image_token_id": 151655,
                "video_token_id": 151656,
                "vocab_size": 10_000,
            }
        )
        self.assertIsInstance(config_from_dict.text_config, qwen3_vl_moe.TextConfig)
        self.assertIsInstance(config_from_dict.vision_config, qwen3_vl_moe.VisionConfig)
        self.assertEqual(config_from_dict.vision_config.patch_size, 16)
        model_from_dict = qwen3_vl_moe.Model(config_from_dict)
        self.assertEqual(model_from_dict.vision_tower.patch_embed.patch_size, 16)

        config = qwen3_vl_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="qwen3_vl_moe",
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=10_000,
        )

        model = qwen3_vl_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Test vision model with proper input format
        # Input shape: (total_patches, channels, temporal_patch_size, patch_size, patch_size)
        # For grid_thw = [1, 28, 28], we have 1*28*28 = 784 patches
        grid_thw = mx.array([[1, 28, 28]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor
        pixel_values = mx.random.uniform(
            shape=(
                num_patches,
                config.vision_config.in_channels,
                config.vision_config.temporal_patch_size,
                config.vision_config.patch_size,
                config.vision_config.patch_size,
            )
        )

        # Forward pass
        hidden_states, _ = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have 28/2 * 28/2 = 14*14 = 196 patches
        expected_patches = (
            grid_thw[0, 1] // config.vision_config.spatial_merge_size
        ) * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

        # Multi-image batch
        self._run_deepstack_multi_image_assertions(
            model.language_model.model._deepstack_process
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )
        self._assert_mrope_decode_uses_rope_deltas_kwarg(
            model.language_model, config.text_config.hidden_size
        )

    def test_qwen3_vl_deepstack_mask_aligned_on_decode(self):
        from mlx_vlm.models import qwen3_vl, qwen3_vl_moe

        cases = [
            (
                qwen3_vl,
                qwen3_vl.TextConfig(
                    model_type="qwen3_vl_text",
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-5,
                    head_dim=4,
                    vocab_size=32,
                    rope_theta=1000,
                    max_position_embeddings=1000,
                    tie_word_embeddings=False,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
            ),
            (
                qwen3_vl_moe,
                qwen3_vl_moe.TextConfig(
                    model_type="qwen3_vl_moe_text",
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-5,
                    head_dim=4,
                    vocab_size=32,
                    decoder_sparse_step=1,
                    mlp_only_layers=[],
                    num_experts_per_tok=1,
                    num_experts=1,
                    moe_intermediate_size=8,
                    rope_theta=1000,
                    max_position_embeddings=1000,
                    tie_word_embeddings=False,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
            ),
        ]

        class Recorder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.visual_pos_masks = None

            def __call__(
                self,
                inputs,
                *,
                visual_pos_masks=None,
                **kwargs,
            ):
                self.visual_pos_masks = visual_pos_masks
                return mx.zeros(
                    (inputs.shape[0], inputs.shape[1], self.hidden_size),
                    dtype=mx.float32,
                )

        for model_module, text_config in cases:
            with self.subTest(model_type=text_config.model_type):
                language_model = model_module.LanguageModel(text_config)
                recorder = Recorder(text_config.hidden_size)
                language_model.model = recorder
                full_visual_mask = mx.array(
                    [[False, True, True, False, True, False, False]]
                )

                language_model(
                    mx.array([[9]]),
                    inputs_embeds=mx.zeros((1, 1, text_config.hidden_size)),
                    cache=[SimpleNamespace(offset=5)],
                    position_ids=mx.zeros((3, 1, 1), dtype=mx.int64),
                    visual_pos_masks=full_visual_mask,
                    deepstack_visual_embeds=[
                        mx.zeros(
                            (
                                int(full_visual_mask.sum().item()),
                                text_config.hidden_size,
                            )
                        )
                    ],
                )

                self.assertEqual(recorder.visual_pos_masks.shape, (1, 1))
                self.assertEqual(recorder.visual_pos_masks.tolist(), [[False]])

    def test_qwen3_vl_deepstack_embeds_aligned_on_chunked_prefill(self):
        """Chunked prefill must realign deepstack embeds per window; otherwise later
        chunks reuse the first chunk's embeds (offset resets to 0). #856 / #1323."""
        from types import SimpleNamespace

        from mlx_vlm.models import qwen3_vl, qwen3_vl_moe

        cases = [
            (
                qwen3_vl,
                qwen3_vl.TextConfig(
                    model_type="qwen3_vl_text",
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-5,
                    head_dim=4,
                    vocab_size=32,
                    rope_theta=1000,
                    max_position_embeddings=1000,
                    tie_word_embeddings=False,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
            ),
            (
                qwen3_vl_moe,
                qwen3_vl_moe.TextConfig(
                    model_type="qwen3_vl_moe_text",
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-5,
                    head_dim=4,
                    vocab_size=32,
                    decoder_sparse_step=1,
                    mlp_only_layers=[],
                    num_experts_per_tok=1,
                    num_experts=1,
                    moe_intermediate_size=8,
                    rope_theta=1000,
                    max_position_embeddings=1000,
                    tie_word_embeddings=False,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
            ),
        ]

        class Recorder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.visual_pos_masks = None
                self.deepstack_visual_embeds = None

            def __call__(
                self,
                inputs,
                *,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                **kwargs,
            ):
                self.visual_pos_masks = visual_pos_masks
                self.deepstack_visual_embeds = deepstack_visual_embeds
                return mx.zeros(
                    (inputs.shape[0], inputs.shape[1], self.hidden_size),
                    dtype=mx.float32,
                )

        H = 8
        # 10 positions, vision tokens at {1,2,4,5,7}; marker embed row i == i+1.
        full_mask = mx.array(
            [[False, True, True, False, True, True, False, True, False, False]]
        )
        embeds = mx.concatenate(
            [mx.full((1, H), float(i + 1)) for i in range(5)], axis=0
        )

        for model_module, text_config in cases:
            with self.subTest(model_type=text_config.model_type):
                language_model = model_module.LanguageModel(text_config)
                language_model.model = Recorder(H)

                # Second chunk: window [4:7); 2 vision tokens precede it -> embeds[2:4].
                start, window = 4, 3
                language_model(
                    mx.zeros((1, window), dtype=mx.int64),
                    inputs_embeds=mx.zeros((1, window, H)),
                    cache=[SimpleNamespace(offset=start)],
                    position_ids=mx.zeros((3, 1, window), dtype=mx.int64),
                    visual_pos_masks=full_mask,
                    deepstack_visual_embeds=[embeds],
                )

                recorder = language_model.model
                self.assertEqual(recorder.visual_pos_masks.shape, (1, window))
                self.assertEqual(
                    recorder.visual_pos_masks.tolist(), [[True, True, False]]
                )
                self.assertEqual(recorder.deepstack_visual_embeds[0].shape, (2, H))
                self.assertEqual(
                    recorder.deepstack_visual_embeds[0].tolist(),
                    embeds[2:4].tolist(),
                )

    def _run_deepstack_multi_image_assertions(self, deepstack_fn):
        """Shared assertions for qwen3_vl / qwen3_vl_moe `_deepstack_process`.

        Exercises the multi-image batch path: sample 0 has 2 visual tokens,
        sample 1 has 3 visual tokens. Pre-PR-1055 this crashed with a
        ``Shapes (N,D)/(M,D) cannot be broadcast`` because every sample saw
        the full (5,H) ``visual_embeds`` instead of its per-sample slice.
        """
        H = 4
        # hidden_states: distinct nonzero values per (batch, position) so
        # we can check element-wise where the scatter-add landed.
        base = mx.arange(2 * 6 * H, dtype=mx.float32).reshape(2, 6, H)
        # mask sample 0 -> visuals at rows {1, 3} (2 visuals)
        # mask sample 1 -> visuals at rows {0, 2, 4} (3 visuals)
        visual_pos_masks = mx.array(
            [
                [False, True, False, True, False, False],
                [True, False, True, False, True, False],
            ]
        )
        # 5 distinct visual embed rows (2 for sample 0 + 3 for sample 1)
        visual_embeds = mx.arange(5 * H, dtype=mx.float32).reshape(5, H) + 100.0

        out = deepstack_fn(base, visual_pos_masks, visual_embeds)
        self.assertEqual(out.shape, base.shape)

        out_l = out.tolist()
        base_l = base.tolist()
        emb_l = visual_embeds.tolist()

        # Sample 0: rows 1 and 3 received visual_embeds[0] and [1]
        self.assertEqual(
            out_l[0][1],
            [base_l[0][1][i] + emb_l[0][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[0][3],
            [base_l[0][3][i] + emb_l[1][i] for i in range(H)],
        )
        # Sample 0: untouched rows
        for r in (0, 2, 4, 5):
            self.assertEqual(out_l[0][r], base_l[0][r])

        # Sample 1: rows 0, 2, 4 received visual_embeds[2], [3], [4]
        self.assertEqual(
            out_l[1][0],
            [base_l[1][0][i] + emb_l[2][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[1][2],
            [base_l[1][2][i] + emb_l[3][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[1][4],
            [base_l[1][4][i] + emb_l[4][i] for i in range(H)],
        )
        # Sample 1: untouched rows
        for r in (1, 3, 5):
            self.assertEqual(out_l[1][r], base_l[1][r])

        # Empty-mask sample passes through unchanged.
        empty_masks = mx.array([[False, False, False]])
        empty_hidden = mx.ones((1, 3, H))
        empty_out = deepstack_fn(empty_hidden, empty_masks, mx.zeros((0, H)))
        self.assertEqual(empty_out.tolist(), empty_hidden.tolist())

    def _assert_mrope_decode_uses_cache_idx(self, language_model, hidden_size):
        """Shared assertion: MRoPE decode-step reads RoPE position from
        ``cache[0]._idx`` (Python int) rather than ``cache[0].offset.item()``
        — the latter forces a per-step GPU sync. Regression guard for the
        cache._idx refactor in PR #1055.
        """
        # Skip the prefill branch: pretend deltas have already been computed.
        language_model._rope_deltas = mx.array([[0]])
        language_model._position_ids = None

        captured = {}

        class _CapturingModel:
            """Stand-in for the inner Qwen text model — captures position_ids
            and exposes ``embed_tokens.as_linear`` so the tied-weights branch
            in ``LanguageModel.__call__`` doesn't crash.
            """

            class _Embed:
                @staticmethod
                def as_linear(x):
                    return x

            embed_tokens = _Embed()

            def __call__(self, inputs, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], hidden_size))

        language_model.model = _CapturingModel()
        language_model.lm_head = lambda x: x  # bypass the real linear (untied path)

        class _StubCacheWithIdx:
            """``_idx`` (Python int) deliberately differs from ``offset``. If
            extraction reads ``offset.item()`` the captured position is 3;
            reading ``_idx`` gives 10. ``offset`` is 0-d so the per-sequence
            ``cache_offsets`` / ``cache_offset_array`` branch is skipped
            uniformly across qwen2_vl, qwen2_5_vl, and qwen3_vl.
            """

            def __init__(self):
                self._idx = 10
                self.offset = mx.array(3)  # 0-d -> never the per-seq path

        language_model(mx.array([[5]]), cache=[_StubCacheWithIdx()])

        position_ids = captured["position_ids"]
        self.assertIsNotNone(position_ids)
        self.assertIn(tuple(position_ids.shape), {(1, 1), (3, 1, 1)})
        # Decode position == cache._idx (10), not cache.offset[0].item() (3).
        if position_ids.ndim == 3:
            self.assertEqual(position_ids[0, 0, 0].item(), 10)
        else:
            self.assertEqual(position_ids[0, 0].item(), 10)

    def _assert_mrope_decode_uses_rope_deltas_kwarg(self, language_model, hidden_size):
        """Shared assertion: under continuous batching, an explicit
        ``rope_deltas`` kwarg passed by ``GenerationBatch._step()`` must
        override the mutable ``language_model._rope_deltas`` attribute. The
        latter can be clobbered mid-decode when a newer request's prefill
        runs ``get_input_embeddings`` on the same GPU thread.
        """
        # Stale per-model state — simulates a newer request having just
        # prefilled and overwritten ``_rope_deltas``.
        language_model._rope_deltas = mx.array([[99]])
        language_model._position_ids = None

        captured = {}

        class _CapturingModel:
            class _Embed:
                @staticmethod
                def as_linear(x):
                    return x

            embed_tokens = _Embed()
            # ``fa_idx`` lets the qwen3_5 / qwen3_5_moe cache-indexing path
            # (``cache[self.model.fa_idx]``) resolve to the stub cache below.
            fa_idx = 0

            def __call__(self, inputs, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], hidden_size))

        language_model.model = _CapturingModel()
        language_model.lm_head = lambda x: x

        class _StubCacheWithIdx:
            def __init__(self):
                self._idx = 10
                self.offset = mx.array(3)  # 0-d -> scalar decode branch

        # Caller-supplied kwarg (the row-local delta from ``GenerationBatch``)
        # disagrees with the stale ``_rope_deltas`` (99). Position must
        # follow the kwarg.
        kwarg_delta = mx.array([[5]])
        language_model(
            mx.array([[7]]),
            cache=[_StubCacheWithIdx()],
            rope_deltas=kwarg_delta,
        )

        position_ids = captured["position_ids"]
        self.assertIsNotNone(position_ids)
        self.assertEqual(tuple(position_ids.shape), (3, 1, 1))
        # Position == cache._idx (10) + kwarg delta (5) == 15.
        # Pre-fix behavior would have read self._rope_deltas (99) -> 109.
        self.assertEqual(position_ids[0, 0, 0].item(), 15)

    def test_glm4v_moe(self):
        from mlx_vlm.models import glm4v_moe

        text_config = glm4v_moe.TextConfig(
            model_type="glm4v_moe",
            vocab_size=257152,
            hidden_size=4096,
            intermediate_size=10944,
            max_position_embeddings=8192,
            moe_intermediate_size=13696,
            norm_topk_prob=False,
            num_attention_heads=32,
            n_group=1,
            head_dim=128,
            topk_group=1,
            n_shared_experts=2,
            n_routed_experts=16,
            routed_scaling_factor=1.0,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            num_hidden_layers=5,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_theta=5000000,
            use_qk_norm=True,
            attention_bias=False,
            partial_rotary_factor=0.5,
            rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12]},
            tie_word_embeddings=False,
        )

        vision_config = glm4v_moe.VisionConfig(
            model_type="glm4v_moe",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            patch_size=14,
            window_size=112,
            image_size=336,
            in_channels=3,
            rms_norm_eps=1e-05,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            initializer_range=0.02,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        config = glm4v_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v_moe",
            vocab_size=257152,
            ignore_index=-100,
            image_token_index=151363,
            image_token_id=151363,
            video_token_index=151364,
            video_token_id=151364,
            vision_start_token_id=151339,
            vision_end_token_id=151340,
            hidden_size=2048,
            pad_token_id=0,
        )

        model = glm4v_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_glm4v(self):
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v",
        )

        vision_config = glm4v.VisionConfig(
            model_type="glm4v",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            patch_size=14,
            window_size=112,
            image_size=336,
            in_channels=3,
            rms_norm_eps=1e-05,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            initializer_range=0.02,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        config = glm4v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v",
            vocab_size=257152,
            ignore_index=-100,
            image_token_index=151363,
            image_token_id=151363,
            video_token_index=151364,
            video_token_id=151364,
            vision_start_token_id=151339,
            vision_end_token_id=151340,
            hidden_size=2048,
            pad_token_id=0,
        )

        model = glm4v.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_lfm2_vl(self):
        from mlx_vlm.models import lfm2_vl

        text_config = lfm2_vl.TextConfig(
            layer_types=[
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ],
        )
        vision_config = lfm2_vl.VisionConfig()
        config = lfm2_vl.ModelConfig(
            text_config=text_config, vision_config=vision_config
        )
        model = lfm2_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # TODO: Add vision test runner for lfm2_vl
        # Rewrite inputs to be defined by the test classes

    def test_lfm2_vl_skips_projector_layernorm_when_disabled(self):
        from mlx_vlm.models import lfm2_vl

        text_config = lfm2_vl.TextConfig(layer_types=["full_attention"])
        vision_config = lfm2_vl.VisionConfig()
        config = lfm2_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_use_layernorm=False,
        )
        model = lfm2_vl.Model(config)

        self.assertFalse(model.multi_modal_projector.projector_use_layernorm)
        # LFM2.5-VL checkpoints ship no projector layer_norm weights, so the
        # module must not exist or strict loading would require them.
        parameters = model.multi_modal_projector.parameters()
        self.assertNotIn("layer_norm", parameters)

    def test_lfm2_vl_projector_skips_disabled_layernorm_branch(self):
        from mlx_vlm.models import lfm2_vl
        from mlx_vlm.models.lfm2_vl.lfm2_vl import Lfm2VlMultiModalProjector

        text_config = lfm2_vl.TextConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            layer_types=["full_attention"],
        )
        vision_config = lfm2_vl.VisionConfig(hidden_size=2)
        config = lfm2_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            downsample_factor=1,
            projector_hidden_size=3,
            projector_use_layernorm=False,
        )
        projector = Lfm2VlMultiModalProjector(config)

        class FailingLayerNorm(nn.Module):
            def __call__(self, x):
                raise AssertionError("layer_norm should be skipped")

        projector.layer_norm = FailingLayerNorm()
        output = projector(mx.zeros((1, 1, 1, 2)))

        self.assertEqual(output.shape, (1, 1, 1, 4))

    def test_mllama(self):
        from mlx_vlm.models import mllama

        vision_config = mllama.VisionConfig(
            image_size=50,
            patch_size=14,
            num_channels=3,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=10,
            num_attention_heads=16,
            max_num_tiles=4,
            max_aspect_ratio_id=8,
            num_global_layers=8,
            norm_eps=1e-5,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            vision_output_dim=7680,
            intermediate_layers_indices=[3, 7, 15, 23, 30],
        )

        text_config = mllama.TextConfig(
            model_type="mllama",
            hidden_size=4096,
            num_hidden_layers=10,
            intermediate_size=14336,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
        )

        model_config = mllama.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mllama",
            ignore_index=-100,
            image_token_index=128256,
            vision_feature_select_strategy="default",
            vision_feature_layer=-2,
            vocab_size=32000,
        )

        # Create the model
        model = mllama.Model(model_config)

        # Create dummy input data
        batch_size = 1
        seq_length = 5
        num_tiles = 4
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        pixel_values = mx.random.normal((batch_size, 1, num_tiles, 3, 50, 50))
        mask = mx.ones((batch_size, seq_length))
        aspect_ratio_ids = mx.zeros((batch_size, 1), dtype=mx.int32)
        aspect_ratio_mask = mx.ones((batch_size, 1, num_tiles))
        cross_attention_mask = mx.ones((batch_size, seq_length, 1, num_tiles))

        # Forward pass
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
        )

        # Check output shape
        expected_shape = (batch_size, seq_length, model_config.vocab_size)
        self.assertEqual(output.logits.shape, expected_shape)

    def test_molmo(self):
        from mlx_vlm.models import molmo

        text_config = molmo.TextConfig()
        vision_config = molmo.VisionConfig()
        config = molmo.ModelConfig(text_config=text_config, vision_config=vision_config)
        model = molmo.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.n_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.d_model,
            config.vision_config.num_channels,
            (576, 588),
        )

    def test_molmo2(self):
        from mlx_vlm.models import molmo2

        text_config = molmo2.TextConfig(
            model_type="molmo2",
            hidden_size=256,  # Reduced for testing
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=151936,
            additional_vocab_size=128,
            hidden_act="silu",
            layer_norm_eps=1e-6,
            rope_theta=5000000.0,
            use_qk_norm=True,
        )

        vit_config = molmo2.config.VitConfig(
            model_type="molmo2",
            hidden_size=128,  # Reduced for testing
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            image_patch_size=14,
            image_num_pos=729,  # 27x27
            image_default_input_size=[378, 378],
        )

        adapter_config = molmo2.config.AdapterConfig(
            model_type="molmo2",
            hidden_size=128,  # Match vit_config
            intermediate_size=256,
            text_hidden_size=256,  # Match text_config
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            vit_layers=[-1, -2],  # Use last two layers
        )

        vision_config = molmo2.VisionConfig(
            vit_config=vit_config,
            adapter_config=adapter_config,
        )

        config = molmo2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="molmo2",
        )
        model = molmo2.Model(config)

        # Test language model
        # Note: vocab_size in logits is base vocab only, additional tokens are handled separately
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_molmo_point_config_accepts_eos_token_id(self):
        from mlx_vlm.models import molmo_point
        from mlx_vlm.utils import apply_generation_config_defaults

        config = molmo_point.ModelConfig.from_dict(
            {
                "model_type": "molmo_point",
                "eos_token_id": 151645,
            }
        )

        self.assertEqual(config.eos_token_id, 151645)

        apply_generation_config_defaults(config, {"eos_token_id": [151645, 151646]})

        self.assertEqual(config.eos_token_id, [151645, 151646])

    def test_molmo2_sanitizes_non_finite_image_features(self):
        from mlx_vlm.models.molmo2.molmo2 import (
            MAX_FLOAT16_IMAGE_FEATURE,
            clip_image_features,
        )

        fp32_features = mx.array(
            [float("nan"), float("inf"), float("-inf"), 70000.0, -70000.0, 42.0],
            dtype=mx.float32,
        )
        fp16_features = mx.array([70000.0, -70000.0, 42.0], dtype=mx.float16)

        fp32_clipped = np.array(clip_image_features(fp32_features))
        fp16_clipped = np.array(clip_image_features(fp16_features))

        self.assertTrue(np.isfinite(fp32_clipped).all())
        self.assertEqual(fp32_clipped[0], 0.0)
        self.assertEqual(fp32_clipped[3], 70000.0)
        self.assertEqual(fp32_clipped[4], -70000.0)
        self.assertEqual(fp32_clipped[5], 42.0)
        self.assertLessEqual(fp16_clipped[0], MAX_FLOAT16_IMAGE_FEATURE)
        self.assertGreaterEqual(fp16_clipped[0], MAX_FLOAT16_IMAGE_FEATURE - 16)
        self.assertGreaterEqual(fp16_clipped[1], -MAX_FLOAT16_IMAGE_FEATURE)
        self.assertLessEqual(fp16_clipped[1], -MAX_FLOAT16_IMAGE_FEATURE + 16)
        self.assertEqual(fp16_clipped[2], 42.0)

    def test_florence2(self):
        from mlx_vlm.models import florence2

        text_config = florence2.TextConfig()
        vision_config = florence2.VisionConfig(drop_path_rate=0.0)
        config = florence2.ModelConfig(
            text_config=text_config, vision_config=vision_config
        )
        model = florence2.Model(config)

        # Create dummy data
        batch_size = 1
        seq_length = 590
        # Create dummy text inputs
        inputs_embeds = mx.zeros((batch_size, seq_length, config.text_config.d_model))

        # Create dummy masks and embeddings
        decoder_inputs_embeds = mx.zeros((batch_size, 1, config.text_config.d_model))

        # Forward pass
        output = model.language_model(
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )

        # Check output shape matches the example shape
        self.assertEqual(
            output.logits.shape, (batch_size, 1, config.text_config.vocab_size)
        )
        self.assertEqual(
            output.encoder_outputs.shape,
            (batch_size, seq_length, config.text_config.d_model),
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.dim_embed[-1],
            config.vision_config.in_chans,
            config.vision_config.image_size,
            channel_first=True,
        )

    def test_deepseek_vl_v2(self):
        from mlx_vlm.models import deepseek_vl_v2

        text_config = deepseek_vl_v2.TextConfig(model_type="text")
        vision_config = deepseek_vl_v2.VisionConfig(model_type="vision")
        projector_config = deepseek_vl_v2.ProjectorConfig()
        config = deepseek_vl_v2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="deepseek_v2",
        )
        model = deepseek_vl_v2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision,
            config.vision_config.model_type,
            config.vision_config.width,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_aya_vision(self):
        from mlx_vlm.models import aya_vision

        text_config = aya_vision.TextConfig(model_type="aya_vision")
        vision_config = aya_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=1152,
            num_attention_heads=16,
            patch_size=14,
            num_hidden_layers=27,
        )
        config = aya_vision.ModelConfig(
            model_type="aya_vision",
            text_config=text_config,
            vision_config=vision_config,
        )
        model = aya_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llama4(self):
        from mlx_vlm.models import llama4

        text_config = llama4.TextConfig(
            model_type="llama4_text",
            hidden_size=5120,
            num_hidden_layers=3,
            intermediate_size=8192,
            intermediate_size_mlp=16384,
            num_attention_heads=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            vocab_size=32000,
            attention_chunk_size=8192,
            attention_dropout=0.0,
            head_dim=128,
            hidden_act="silu",
            attention_bias=False,
        )
        vision_config = llama4.VisionConfig(
            model_type="llama4_vision_model",
            image_size=336,
            patch_size=14,
            num_channels=3,
            num_hidden_layers=3,
            hidden_size=1408,
            intermediate_size=5632,
            num_attention_heads=16,
            norm_eps=1e-05,
            initializer_range=0.02,
            pixel_shuffle_ratio=0.5,
            projector_input_dim=4096,
            projector_output_dim=4096,
            projector_dropout=0.0,
            vision_output_dim=4096,
            rope_theta=10000,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        config = llama4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llama4",
        )
        model = llama4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            channel_first=True,
            projector_output_dim=config.vision_config.projector_output_dim,
        )

    def test_kimi_vl(self):
        pass

        from mlx_vlm.models import kimi_vl

        text_config = kimi_vl.TextConfig()
        vision_config = kimi_vl.VisionConfig()
        config = kimi_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="kimi_vl",
        )
        model = kimi_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.patch_size, config.vision_config.patch_size),
            grid_thw=mx.array(
                [[20, 28], [22, 28]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
            batch_size=1176,
            vision_feature_layer=-1,
        )

    def test_gemma3(self):
        from mlx_vlm.models import gemma3

        text_config = gemma3.TextConfig(
            model_type="gemma3",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
        )
        vision_config = gemma3.VisionConfig(
            model_type="gemma3",
            image_size=224,
            patch_size=14,
            num_channels=3,
            num_hidden_layers=18,
            hidden_size=2048,
            intermediate_size=16384,
            num_attention_heads=8,
        )
        config = gemma3.ModelConfig(
            text_config=text_config, vision_config=vision_config, model_type="gemma3"
        )
        model = gemma3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_gemma4(self):
        import tempfile
        from pathlib import Path

        from mlx_lm.utils import quantize_model

        from mlx_vlm.models import gemma4
        from mlx_vlm.utils import load_model, save_config, save_weights

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=4,
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

        sanitize_model = gemma4.Model(config)
        sanitized = sanitize_model.sanitize(
            {
                "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros(
                    (32, 32), dtype=mx.bfloat16
                ),
                "model.language_model.layers.0.self_attn.q_proj.scales": mx.zeros(
                    (32,), dtype=mx.bfloat16
                ),
                "model.vision_tower.layers.0.self_attn.q_proj.weight": mx.zeros(
                    (32, 32), dtype=mx.bfloat16
                ),
            }
        )
        self.assertEqual(
            sanitized["language_model.model.layers.0.self_attn.q_proj.weight"].dtype,
            mx.float32,
        )
        self.assertEqual(
            sanitized["language_model.model.layers.0.self_attn.q_proj.scales"].dtype,
            mx.bfloat16,
        )
        self.assertEqual(
            sanitized["vision_tower.layers.0.self_attn.q_proj.weight"].dtype,
            mx.bfloat16,
        )

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,  # num_channels
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Full model forward: text-only (no image)
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image tokens
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

        def image(h, w):
            return np.zeros((h, w, 3), dtype=np.uint8)

        image_processor = gemma4.Gemma4ImageProcessor()
        for h, w in ((1125, 1500), (1650, 1275), (480, 640)):
            counts = []
            for budget in (280, 560, 1120):
                _, soft_tokens = image_processor(
                    image(h, w),
                    max_soft_tokens=budget,
                )
                self.assertLessEqual(soft_tokens[0], budget)
                counts.append(soft_tokens[0])
            self.assertTrue(counts[0] < counts[1] < counts[2])

        _, soft_tokens = image_processor(image(1125, 1500))
        self.assertLessEqual(soft_tokens[0], 280)

        tiny_image_processor = gemma4.Gemma4ImageProcessor(
            patch_size=vision_config.patch_size,
            pooling_kernel_size=vision_config.pooling_kernel_size,
            max_soft_tokens=vision_config.default_output_length,
        )
        tiny_tower = gemma4.VisionModel(vision_config)
        tiny_tower.eval()
        variable_image = image(256, 256)
        for budget in (
            vision_config.default_output_length,
            vision_config.default_output_length * 4,
        ):
            data, soft_tokens = tiny_image_processor(
                variable_image,
                max_soft_tokens=budget,
                patch_size=vision_config.patch_size,
                pooling_kernel_size=vision_config.pooling_kernel_size,
            )
            output = tiny_tower(mx.array(data["pixel_values"]))
            mx.eval(output)
            self.assertEqual(output.shape[1], soft_tokens[0])

        patch_dim = 3 * vision_config.patch_size**2
        patch_positions = mx.array([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        patch_values = mx.ones((1, 4, patch_dim)) * 0.5
        output = tiny_tower(patch_values, patch_positions)
        mx.eval(output)
        self.assertEqual(output.shape, (1, 1, vision_config.hidden_size))

        video_patch_values = mx.ones((1, 2, 4, patch_dim)) * 0.5
        video_patch_positions = mx.array([[[[0, 0], [1, 0], [0, 1], [1, 1]]] * 2])
        output = tiny_tower(video_patch_values, video_patch_positions)
        mx.eval(output)
        self.assertEqual(output.shape, (1, 2, vision_config.hidden_size))

        image_token, image_token_id = "<image>", 100
        tokenizer_kwargs = []

        class _Tokenizer:
            def __call__(self, text=None, **kwargs):
                tokenizer_kwargs.append(kwargs)
                return {
                    "input_ids": [
                        [image_token_id] * prompt.count(image_token) for prompt in text
                    ]
                }

        proc = gemma4.Gemma4Processor.__new__(gemma4.Gemma4Processor)
        proc.tokenizer = _Tokenizer()
        proc.image_processor = tiny_image_processor
        proc.video_processor = None
        proc.feature_extractor = None
        proc.image_token = image_token
        proc.image_token_id = image_token_id
        proc.boi_token = "<boi>"
        proc.eoi_token = "<eoi>"
        proc.image_seq_length = vision_config.default_output_length
        proc.audio_token = ""
        proc.boa_token = ""
        proc.eoa_token = ""
        proc.audio_token_id = None
        proc.full_image_sequence = (
            "<boi>" + image_token * vision_config.default_output_length + "<eoi>"
        )
        proc.full_audio_sequence = None

        threaded_counts = []
        for budget in (
            vision_config.default_output_length,
            vision_config.default_output_length * 4,
        ):
            output = proc(
                images=[variable_image],
                text=[f"{image_token} describe"],
                images_kwargs={
                    "max_soft_tokens": budget,
                    "patch_size": vision_config.patch_size,
                    "pooling_kernel_size": vision_config.pooling_kernel_size,
                },
                return_mm_token_type_ids=False,
            )
            ids = np.array(output["input_ids"][0])
            threaded_counts.append(int((ids == image_token_id).sum()))

        self.assertTrue(threaded_counts[0] < threaded_counts[1])
        self.assertTrue(
            all(
                "images_kwargs" not in kwargs and "max_soft_tokens" not in kwargs
                for kwargs in tokenizer_kwargs
            )
        )

        # Quantized save/load regression for per-layer projection.
        quant_model = gemma4.Model(config)

        def quantize_per_layer_projection(path: str, _module: nn.Module):
            return path == "language_model.model.per_layer_model_projection"

        quant_model, quantized_config = quantize_model(
            quant_model,
            {
                "model_type": "gemma4",
                "vocab_size": config.vocab_size,
                "image_token_id": config.image_token_id,
                "audio_config": None,
                "text_config": vars(text_config).copy(),
                "vision_config": vars(vision_config).copy(),
            },
            group_size=32,
            bits=4,
            quant_predicate=quantize_per_layer_projection,
        )
        self.assertTrue(
            hasattr(
                quant_model.language_model.model.per_layer_model_projection, "scales"
            )
        )
        quantized_config["quantization"][
            "language_model.model.per_layer_model_projection"
        ] = {
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        }

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir)
            save_weights(model_path, quant_model)
            save_config(quantized_config, model_path / "config.json")
            loaded = load_model(model_path)

        self.assertTrue(
            hasattr(loaded.language_model.model.per_layer_model_projection, "scales")
        )
        logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32)).logits
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, config.vocab_size))

        # Full model forward: text + audio tokens
        audio_config = gemma4.AudioConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            conv_kernel_size=3,
            attention_chunk_size=4,
            attention_context_left=5,
            attention_context_right=0,
            subsampling_conv_channels=(8, 4),
            output_proj_dims=32,
        )
        config_with_audio = gemma4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            model_type="gemma4",
            vocab_size=64,
            image_token_id=63,
            audio_token_id=62,
        )
        model_audio = gemma4.Model(config_with_audio)
        aud_id = config_with_audio.audio_token_id
        input_ids_audio = mx.array([[0, aud_id, aud_id, aud_id, aud_id, 1]])
        audio_features = mx.random.normal((1, 64, 128))
        audio_mask = mx.zeros((1, 64), dtype=mx.bool_)
        output = model_audio(
            input_ids_audio, audio_features=audio_features, audio_mask=audio_mask
        )
        self.assertEqual(
            output.logits.shape,
            (1, 6, config_with_audio.text_config.vocab_size),
        )

        hf_audio_weights = model_audio.sanitize(
            {
                "audio_tower.subsample_conv_projection.layer0.conv.weight": mx.zeros(
                    (8, 1, 3, 3)
                ),
                "audio_tower.subsample_conv_projection.layer1.conv.weight": mx.zeros(
                    (4, 8, 3, 3)
                ),
                "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": mx.zeros(
                    (32, 1, 3)
                ),
            }
        )
        self.assertEqual(
            hf_audio_weights[
                "audio_tower.subsample_conv_projection.layer0.conv.weight"
            ].shape,
            (8, 3, 3, 1),
        )
        self.assertEqual(
            hf_audio_weights[
                "audio_tower.subsample_conv_projection.layer1.conv.weight"
            ].shape,
            (4, 3, 3, 8),
        )
        self.assertEqual(
            hf_audio_weights[
                "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"
            ].shape,
            (32, 3, 1),
        )

        mlx_audio_weights = model_audio.sanitize(
            {
                "audio_tower.subsample_conv_projection.layer0.conv.weight": mx.zeros(
                    (8, 3, 3, 1)
                ),
                "audio_tower.subsample_conv_projection.layer1.conv.weight": mx.zeros(
                    (4, 3, 3, 8)
                ),
                "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": mx.zeros(
                    (32, 3, 1)
                ),
            }
        )
        self.assertEqual(
            mlx_audio_weights[
                "audio_tower.subsample_conv_projection.layer0.conv.weight"
            ].shape,
            (8, 3, 3, 1),
        )
        self.assertEqual(
            mlx_audio_weights[
                "audio_tower.subsample_conv_projection.layer1.conv.weight"
            ].shape,
            (4, 3, 3, 8),
        )
        self.assertEqual(
            mlx_audio_weights[
                "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"
            ].shape,
            (32, 3, 1),
        )

        shared_text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=8,
            num_kv_shared_layers=2,
            sliding_window=32,
            sliding_window_pattern=2,
        )
        shared_model = gemma4.Model(
            gemma4.ModelConfig(
                text_config=shared_text_config,
                vision_config=vision_config,
                model_type="gemma4",
                vocab_size=config.vocab_size,
                image_token_id=config.image_token_id,
                audio_config=None,
            )
        )

        weights = shared_model.sanitize(
            {
                "model.language_model.layers.1.self_attn.k_proj.weight": mx.zeros(
                    (8, 16)
                ),
                "model.language_model.layers.2.self_attn.k_proj.weight": mx.zeros(
                    (8, 16)
                ),
                "model.language_model.layers.2.self_attn.v_proj.weight": mx.zeros(
                    (8, 16)
                ),
                "model.language_model.layers.2.self_attn.k_norm.weight": mx.zeros((8,)),
                "model.language_model.layers.2.self_attn.q_proj.weight": mx.zeros(
                    (16, 16)
                ),
                "model.language_model.layers.3.self_attn.v_proj.weight": mx.zeros(
                    (8, 16)
                ),
            }
        )
        weights = shared_model.language_model.sanitize(weights)

        self.assertIn("language_model.model.layers.1.self_attn.k_proj.weight", weights)
        self.assertIn("language_model.model.layers.2.self_attn.q_proj.weight", weights)
        self.assertNotIn(
            "language_model.model.layers.2.self_attn.k_proj.weight", weights
        )
        self.assertNotIn(
            "language_model.model.layers.2.self_attn.v_proj.weight", weights
        )
        self.assertNotIn(
            "language_model.model.layers.2.self_attn.k_norm.weight", weights
        )
        self.assertNotIn(
            "language_model.model.layers.3.self_attn.v_proj.weight", weights
        )

    def test_gemma4_unified(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.models import gemma4_unified
        from mlx_vlm.utils import load_model, save_config, save_weights

        text_config = gemma4_unified.TextConfig(
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            sliding_window=32,
            sliding_window_pattern=2,
            attention_k_eq_v=True,
        )
        vision_config = gemma4_unified.VisionConfig(
            patch_size=2,
            pooling_kernel_size=2,
            model_patch_size=4,
            mm_embed_dim=32,
            output_proj_dims=32,
            mm_posemb_size=16,
            num_soft_tokens=4,
        )
        audio_config = gemma4_unified.AudioConfig(
            audio_samples_per_token=8,
            audio_embed_dim=8,
            hidden_size=8,
            output_proj_dims=8,
        )
        config = gemma4_unified.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            vocab_size=64,
            image_token_id=63,
            audio_token_id=62,
            video_token_id=61,
        )
        model = gemma4_unified.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        patch_dim = config.vision_config.model_patch_size**2 * 3
        self.vision_test_runner(
            model.vision_embedder,
            config.vision_config.model_type,
            config.vision_config.mm_embed_dim,
            patch_dim,
            (4, 1),
            vision_feature_layer=0,
        )
        # Restore float32 params so the subsequent full-model forwards run in a
        # consistent dtype.
        model.vision_embedder.update(
            tree_map(lambda p: p.astype(mx.float32), model.vision_embedder.parameters())
        )
        model.language_model.update(
            tree_map(lambda p: p.astype(mx.float32), model.language_model.parameters())
        )

        input_ids = mx.array([[0, 63, 63, 63, 63, 1]], dtype=mx.int32)
        pixel_values = mx.random.uniform(shape=(1, 4, 4 * 4 * 3))
        image_position_ids = mx.array(
            [[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=mx.int32
        )
        output = model(
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        self.assertEqual(output.logits.shape, (1, 6, config.vocab_size))

        input_ids_audio = mx.array([[0, 62, 62, 1]], dtype=mx.int32)
        input_features = mx.random.uniform(shape=(1, 2, 8))
        input_features_mask = mx.array([[True, True]])
        output = model(
            input_ids_audio,
            input_features=input_features,
            input_features_mask=input_features_mask,
        )
        self.assertEqual(output.logits.shape, (1, 4, config.vocab_size))

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir)
            save_weights(model_path, model)
            save_config(
                {
                    "model_type": "gemma4_unified",
                    "vocab_size": config.vocab_size,
                    "image_token_id": config.image_token_id,
                    "audio_token_id": config.audio_token_id,
                    "video_token_id": config.video_token_id,
                    "text_config": vars(text_config).copy(),
                    "vision_config": vars(vision_config).copy(),
                    "audio_config": vars(audio_config).copy(),
                },
                model_path / "config.json",
            )
            loaded_model = load_model(model_path)

        output = loaded_model(mx.array([[0, 1, 2]], dtype=mx.int32))
        self.assertEqual(output.logits.shape, (1, 3, config.vocab_size))

        sanitized = model.sanitize(
            {
                "model.language_model.embed_tokens.weight": mx.zeros((64, 32)),
                "model.vision_embedder.patch_dense.weight": mx.zeros((32, 48)),
                "lm_head.weight": mx.zeros((64, 32)),
            }
        )
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("vision_embedder.patch_dense.weight", sanitized)
        self.assertNotIn("lm_head.weight", sanitized)

        loaded = gemma4_unified.ModelConfig.from_dict(
            {
                "model_type": "gemma4_unified",
                "eoa_token_index": 258883,
                "text_config": {"model_type": "gemma4_unified_text"},
                "vision_config": {"model_type": "gemma4_unified_vision"},
                "audio_config": {"model_type": "gemma4_unified_audio"},
            }
        )
        self.assertEqual(loaded.eoa_token_id, 258883)
        self.assertEqual(loaded.text_config.model_type, "gemma4_unified_text")

    def test_gemma4_moe(self):
        """Gemma 4 MoE variant: MoE, K-eq-V, no per-layer inputs."""
        from mlx_vlm.models import gemma4

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=5,
            final_logit_softcapping=30.0,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=16,
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

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Verify MoE layers are created
        for layer in model.language_model.model.layers:
            self.assertTrue(layer.enable_moe)
            self.assertIsNotNone(layer.router)
            self.assertIsNotNone(layer.experts)

        # Verify K-eq-V on full attention layers
        for layer in model.language_model.model.layers:
            if layer.layer_type == "full_attention":
                self.assertTrue(layer.self_attn.use_k_eq_v)
                self.assertFalse(hasattr(layer.self_attn, "v_proj"))
            else:
                self.assertFalse(layer.self_attn.use_k_eq_v)

        # Verify layer_scalar exists on all layers
        for layer in model.language_model.model.layers:
            self.assertIsNotNone(layer.layer_scalar)

        # Full model forward: text-only
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

    def test_gemma4_attention_snapshots_cache_offset(self):
        """Gemma 4 Attention must snapshot cache.offset to prevent in-place
        mutation aliasing under batched caches where cache.offset is an
        mx.array. Without the snapshot, cache.update_and_fetch would mutate
        the local offset variable between K-rope and Q-rope, producing a
        one-position shift and a deterministic decode loop. See the equivalent
        defense in mlx_lm/models/gemma4_text.py (offset = mx.array(cache.offset)).
        """
        from mlx_vlm.models.gemma4 import language

        text_config = language.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=4,
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
        attn = language.Attention(text_config, layer_idx=0)

        # Stub cache mirroring the BatchRotatingKVCache shape: cache.offset
        # is an mx.array, update_and_fetch advances it in place via +=.
        class _StubMxArrayCache:
            def __init__(self, start):
                self.offset = mx.array([start])
                self.state = (
                    mx.zeros((1, 1, 0, 16)),
                    mx.zeros((1, 1, 0, 16)),
                )
                self.max_size = 2048

            def update_and_fetch(self, keys, values):
                self.offset += keys.shape[-2]
                new_keys = mx.concatenate([self.state[0], keys], axis=-2)
                new_values = mx.concatenate([self.state[1], values], axis=-2)
                self.state = (new_keys, new_values)
                return new_keys, new_values

        cache = _StubMxArrayCache(start=21)
        cache_offset_id = id(cache.offset)

        rope_ids = []
        rope_values = []
        original_rope = attn.rope

        def _recording_rope(x, offset=None):
            rope_ids.append(id(offset) if offset is not None else None)
            rope_values.append(offset.tolist() if hasattr(offset, "tolist") else offset)
            return original_rope(x, offset=offset)

        attn.rope = _recording_rope

        x = mx.random.uniform(shape=(1, 1, text_config.hidden_size))
        output = attn(x, mask=None, cache=cache)
        mx.eval(output)

        # Both K-rope and Q-rope must fire.
        self.assertGreaterEqual(len(rope_ids), 2)

        # The offset object passed to rope must not alias cache.offset;
        # otherwise cache.update_and_fetch would mutate it between K-rope
        # and Q-rope.
        for i, oid in enumerate(rope_ids):
            self.assertNotEqual(
                oid,
                cache_offset_id,
                f"rope call #{i} aliased cache.offset instead of snapshotting",
            )

        # Stub advanced in place, confirming the mx.array mutation path.
        self.assertEqual(cache.offset.tolist(), [22])

        # Both rope calls must see the same pre-update value.
        self.assertEqual(rope_values[0], [21])
        self.assertEqual(rope_values[1], [21])
        self.assertEqual(rope_values[0], rope_values[1])

    def test_gemma4_dense(self):
        """Gemma 4 dense variant: K-eq-V, no per-layer inputs, no MoE."""
        from mlx_vlm.models import gemma4

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=5,
            final_logit_softcapping=30.0,
            attention_k_eq_v=True,
            enable_moe_block=False,
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

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Verify NO MoE layers
        for layer in model.language_model.model.layers:
            self.assertFalse(layer.enable_moe)

        # Verify K-eq-V on full attention layers
        for layer in model.language_model.model.layers:
            if layer.layer_type == "full_attention":
                self.assertTrue(layer.self_attn.use_k_eq_v)
                self.assertFalse(hasattr(layer.self_attn, "v_proj"))
            else:
                self.assertFalse(layer.self_attn.use_k_eq_v)

        # Verify layer_scalar exists on all layers
        for layer in model.language_model.model.layers:
            self.assertIsNotNone(layer.layer_scalar)

        # Full model forward: text-only
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

    def test_deepseekocr(self):
        from mlx_vlm.models import deepseekocr

        text_config = deepseekocr.TextConfig()
        vision_config = deepseekocr.VisionConfig(model_type="vision")
        projector_config = deepseekocr.ProjectorConfig(projector_type="linear")
        config = deepseekocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="deepseekocr",
        )
        model = deepseekocr.Model(config)
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # TODO: Add test for vision model. Ensure I can pass input type and shapes.

    def test_unlimited_ocr(self):
        from mlx_vlm.models import unlimited_ocr

        text_config = unlimited_ocr.TextConfig(
            model_type="deepseek_v2",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            vocab_size=32,
            num_key_value_heads=2,
            kv_lora_rank=None,
            q_lora_rank=None,
            qk_rope_head_dim=0,
            v_head_dim=8,
            qk_nope_head_dim=0,
            moe_intermediate_size=16,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
        )
        vision_config = unlimited_ocr.VisionConfig(
            model_type="vision",
            layers=1,
            width=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
        )
        projector_config = unlimited_ocr.ProjectorConfig(
            projector_type="linear",
            input_dim=16,
            n_embed=16,
        )
        config = unlimited_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="unlimited-ocr",
        )
        model = unlimited_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        pixel_values = mx.random.uniform(
            shape=(
                1,
                config.vision_config.image_size,
                config.vision_config.image_size,
                config.vision_config.num_channels,
            ),
            dtype=mx.float32,
        )
        vision_features = model.vision_model(pixel_values)
        self.assertEqual(vision_features.shape[0], 1)
        self.assertEqual(vision_features.shape[-1], config.vision_config.hidden_size)

        projector_input = mx.random.uniform(
            shape=(1, 4, config.projector_config.input_dim),
            dtype=mx.float32,
        )
        projected = model.projector(projector_input)
        self.assertEqual(projected.shape, (1, 4, config.projector_config.n_embed))

    def test_jina_vlm(self):
        from mlx_vlm.models import jina_vlm

        text_config = jina_vlm.TextConfig(
            model_type="jina_vlm",
            hidden_size=2048,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            additional_vocab_size=128,
            intermediate_size=6144,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            use_qk_norm=True,
        )

        vision_config = jina_vlm.VisionConfig(
            model_type="jina_vlm",
            hidden_size=1152,
            num_hidden_layers=4,
            num_attention_heads=16,
            head_dim=72,
            patch_size=14,
            image_size=378,
            num_channels=3,
            intermediate_size=4304,
            vit_layers=(-2, -4),
            output_size=2048,
            pooling_h=2,
            pooling_w=2,
            connector_hidden_size=6144,
        )

        config = jina_vlm.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="jina_vlm",
            vocab_size=151936,
        )

        model = jina_vlm.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Vision model expects patchified input from processor, skip standard test
        # Test basic forward pass with patchified input instead
        batch_size = 1
        n_patches = (
            config.vision_config.image_size // config.vision_config.patch_size
        ) ** 2
        patch_dim = (
            config.vision_config.patch_size**2 * config.vision_config.num_channels
        )
        pixel_values = mx.random.uniform(shape=(batch_size, n_patches, patch_dim))
        output, hidden_states = model.vision_model(pixel_values)
        # Check output shape matches hidden size
        self.assertEqual(output.shape[-1], config.vision_config.hidden_size)
        self.assertEqual(len(hidden_states), config.vision_config.num_hidden_layers + 1)

    def test_hunyuan_vl(self):
        from mlx_vlm.models import hunyuan_vl

        text_config = hunyuan_vl.TextConfig(
            model_type="hunyuan_vl",
            vocab_size=120818,
            org_vocab_size=120818,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            attention_head_dim=128,
            intermediate_size=3584,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            use_qk_norm=True,
            rope_theta=10000.0,
            rope_scaling={
                "alpha": 1000.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "type": "xdrope",
                "xdrope_section": [16, 16, 16, 16],
            },
            max_position_embeddings=32768,
            rms_norm_eps=1e-5,
            norm_type="rms",
            tie_word_embeddings=True,
            use_cache=True,
            initializer_range=0.02,
            routed_scaling_factor=1.0,
            bos_token_id=120000,
            eos_token_id=120020,
            eod_token_id=120020,
            pad_token_id=-1,
            pad_id=120002,
        )

        vision_config = hunyuan_vl.VisionConfig(
            model_type="hunyuan_vl",
            hidden_size=1152,
            out_hidden_size=1024,
            num_hidden_layers=5,
            num_attention_heads=16,
            intermediate_size=4304,
            patch_size=16,
            num_channels=3,
            spatial_merge_size=2,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rms_norm_eps=1e-5,
            interpolate_mode="bilinear",
            cat_extra_token=1,
            img_max_token_num=4096,
            max_vit_seq_len=16384,
            add_patchemb_bias=True,
            max_image_size=2048,
            hidden_act="gelu",
        )

        config = hunyuan_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="hunyuan_vl",
            image_token_id=120120,
            image_start_token_id=120118,
            image_end_token_id=120119,
            image_newline_token_id=120121,
            bos_token_id=120000,
            eos_token_id=120020,
            pad_token_id=-1,
            pad_id=120002,
            vocab_size=120818,
            org_vocab_size=120818,
            tie_word_embeddings=True,
        )

        model = hunyuan_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.num_channels,
            (1080, 768),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 18, 60]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_ernie4_5_moe_vl(self):
        from mlx_vlm.models import ernie4_5_moe_vl

        # Config based on baidu/ERNIE-4.5-VL-28B-A3B-Thinking (scaled down for testing)
        text_config = ernie4_5_moe_vl.TextConfig(
            model_type="ernie",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rope_theta=500000,
            rope_scaling={"type": "default", "mrope_section": [12, 12, 8]},
            tie_word_embeddings=True,
            moe_num_experts=[8, 8],
            moe_intermediate_size=[128, 64],
            moe_k=2,
            moe_layer_start_index=[1, 1],
            moe_layer_end_index=[4, 3],
            moe_num_shared_experts=1,
            use_bias=False,
        )

        vision_config = ernie4_5_moe_vl.VisionConfig(
            model_type="DFNRope_vision_transformer",
            depth=4,
            embed_dim=128,
            hidden_size=128,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            in_channels=3,
        )

        config = ernie4_5_moe_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            pixel_hidden_size=128,
            spatial_conv_size=2,
            temporal_conv_size=2,
            vocab_size=1000,
        )

        model = ernie4_5_moe_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_minicpmo(self):
        from mlx_vlm.models import minicpmo

        text_config = minicpmo.TextConfig(
            model_type="minicpmo",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=4,
            head_dim=16,
            rope_theta=10000.0,
            max_position_embeddings=2048,
        )
        vision_config = minicpmo.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=28,
            patch_size=14,
        )
        setattr(vision_config, "spatial_merge_size", 1)
        config = minicpmo.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            query_num=4,
        )
        setattr(config, "image_token_id", 1)
        setattr(config, "video_token_id", 2)
        setattr(config, "vision_start_token_id", 3)
        model = minicpmo.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=0,
        )

    def test_phi4mm(self):
        from mlx_vlm.models import phi4mm

        config = phi4mm.ModelConfig(
            text_config=phi4mm.TextConfig(
                model_type="phi4mm",
                max_position_embeddings=2048,
            ),
            vision_config=phi4mm.VisionConfig(
                model_type="siglip2_vision_model",
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_hidden_layers=2,
                patch_size=14,
                image_size=28,
                num_channels=3,
                layer_norm_eps=1e-6,
            ),
            model_type="phi4mm",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            mm_hidden_size=32,
            image_token_index=-200,
            audio_token_index=200,
            audio_processor={
                "config": {
                    "attention_dim": 32,
                    "attention_heads": 4,
                    "num_blocks": 2,
                    "linear_units": 64,
                    "input_size": 80,
                    "time_reduction": 8,
                    "kernel_size": 3,
                    "conv_channels": 32,
                    "ext_pw_out_channel": 32,
                    "depthwise_seperable_out_channel": 32,
                }
            },
        )
        model = phi4mm.Model(config)

        # Language model
        self.assertEqual(model.model_type, config.model_type)
        self.assertEqual(len(model.layers), config.num_hidden_layers)

        batch_size = 1
        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 2, config.vocab_size))
            self.assertEqual(logits.dtype, t)

        # Vision tower: SigLIP2 with NaFlex
        # Input: (B, num_patches, patch_dim) where patch_dim = P*P*C
        patch_size = config.vision_config.patch_size
        num_channels = config.vision_config.num_channels
        img_size = config.vision_config.image_size
        patch_dim = patch_size * patch_size * num_channels
        num_patches = (img_size // patch_size) ** 2
        pixel_values = mx.random.uniform(shape=(1, num_patches, patch_dim))
        features = model.vision_tower(pixel_values)
        self.assertEqual(features.shape[-1], config.vision_config.hidden_size)

        # MM projector: projects vision features to LLM hidden size
        projected = model.apply_mm_projector(features)
        self.assertEqual(projected.shape[-1], config.hidden_size)

        # Audio encoder: Conformer
        # Input: (B, T, 80) mel spectrogram features
        audio_input = mx.random.uniform(shape=(1, 100, 80))
        audio_mask = mx.ones((1, 100))
        encoded_audio, _ = model.audio_encoder(audio_input, audio_mask)
        audio_config = getattr(config, "_audio_config")
        self.assertEqual(encoded_audio.shape[-1], audio_config.attention_dim)

        # Audio projection: projects audio features to LLM hidden size
        audio_projected = model.audio_projection(encoded_audio, mode="speech")
        self.assertEqual(audio_projected.shape[-1], config.hidden_size)

    def test_glm_ocr(self):
        from mlx_vlm.models import glm_ocr

        text_config = glm_ocr.TextConfig(
            model_type="glm_ocr_text",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-5,
            max_position_embeddings=1000,
            rope_parameters={
                "rope_type": "default",
                "mrope_section": [8, 12, 12],
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000,
            },
        )

        vision_config = glm_ocr.VisionConfig(
            model_type="glm_ocr_vision",
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            out_hidden_size=128,
            spatial_merge_size=2,
            temporal_patch_size=2,
            rms_norm_eps=1e-5,
        )

        config = glm_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm_ocr",
            image_token_id=999,
            video_token_id=998,
            vocab_size=1000,
        )

        model = glm_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        rope_input_ids = mx.array([[10, 11, 999, 999, 999, 999, 999, 999, 12, 13]])
        rope_grid_thw = mx.array([[1, 4, 6]], dtype=mx.int64)
        position_ids, rope_deltas = model.language_model.get_rope_index(
            rope_input_ids,
            image_grid_thw=rope_grid_thw,
        )
        expected_position_ids = mx.array(
            [
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
                ],
                [
                    [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
                ],
                [
                    [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
                ],
            ],
            dtype=rope_input_ids.dtype,
        )
        self.assertTrue(mx.array_equal(position_ids, expected_position_ids).item())
        self.assertEqual(rope_deltas.item(), -3)

        # Test vision model with proper input format
        # grid_thw format: [temporal, height/patch, width/patch]
        # For grid_thw = [2, 14, 14], we have 2*14*14 = 392 patches
        # Height/width must be divisible by spatial_merge_size (2)
        grid_thw = mx.array([[2, 14, 14]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor - flat array that gets reshaped internally
        # Shape: (num_patches * in_channels * temporal_patch_size * patch_size * patch_size)
        total_elements = (
            num_patches
            * config.vision_config.in_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size
            * config.vision_config.patch_size
        )
        pixel_values = mx.random.uniform(shape=(total_elements,))

        # Forward pass
        hidden_states = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have:
        # temporal * (height/spatial_merge) * (width/spatial_merge)
        # = 2 * (14/2) * (14/2) = 2 * 7 * 7 = 98 patches
        expected_patches = int(
            grid_thw[0, 0]
            * (grid_thw[0, 1] // config.vision_config.spatial_merge_size)
            * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        )
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

    def test_phi4_siglip(self):
        from mlx_vlm.models import phi4_siglip

        text_config = phi4_siglip.TextConfig(
            model_type="phi4-siglip",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            partial_rotary_factor=1.0,
        )

        vision_config = phi4_siglip.VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            patch_size=14,
            num_patches=256,
        )

        config = phi4_siglip.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="phi4-siglip",
            mm_hidden_size=16,
        )

        model = phi4_siglip.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_moondream2(self):
        from mlx_vlm.models import moondream2

        text_config = moondream2.TextConfig(
            model_type="moondream2",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=256,
            partial_rotary_factor=0.5,
            rms_norm_eps=1e-5,
        )

        vision_config = moondream2.VisionConfig(
            model_type="moondream2_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=14,
            crop_size=28,
            in_channels=3,
            proj_inner_dim=64,
            proj_out_dim=64,
            attention_bias=True,
            layer_norm_eps=1e-5,
        )

        config = moondream2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="moondream2",
        )
        model = moondream2.Model(config)

        self.language_test_runner(
            model.language_model,
            text_config.model_type,
            text_config.vocab_size,
            text_config.num_hidden_layers,
        )

        batch_size = 1
        crop_size = vision_config.crop_size
        pixel_values = mx.random.uniform(shape=(batch_size, crop_size, crop_size, 3))
        features = model.vision.encoder(pixel_values)
        grid_size = crop_size // vision_config.patch_size
        num_patches = grid_size * grid_size
        self.assertEqual(
            features.shape, (batch_size, num_patches, vision_config.hidden_size)
        )

    def test_moondream3(self):
        from mlx_vlm.models import moondream3

        text_config = moondream3.TextConfig(
            model_type="moondream3",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=256,
            rope_theta=1500000.0,
            rope_dim=8,
            rms_norm_eps=1e-5,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_start_layer=2,
            attention_bias=True,
            prefix_attn=5,
        )

        vision_config = moondream3.VisionConfig(
            model_type="moondream3_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=14,
            crop_size=28,
            in_channels=3,
            proj_inner_dim=64,
            proj_out_dim=64,
            attention_bias=True,
            layer_norm_eps=1e-6,
        )

        config = moondream3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="moondream3",
        )

        model = moondream3.Model(config)

        # Language model test
        self.language_test_runner(
            model.language_model,
            text_config.model_type,
            text_config.vocab_size,
            text_config.num_hidden_layers,
        )

        # Vision encoder test
        batch_size = 1
        crop_size = vision_config.crop_size
        pixel_values = mx.random.uniform(shape=(batch_size, crop_size, crop_size, 3))
        features = model.vision.encoder(pixel_values)
        grid_size = crop_size // vision_config.patch_size
        num_patches = grid_size * grid_size
        self.assertEqual(
            features.shape, (batch_size, num_patches, vision_config.hidden_size)
        )

        # Vision projection: concat global+local -> project
        combined = mx.concatenate([features, features], axis=-1)
        projected = model.vision.proj_mlp(combined)
        self.assertEqual(
            projected.shape,
            (batch_size, num_patches, vision_config.proj_out_dim),
        )

        # Full model forward with vision
        # Input: BOS + num_patches placeholders + 2 text tokens
        input_ids = mx.zeros((1, 1 + num_patches + 2), dtype=mx.int32)
        input_ids[0, -2:] = mx.array([1, 2])
        outputs = model(
            input_ids,
            pixel_values=pixel_values,
            num_crops=[1],
            crop_layouts=[(1, 1)],
        )
        self.assertEqual(outputs.logits.shape[-1], text_config.vocab_size)

    def test_mistral4(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            model_type="mistral4",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=256,
            max_position_embeddings=1024,
            rope_traditional=False,
            rope_parameters={
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "factor": 4.0,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 512,
            },
            tie_word_embeddings=False,
            attention_bias=False,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            first_k_dense_replace=0,
        )

        vision_config = mistral3.VisionConfig(
            model_type="pixtral",
            hidden_size=1024,
            num_hidden_layers=2,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            num_channels=3,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mistral3",
        )

        model = mistral3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_falcon_ocr(self):
        from mlx_vlm.models import falcon_ocr

        text_config = falcon_ocr.TextConfig(
            model_type="falcon_ocr",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
            vocab_size=256,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

        vision_config = falcon_ocr.VisionConfig(
            model_type="falcon_ocr",
            spatial_patch_size=4,
            temporal_patch_size=1,
            channel_size=3,
        )

        config = falcon_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="falcon_ocr",
            vocab_size=256,
            img_id=227,
            image_cls_token_id=244,
            img_end_id=230,
        )

        model = falcon_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_falcon_perception(self):
        from mlx_vlm.models import falcon_perception

        text_config = falcon_perception.TextConfig(
            model_type="falcon_perception",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
            vocab_size=256,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

        vision_config = falcon_perception.VisionConfig(
            model_type="falcon_perception",
            spatial_patch_size=4,
            temporal_patch_size=1,
            channel_size=3,
        )

        config = falcon_perception.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="falcon_perception",
            vocab_size=256,
            img_id=227,
            image_cls_token_id=244,
            img_end_id=230,
            coord_token_id=240,
            size_token_id=241,
            seg_token_id=262,
            coord_enc_dim=64,
            coord_dec_dim=128,
            coord_out_dim=256,
            size_enc_dim=64,
            size_dec_dim=128,
            size_out_dim=256,
            do_segmentation=False,
            segm_out_dim=64,
            num_segm_layers=1,
        )

        model = falcon_perception.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite_vision(self):
        from mlx_vlm.models import granite_vision

        text_config = granite_vision.TextConfig(
            model_type="granite",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=300000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=8.0,
        )

        vision_config = granite_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=56,
            patch_size=14,
        )

        config = granite_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite_vision",
            vision_feature_layer=[-2, -1],
            vision_feature_select_strategy="full",
        )

        model = granite_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite4_vision(self):
        from mlx_vlm.models import granite4_vision

        text_config = granite4_vision.TextConfig(
            model_type="granitemoehybrid",
            hidden_size=64,
            intermediate_size=128,
            shared_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=10000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=10.0,
        )

        vision_config = granite4_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=48,
            patch_size=16,
        )

        config = granite4_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite4_vision",
            deepstack_layer_map=[[-1, 0]],
            use_spatial_sampling=False,
            downsample_rate="3/3",
            use_image_newline_parameter=False,
        )

        model = granite4_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite4_1_vision(self):
        from mlx_vlm.models import granite4_vision

        text_config = granite4_vision.TextConfig(
            model_type="granite",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=10000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=10.0,
        )

        vision_config = granite4_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=48,
            patch_size=16,
        )

        config = granite4_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite4_vision",
            deepstack_layer_map=[[-1, 0]],
            use_spatial_sampling=False,
            downsample_rate="3/3",
            use_image_newline_parameter=False,
        )

        model = granite4_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=0,
        )

    def test_youtu_vl(self):
        from mlx_vlm.models import youtu_vl

        text_config = youtu_vl.TextConfig(
            model_type="youtu_vl",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1024,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=32,
            rope_theta=500000.0,
            rope_interleave=True,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )
        vision_config = youtu_vl.VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=128,
            out_hidden_size=256,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            window_size=64,
            fullatt_block_indexes=[1],
        )
        config = youtu_vl.ModelConfig(
            model_type="youtu_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=100,
            video_token_id=101,
            vocab_size=1024,
        )
        model = youtu_vl.Model(config)

        # Language model: MLA with absorb — fp32/fp16 forward + cached decode check
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Vision tower takes packed patches + spatial_shapes:
        #   pixel_values: (num_patches, patch_size**2 * channels)
        #   spatial_shapes: (batch, 2) — (h_patches, w_patches)
        patch_dim = vision_config.patch_size**2 * vision_config.num_channels
        h_p, w_p = 4, 4
        num_patches = h_p * w_p
        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.num_channels,
            (num_patches, patch_dim),
            vision_feature_layer=-1,
            spatial_shapes=mx.array([[h_p, w_p]], dtype=mx.int32),
        )

        # sanitize splits kv_b_proj per-head into embed_q (k) + unembed_out (v)
        H, nope, v_head = 4, 32, 32
        kv_rank = text_config.kv_lora_rank
        w = mx.arange(H * (nope + v_head) * kv_rank, dtype=mx.float32).reshape(
            H * (nope + v_head), kv_rank
        )
        sanitized = model.sanitize(
            {
                "model.layers.0.self_attn.kv_b_proj.weight": w,
                "lm_head.weight": mx.zeros((1, 1)),  # tied; must be dropped
            }
        )
        prefix = "language_model.model.layers.0.self_attn"
        self.assertNotIn(f"{prefix}.kv_b_proj.weight", sanitized)
        self.assertNotIn("language_model.lm_head.weight", sanitized)
        self.assertEqual(
            sanitized[f"{prefix}.embed_q.weight"].shape, (H, kv_rank, nope)
        )
        self.assertEqual(
            sanitized[f"{prefix}.unembed_out.weight"].shape, (H, v_head, kv_rank)
        )

    def test_zaya1_vl(self):
        from mlx_vlm.models import zaya1_vl
        from mlx_vlm.models.zaya1_vl.language import CCA, ZayaRouter

        text_config = zaya1_vl.TextConfig(
            model_type="zaya1_vl",
            hidden_size=8,
            ffn_hidden_size=16,
            num_hidden_layers=1,
            num_experts=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_query_groups=1,
            head_dim=4,
            zaya_mlp_expansion=4,
            vocab_size=32,
            vision_lora=False,
        )
        vision_config = zaya1_vl.VisionConfig(
            model_type="qwen2_5_vl",
            depth=1,
            hidden_size=8,
            intermediate_size=16,
            out_hidden_size=8,
            num_heads=2,
            image_size=4,
            patch_size=2,
            in_channels=3,
            spatial_patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            window_size=4,
            fullatt_block_indexes=[0],
        )
        config = zaya1_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="zaya1_vl",
            image_token_id=31,
            vocab_size=32,
        )
        model = zaya1_vl.Model(config)
        layer = model.language_model.model.layers[0]
        self.assertIsInstance(model.language_model.model.final_norm, nn.RMSNorm)
        self.assertIsInstance(layer.attn.input_norm, nn.RMSNorm)
        self.assertIsInstance(layer.mlp.input_norm, nn.RMSNorm)
        self.assertIsInstance(layer.mlp.zaya_block.router.rmsnorm_eda, nn.RMSNorm)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )
        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (4, 12),
            vision_feature_layer=-1,
            grid_thw=mx.array([[1, 2, 2]], dtype=mx.int64),
        )

        router = ZayaRouter(text_config, layer_number=1)
        self.assertIsInstance(router.rmsnorm_eda, nn.RMSNorm)
        self.assertEqual(router.rmsnorm_eda.eps, text_config.norm_epsilon)

        cca = CCA(text_config, layer_number=0)
        for linear in (cca.linear_q, cca.linear_k, cca.val_proj1, cca.val_proj2):
            linear.weight = mx.zeros_like(linear.weight)
        cca.linear_q.weight[0, 0] = 1e-4
        cca.linear_k.weight[0, 0] = 1e-4
        for conv in cca.conv_qk:
            conv.weight = mx.zeros_like(conv.weight)
            conv.bias = mx.zeros_like(conv.bias)

        x_cca = mx.zeros((1, 1, text_config.hidden_size), dtype=mx.float32)
        x_cca[0, 0, 0] = 1.0
        query, _, _ = cca(x_cca)
        expected_query = mx.zeros((1, 1, text_config.hidden_size), dtype=mx.float32)
        for h in range(text_config.num_attention_heads):
            expected_query[0, 0, h * text_config.head_dim] = 2.0
        mx.eval(query, expected_query)
        self.assertTrue(mx.allclose(query, expected_query, rtol=1e-5).item())

    # ------------------------------------------------------------------
    # Structural tests for models with zero prior test coverage
    # ------------------------------------------------------------------

    def test_llada2_moe_language_model(self):
        # Config source: mlx_vlm/models/llada2_moe/config.py
        from mlx_vlm.models import llada2_moe

        config = llada2_moe.ModelConfig(
            model_type="llada2_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            num_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            moe_intermediate_size=32,
            first_k_dense_replace=1,
            tie_word_embeddings=False,
        )

        model = llada2_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_nemotron_labs_diffusion_language_model(self):
        # Config source: mlx_vlm/models/nemotron_labs_diffusion/config.py
        from mlx_vlm.models import nemotron_labs_diffusion

        config = nemotron_labs_diffusion.ModelConfig(
            model_type="nemotron_labs_diffusion",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            mask_token_id=100,
            dlm_paradigm="autoregressive",
        )

        model = nemotron_labs_diffusion.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_step3p7_language_model(self):
        # Config source: mlx_vlm/models/step3p7/config.py
        from mlx_vlm.models import step3p7

        text_config = step3p7.TextConfig(
            model_type="step3p5",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_attention_groups=2,
            num_hidden_layers=2,
            vocab_size=128,
            rms_norm_eps=1e-5,
            moe_intermediate_size=32,
            moe_num_experts=4,
            moe_top_k=2,
            head_dim=16,
            share_expert_dim=32,
            moe_layers_enum=[1],
            tie_word_embeddings=False,
        )

        vision_config = step3p7.VisionConfig(
            model_type="perception_encoder",
            width=64,
            layers=2,
            heads=4,
            patch_size=14,
            image_size=56,
        )

        config = step3p7.ModelConfig(
            model_type="step3p7",
            text_config=text_config,
            vision_config=vision_config,
            vocab_size=128,
        )

        model = step3p7.Model(config)

        # step3p7 LanguageModel does not expose model_type, so we skip
        # language_test_runner and test the forward pass directly.
        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, text_config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_deepseekocr_2_language_model(self):
        # Config source: mlx_vlm/models/deepseekocr_2/config.py
        from mlx_vlm.models import deepseekocr_2

        text_config = deepseekocr_2.TextConfig(
            model_type="deepseek_v2",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            first_k_dense_replace=0,
        )

        vision_config = deepseekocr_2.VisionConfig(
            model_type="vision",
            layers=2,
            width=64,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            image_size=56,
            patch_size=14,
            params={
                "qwen2": {
                    "dim": 64,
                    "layers": 2,
                    "heads": 4,
                    "kv_heads": 2,
                    "intermediate_size": 128,
                    "rms_norm_eps": 1e-6,
                },
                "sam": {},
            },
        )

        projector_config = deepseekocr_2.ProjectorConfig(
            input_dim=64,
            n_embed=64,
            depth=2,
            mlp_ratio=1,
        )

        config = deepseekocr_2.ModelConfig(
            model_type="deepseekocr_2",
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            vocab_size=128,
        )

        model = deepseekocr_2.Model(config)

        self.language_test_runner(
            model.language_model,
            text_config.model_type,
            text_config.vocab_size,
            text_config.num_hidden_layers,
        )

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, text_config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_minimax_m3_vl_language_model(self):
        # Config source: mlx_vlm/models/minimax_m3_vl/config.py
        from mlx_vlm.models import minimax_m3_vl

        text_config = minimax_m3_vl.TextConfig(
            model_type="minimax_m3",
            hidden_size=64,
            intermediate_size=32,
            dense_intermediate_size=128,
            shared_intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            vocab_size=128,
            tie_word_embeddings=False,
            num_local_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_layer_freq=[0, 1],
            layer_types=["minimax_m3_dense", "minimax_m3_sparse"],
        )

        vision_config = minimax_m3_vl.VisionConfig(
            model_type="clip_vision_model",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            image_size=56,
            patch_size=14,
        )

        config = minimax_m3_vl.ModelConfig(
            model_type="minimax_m3_vl",
            text_config=text_config,
            vision_config=vision_config,
            projector_hidden_size=64,
            vocab_size=128,
        )

        model = minimax_m3_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            text_config.model_type,
            text_config.vocab_size,
            text_config.num_hidden_layers,
        )

        logits = model(mx.array([[1, 2, 3, 4]])).logits
        self.assertEqual(logits.shape, (1, 4, text_config.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_rfdetr_forward(self):
        # Config source: mlx_vlm/models/rfdetr/config.py
        from mlx_vlm.models import rfdetr

        config = rfdetr.ModelConfig(
            model_type="rf-detr",
            resolution=56,
            hidden_dim=256,
            num_classes=10,
            num_queries=4,
            dec_layers=1,
            sa_nheads=4,
            ca_nheads=4,
            dec_n_points=2,
            group_detr=1,
            patch_size=14,
            num_windows=1,
            out_feature_indexes=[2, 5, 8, 11],
            projector_scale=["P4"],
        )

        model = rfdetr.Model(config)
        model.eval()

        pixel = mx.random.normal((1, 56, 56, 3))
        out = model(pixel)
        mx.eval(out["pred_logits"], out["pred_boxes"])

        # num_classes + 1 for background
        self.assertEqual(out["pred_logits"].shape, (1, 4, 11))
        self.assertEqual(out["pred_boxes"].shape, (1, 4, 4))
        self.assertTrue(mx.all(mx.isfinite(out["pred_logits"])).item())
        self.assertTrue(mx.all(mx.isfinite(out["pred_boxes"])).item())

    def test_sam3_1_config_and_model(self):
        # Config source: mlx_vlm/models/sam3_1/config.py
        from mlx_vlm.models.sam3_1 import Model, ModelConfig

        config = ModelConfig(model_type="sam3.1_video")
        model = Model(config)

        # Verify structure
        self.assertEqual(config.model_type, "sam3.1_video")
        self.assertIsNotNone(model.detector_model)
        self.assertIsNotNone(model.tracker_model)
        # Verify sub-configs were populated
        self.assertIsNotNone(config.detector_config)
        self.assertIsNotNone(config.tracker_config)
        self.assertIsNotNone(config.text_config)
        self.assertIsNotNone(config.vision_config)

    def test_sam3d_body_model(self):
        # Config source: mlx_vlm/models/sam3d_body/config.py
        from mlx_vlm.models.sam3d_body import Model
        from mlx_vlm.models.sam3d_body.config import SAM3DConfig

        config = SAM3DConfig(
            embed_dim=64,
            depth=2,
            num_heads=4,
            head_dim=16,
            patch_size=16,
            image_size=(64, 48),
            ffn_ratio=2.0,
            num_storage_tokens=2,
            decoder_dim=32,
            decoder_depth=2,
            decoder_heads=4,
            decoder_head_dim=8,
            decoder_mlp_dim=64,
            num_joints=127,
            num_vertices=18439,
            num_faces=36874,
            num_shape_comps=45,
            num_face_comps=72,
            pose_output_dim=519,
            camera_output_dim=3,
            num_point_embeddings=70,
            prompt_embed_dim=64,
        )

        model = Model(config)
        model.eval()

        # Verify model construction and backbone forward pass
        self.assertEqual(config.model_type, "sam3d_body")
        self.assertIsNotNone(model.backbone)
        self.assertIsNotNone(model.decoder)
        self.assertIsNotNone(model.head_pose)
        self.assertIsNotNone(model.head_camera)

        image = mx.random.normal((1, 64, 48, 3))
        features = model.backbone(image)
        mx.eval(features)
        # patch grid: 64/16=4 height, 48/16=3 width
        self.assertEqual(features.shape, (1, 4, 3, config.embed_dim))
        self.assertTrue(mx.all(mx.isfinite(features)).item())


class TestGetInputEmbeddings(unittest.TestCase):
    """Test that all models with get_input_embeddings return InputEmbeddingsFeatures."""

    def _check_returns_input_embeddings_features(self, model, model_name):
        """Helper to test get_input_embeddings returns InputEmbeddingsFeatures."""
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids=input_ids)
        self.assertIsInstance(
            result,
            InputEmbeddingsFeatures,
            f"{model_name}: expected InputEmbeddingsFeatures, got {type(result).__name__}",
        )
        self.assertIsNotNone(result.inputs_embeds)

    def _assert_qwen_request_owned_mrope_kwargs(self, model):
        stale_position_ids = mx.array([[[9, 9, 9]]], dtype=mx.int32)
        stale_rope_deltas = mx.array([[99]], dtype=mx.int32)
        model.language_model._position_ids = stale_position_ids
        model.language_model._rope_deltas = stale_rope_deltas

        result = model.get_input_embeddings(
            input_ids=mx.array([[1, 2, 3]], dtype=mx.int32)
        )

        self.assertIsNotNone(result.position_ids)
        self.assertIsNotNone(result.rope_deltas)
        self.assertEqual(result.position_ids.shape, (1, 3))
        self.assertEqual(result.position_ids.tolist(), [[0, 1, 2]])
        self.assertEqual(result.rope_deltas.tolist(), [[0]])
        self.assertTrue(
            mx.array_equal(
                model.language_model._position_ids, stale_position_ids
            ).item()
        )
        self.assertTrue(
            mx.array_equal(model.language_model._rope_deltas, stale_rope_deltas).item()
        )

    def test_llava_input_embeddings(self):
        from mlx_vlm.models import llava

        model = llava.Model(
            llava.ModelConfig(
                text_config=llava.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=llava.VisionConfig(
                    model_type="clip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava")

    def test_llava_bunny_input_embeddings(self):
        from mlx_vlm.models import llava_bunny

        model = llava_bunny.Model(
            llava_bunny.ModelConfig(
                text_config=llava_bunny.TextConfig(
                    model_type="qwen2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                ),
                vision_config=llava_bunny.VisionConfig(
                    model_type="siglip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava-qwen2",
                auto_map={
                    "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                    "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
                },
                hidden_size=16,
                mm_hidden_size=16,
                image_token_index=-200,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava_bunny")

    def test_llava_next_input_embeddings(self):
        from mlx_vlm.models import llava_next

        model = llava_next.Model(
            llava_next.ModelConfig(
                text_config=llava_next.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=llava_next.VisionConfig(
                    model_type="clip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava_next")

    def test_qwen2_vl_input_embeddings(self):
        from mlx_vlm.models import qwen2_vl

        model = qwen2_vl.Model(
            qwen2_vl.ModelConfig(
                text_config=qwen2_vl.TextConfig(
                    model_type="qwen2_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen2_vl.VisionConfig(
                    model_type="qwen2_vl",
                    depth=1,
                    embed_dim=16,
                    hidden_size=16,
                    num_heads=2,
                    image_size=28,
                    patch_size=14,
                    in_channels=3,
                ),
                model_type="qwen2_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen2_vl")
        self._assert_qwen_request_owned_mrope_kwargs(model)

    def test_qwen2_5_vl_input_embeddings(self):
        from mlx_vlm.models import qwen2_5_vl

        model = qwen2_5_vl.Model(
            qwen2_5_vl.ModelConfig(
                text_config=qwen2_5_vl.TextConfig(
                    model_type="qwen2_5_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen2_5_vl.VisionConfig(
                    model_type="qwen2_5_vl",
                    depth=1,
                    hidden_size=16,
                    num_heads=2,
                    out_hidden_size=16,
                    image_size=28,
                    patch_size=14,
                    in_channels=3,
                    fullatt_block_indexes=[0],
                    window_size=14,
                ),
                model_type="qwen2_5_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen2_5_vl")
        self._assert_qwen_request_owned_mrope_kwargs(model)

    def test_qwen3_vl_input_embeddings(self):
        from mlx_vlm.models import qwen3_vl

        model = qwen3_vl.Model(
            qwen3_vl.ModelConfig(
                text_config=qwen3_vl.TextConfig(
                    model_type="qwen3_vl_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    head_dim=8,
                    rope_theta=1000.0,
                    max_position_embeddings=1000,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen3_vl.VisionConfig(
                    model_type="qwen3_vl",
                    depth=1,
                    hidden_size=16,
                    num_heads=2,
                    out_hidden_size=16,
                    patch_size=14,
                    in_channels=3,
                    num_position_embeddings=4,
                    deepstack_visual_indexes=[],
                ),
                model_type="qwen3_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_vl")
        self._assert_qwen_request_owned_mrope_kwargs(model)

    def test_paligemma_input_embeddings(self):
        from mlx_vlm.models import paligemma

        model = paligemma.Model(
            paligemma.ModelConfig(
                text_config=paligemma.TextConfig(
                    model_type="gemma",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-6,
                ),
                vision_config=paligemma.VisionConfig(
                    model_type="siglip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                    projection_dim=16,
                ),
                model_type="paligemma",
                image_token_index=31,
                hidden_size=16,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "paligemma")

    def test_idefics2_input_embeddings(self):
        from mlx_vlm.models import idefics2

        model = idefics2.Model(
            idefics2.ModelConfig(
                text_config=idefics2.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=idefics2.VisionConfig(
                    model_type="idefics2",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                perceiver_config=idefics2.PerceiverConfig(
                    model_type="idefics2Perceiver",
                    resampler_n_latents=2,
                    resampler_depth=1,
                    resampler_n_heads=2,
                    resampler_head_dim=8,
                    num_key_value_heads=2,
                ),
                model_type="idefics2",
                image_token_index=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "idefics2")

    def test_idefics3_input_embeddings(self):
        from mlx_vlm.models import idefics3

        model = idefics3.Model(
            idefics3.ModelConfig(
                text_config=idefics3.TextConfig(
                    model_type="idefics3",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=idefics3.VisionConfig(
                    model_type="idefics3",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="idefics3",
                image_token_id=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "idefics3")

    def test_gemma3_input_embeddings(self):
        from mlx_vlm.models import gemma3

        model = gemma3.Model(
            gemma3.ModelConfig(
                text_config=gemma3.TextConfig(
                    model_type="gemma3",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-6,
                    num_key_value_heads=2,
                    head_dim=8,
                ),
                vision_config=gemma3.VisionConfig(
                    model_type="gemma3",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="gemma3",
                hidden_size=16,
                pad_token_id=0,
                image_token_index=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma3")

    def test_pixtral_input_embeddings(self):
        from mlx_vlm.models import pixtral

        model = pixtral.Model(
            pixtral.ModelConfig(
                text_config=pixtral.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=pixtral.VisionConfig(
                    model_type="pixtral",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="pixtral",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "pixtral")

    def test_mistral3_input_embeddings(self):
        from mlx_vlm.models import mistral3

        model = mistral3.Model(
            mistral3.ModelConfig(
                text_config=mistral3.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    head_dim=8,
                    layer_types=["full_attention"],
                ),
                vision_config=mistral3.VisionConfig(
                    model_type="pixtral",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="mistral3",
            )
        )
        self._check_returns_input_embeddings_features(model, "mistral3")

    def test_multi_modality_input_embeddings(self):
        from mlx_vlm.models import multi_modality

        model = multi_modality.Model(
            multi_modality.ModelConfig(
                text_config=multi_modality.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-6,
                ),
                vision_config=multi_modality.VisionConfig(
                    model_type="vision",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                    params={},
                ),
                projector_config=multi_modality.ProjectorConfig(
                    cls="MlpProjector",
                    model_type="projector",
                    params={
                        "depth": 1,
                        "input_dim": 16,
                        "n_embed": 16,
                        "projector_type": "mlp_gelu",
                    },
                ),
                model_type="multi_modality",
                image_token_index=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "multi_modality")

    def test_aya_vision_input_embeddings(self):
        from mlx_vlm.models import aya_vision

        model = aya_vision.Model(
            aya_vision.ModelConfig(
                text_config=aya_vision.TextConfig(
                    model_type="aya_vision",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                ),
                vision_config=aya_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=16,
                    num_attention_heads=2,
                    patch_size=14,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    image_size=28,
                ),
                model_type="aya_vision",
            )
        )
        self._check_returns_input_embeddings_features(model, "aya_vision")

    def test_deepseek_vl_v2_input_embeddings(self):
        from mlx_vlm.models import deepseek_vl_v2

        model = deepseek_vl_v2.Model(
            deepseek_vl_v2.ModelConfig(
                text_config=deepseek_vl_v2.TextConfig(
                    model_type="deepseek_v2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    kv_lora_rank=8,
                    q_lora_rank=16,
                    qk_rope_head_dim=8,
                    v_head_dim=8,
                    qk_nope_head_dim=0,
                    moe_intermediate_size=16,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    num_experts_per_tok=1,
                ),
                vision_config=deepseek_vl_v2.VisionConfig(
                    model_type="vision",
                    layers=1,
                    width=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                ),
                projector_config=deepseek_vl_v2.ProjectorConfig(
                    projector_type="downsample_mlp_gelu",
                    input_dim=16,
                    n_embed=16,
                ),
                model_type="deepseek_vl_v2",
            )
        )
        self._check_returns_input_embeddings_features(model, "deepseek_vl_v2")

    def test_deepseekocr_input_embeddings(self):
        from mlx_vlm.models import deepseekocr

        model = deepseekocr.Model(
            deepseekocr.ModelConfig(
                text_config=deepseekocr.TextConfig(
                    model_type="deepseek_v2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    kv_lora_rank=8,
                    q_lora_rank=16,
                    qk_rope_head_dim=0,
                    v_head_dim=8,
                    qk_nope_head_dim=0,
                    moe_intermediate_size=16,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    num_experts_per_tok=1,
                ),
                vision_config=deepseekocr.VisionConfig(
                    model_type="vision",
                    layers=1,
                    width=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                ),
                projector_config=deepseekocr.ProjectorConfig(
                    projector_type="linear",
                    input_dim=16,
                    n_embed=16,
                ),
                model_type="deepseekocr",
            )
        )
        self._check_returns_input_embeddings_features(model, "deepseekocr")

    def test_unlimited_ocr_input_embeddings(self):
        from mlx_vlm.models import unlimited_ocr

        model = unlimited_ocr.Model(
            unlimited_ocr.ModelConfig(
                text_config=unlimited_ocr.TextConfig(
                    model_type="deepseek_v2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    kv_lora_rank=None,
                    q_lora_rank=None,
                    qk_rope_head_dim=0,
                    v_head_dim=8,
                    qk_nope_head_dim=0,
                    moe_intermediate_size=16,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    num_experts_per_tok=1,
                ),
                vision_config=unlimited_ocr.VisionConfig(
                    model_type="vision",
                    layers=1,
                    width=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                ),
                projector_config=unlimited_ocr.ProjectorConfig(
                    projector_type="linear",
                    input_dim=16,
                    n_embed=16,
                ),
                model_type="unlimited-ocr",
            )
        )
        self._check_returns_input_embeddings_features(model, "unlimited_ocr")

    def test_falcon_ocr_input_embeddings(self):
        from mlx_vlm.models import falcon_ocr

        model = falcon_ocr.Model(
            falcon_ocr.ModelConfig(
                text_config=falcon_ocr.TextConfig(
                    model_type="falcon_ocr",
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    head_dim=16,
                    num_key_value_heads=2,
                    vocab_size=256,
                    intermediate_size=128,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=512,
                    rope_theta=10000.0,
                    tie_word_embeddings=False,
                ),
                vision_config=falcon_ocr.VisionConfig(
                    model_type="falcon_ocr",
                    spatial_patch_size=4,
                    temporal_patch_size=1,
                    channel_size=3,
                ),
                model_type="falcon_ocr",
                vocab_size=256,
                img_id=227,
                image_cls_token_id=244,
                img_end_id=230,
            )
        )
        self._check_returns_input_embeddings_features(model, "falcon_ocr")

    def test_falcon_perception_input_embeddings(self):
        from mlx_vlm.models import falcon_perception

        model = falcon_perception.Model(
            falcon_perception.ModelConfig(
                text_config=falcon_perception.TextConfig(
                    model_type="falcon_perception",
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    head_dim=16,
                    num_key_value_heads=2,
                    vocab_size=256,
                    intermediate_size=128,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=512,
                    rope_theta=10000.0,
                    tie_word_embeddings=False,
                ),
                vision_config=falcon_perception.VisionConfig(
                    model_type="falcon_perception",
                    spatial_patch_size=4,
                    temporal_patch_size=1,
                    channel_size=3,
                ),
                model_type="falcon_perception",
                vocab_size=256,
                img_id=227,
                image_cls_token_id=244,
                img_end_id=230,
                coord_token_id=240,
                size_token_id=241,
                seg_token_id=262,
                coord_enc_dim=64,
                coord_dec_dim=128,
                coord_out_dim=256,
                size_enc_dim=64,
                size_dec_dim=128,
                size_out_dim=256,
                do_segmentation=False,
                segm_out_dim=64,
                num_segm_layers=1,
            )
        )
        self._check_returns_input_embeddings_features(model, "falcon_perception")

    def test_fastvlm_input_embeddings(self):
        from mlx_vlm.models import fastvlm

        model = fastvlm.Model(
            fastvlm.ModelConfig(
                text_config=fastvlm.TextConfig(
                    model_type="fastvlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                ),
                vision_config=fastvlm.VisionConfig(
                    model_type="llava_qwen2",
                    hidden_size=16,
                ),
                model_type="llava_qwen2",
                mm_hidden_size=16,
            )
        )
        self._check_returns_input_embeddings_features(model, "fastvlm")

    def test_florence2_input_embeddings(self):
        from mlx_vlm.models import florence2

        model = florence2.Model(
            florence2.ModelConfig(
                vision_config=florence2.VisionConfig(
                    model_type="davit",
                    hidden_size=16,
                    in_chans=3,
                    depths=[1],
                    dim_embed=[16],
                    num_heads=[2],
                    num_groups=[2],
                    patch_size=[7],
                    patch_stride=[4],
                    patch_padding=[3],
                    patch_prenorm=[False],
                ),
                text_config=florence2.TextConfig(
                    d_model=16,
                    encoder_attention_heads=2,
                    decoder_attention_heads=2,
                    encoder_ffn_dim=32,
                    decoder_ffn_dim=32,
                    encoder_layers=1,
                    decoder_layers=1,
                    vocab_size=32,
                ),
                model_type="florence2",
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "florence2")

    def test_gemma3n_input_embeddings(self):
        from mlx_vlm.models import gemma3n

        model = gemma3n.Model(
            gemma3n.ModelConfig(
                text_config=gemma3n.TextConfig(
                    model_type="gemma3n",
                    hidden_size=16,
                    num_hidden_layers=2,
                    intermediate_size=[32, 32],
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    vocab_size_per_layer_input=32,
                    hidden_size_per_layer_input=16,
                    altup_num_inputs=2,
                    laurel_rank=8,
                    layer_types=["sliding_attention", "full_attention"],
                    sliding_window_pattern=1,
                ),
                vision_config=gemma3n.VisionConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=4,
                    vocab_offset=32,
                ),
                audio_config=gemma3n.AudioConfig(
                    vocab_size=4,
                    vocab_offset=36,
                    hidden_size=16,
                ),
                model_type="gemma3n",
                hidden_size=16,
                vocab_size=40,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma3n")

    def test_gemma4_input_embeddings(self):
        from mlx_vlm.models import gemma4

        model = gemma4.Model(
            gemma4.ModelConfig(
                text_config=gemma4.TextConfig(
                    model_type="gemma4_text",
                    hidden_size=16,
                    num_hidden_layers=2,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    global_head_dim=8,
                    vocab_size=32,
                    vocab_size_per_layer_input=32,
                    hidden_size_per_layer_input=8,
                    num_kv_shared_layers=0,
                    sliding_window=32,
                    sliding_window_pattern=1,
                ),
                vision_config=gemma4.VisionConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    patch_size=16,
                    pooling_kernel_size=2,
                    default_output_length=4,
                    position_embedding_size=64,
                    use_clipped_linears=False,
                ),
                model_type="gemma4",
                hidden_size=16,
                vocab_size=32,
                image_token_id=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma4")

    def test_gemma4_unified_input_embeddings(self):
        from mlx_vlm.models import gemma4_unified

        model = gemma4_unified.Model(
            gemma4_unified.ModelConfig(
                text_config=gemma4_unified.TextConfig(
                    hidden_size=16,
                    num_hidden_layers=2,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    num_global_key_value_heads=1,
                    head_dim=8,
                    global_head_dim=8,
                    vocab_size=32,
                    hidden_size_per_layer_input=0,
                    sliding_window=32,
                    sliding_window_pattern=1,
                    attention_k_eq_v=True,
                ),
                vision_config=None,
                audio_config=None,
                vocab_size=32,
                hidden_size=16,
                image_token_id=31,
                audio_token_id=30,
                video_token_id=None,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma4_unified")

        text_only_types = mx.array([[0, 0, 0]])
        visual_types = mx.array([[0, 1, 1, 0]])
        mixed_audio_types = mx.array([[0, 1, 1, 3]])

        # Text-only Gemma4 unified prompts should keep chunked prefill enabled.
        model.get_input_embeddings(
            mx.array([[1, 2, 3]]), mm_token_type_ids=text_only_types
        )
        self.assertFalse(model.no_chunked_prefill)
        self.assertFalse(model.language_model.no_chunked_prefill)

        # Visual token spans need the bidirectional mask overlay, so do not chunk.
        model.get_input_embeddings(
            mx.array([[1, 2, 3, 4]]), mm_token_type_ids=visual_types
        )
        self.assertTrue(model.no_chunked_prefill)
        self.assertTrue(model.language_model.no_chunked_prefill)

        # Mixed audio prompts stay causal and can use chunked prefill.
        model.get_input_embeddings(
            mx.array([[1, 2, 3, 4]]), mm_token_type_ids=mixed_audio_types
        )
        self.assertFalse(model.no_chunked_prefill)
        self.assertFalse(model.language_model.no_chunked_prefill)

    def test_gemma4_unified_vision_tokens_use_bidirectional_mask(self):
        from mlx_vlm.models import gemma4_unified
        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        config = gemma4_unified.TextConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            hidden_size_per_layer_input=0,
            sliding_window=8,
            sliding_window_pattern=1,
            layer_types=["full_attention"],
            use_bidirectional_attention="vision",
        )
        model = Gemma4TextModel(config)
        hidden_states = mx.zeros((1, 4, config.hidden_size))
        mm_token_type_ids = mx.array([[0, 1, 1, 0]])

        mask = model._make_masks(hidden_states, [None], mm_token_type_ids)[0]

        self.assertTrue(bool(mask[0, 0, 1, 2].item()))
        self.assertFalse(bool(mask[0, 0, 0, 2].item()))
        self.assertTrue(bool(mask[0, 0, 3, 2].item()))

    def test_gemma4_unified_text_only_mm_ids_keep_causal_mask(self):
        from mlx_vlm.models import gemma4_unified
        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        config = gemma4_unified.TextConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            hidden_size_per_layer_input=0,
            sliding_window=8,
            sliding_window_pattern=1,
            layer_types=["full_attention"],
            use_bidirectional_attention="vision",
        )
        model = Gemma4TextModel(config)
        hidden_states = mx.zeros((1, 4, config.hidden_size))
        mm_token_type_ids = mx.array([[0, 0, 0, 0]])

        mask = model._make_masks(hidden_states, [None], mm_token_type_ids)[0]

        self.assertEqual(mask, "causal")

    def test_gemma4_full_attention_cached_chunk_uses_causal_mask(self):
        from mlx_vlm.models import cache, gemma4_unified
        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        config = gemma4_unified.TextConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            hidden_size_per_layer_input=0,
            sliding_window=8,
            sliding_window_pattern=1,
            layer_types=["full_attention"],
        )
        model = Gemma4TextModel(config)
        hidden_states = mx.zeros((1, 4, config.hidden_size))
        kv_cache = cache.KVCache()
        kv_cache.offset = 8

        mask = model._make_masks(hidden_states, [kv_cache])[0]

        self.assertEqual(mask, "causal")

    def test_gemma4_unified_audio_tokens_keep_vision_mask_causal(self):
        from mlx_vlm.models import gemma4_unified
        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        config = gemma4_unified.TextConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            hidden_size_per_layer_input=0,
            sliding_window=8,
            sliding_window_pattern=1,
            layer_types=["full_attention"],
            use_bidirectional_attention="vision",
        )
        model = Gemma4TextModel(config)
        hidden_states = mx.zeros((1, 6, config.hidden_size))
        mm_token_type_ids = mx.array([[0, 1, 1, 0, 3, 3]])

        mask = model._make_masks(hidden_states, [None], mm_token_type_ids)[0]

        self.assertEqual(mask, "causal")

    def test_glm4v_input_embeddings(self):
        from mlx_vlm.models import glm4v

        model = glm4v.Model(
            glm4v.ModelConfig(
                text_config=glm4v.TextConfig(
                    model_type="glm4v_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                ),
                vision_config=glm4v.VisionConfig(
                    model_type="glm4v_vision",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    patch_size=14,
                    out_hidden_size=16,
                ),
                model_type="glm4v",
            )
        )
        self._check_returns_input_embeddings_features(model, "glm4v")

    def test_glm4v_moe_input_embeddings(self):
        from mlx_vlm.models import glm4v_moe

        model = glm4v_moe.Model(
            glm4v_moe.ModelConfig(
                text_config=glm4v_moe.TextConfig(
                    model_type="glm4v_text",
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    max_position_embeddings=128,
                    moe_intermediate_size=16,
                    norm_topk_prob=True,
                    num_attention_heads=2,
                    n_group=1,
                    head_dim=8,
                    topk_group=1,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    routed_scaling_factor=1.0,
                    num_experts_per_tok=1,
                    first_k_dense_replace=0,
                    num_hidden_layers=1,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    use_qk_norm=False,
                    attention_bias=False,
                    partial_rotary_factor=0.5,
                    rope_theta=10000.0,
                ),
                vision_config=glm4v_moe.VisionConfig(
                    model_type="glm4v_moe",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    patch_size=14,
                    out_hidden_size=16,
                ),
                model_type="glm4v_moe",
            )
        )
        self._check_returns_input_embeddings_features(model, "glm4v_moe")

    def test_hunyuan_vl_input_embeddings(self):
        from mlx_vlm.models import hunyuan_vl

        model = hunyuan_vl.Model(
            hunyuan_vl.ModelConfig(
                text_config=hunyuan_vl.TextConfig(
                    model_type="hunyuan_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                    attention_head_dim=8,
                ),
                vision_config=hunyuan_vl.VisionConfig(
                    model_type="hunyuan_vl",
                    hidden_size=16,
                    out_hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                ),
                model_type="hunyuan_vl",
            )
        )
        self._check_returns_input_embeddings_features(model, "hunyuan_vl")

    def test_jina_vlm_input_embeddings(self):
        from mlx_vlm.models import jina_vlm

        model = jina_vlm.Model(
            jina_vlm.ModelConfig(
                text_config=jina_vlm.TextConfig(
                    model_type="jina_vlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    intermediate_size=32,
                ),
                vision_config=jina_vlm.VisionConfig(
                    model_type="jina_vlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    head_dim=8,
                    intermediate_size=32,
                    output_size=16,
                    connector_hidden_size=32,
                ),
                model_type="jina_vlm",
            )
        )
        self._check_returns_input_embeddings_features(model, "jina_vlm")

    def test_lfm2_vl_input_embeddings(self):
        from mlx_vlm.models import lfm2_vl

        model = lfm2_vl.Model(
            lfm2_vl.ModelConfig(
                text_config=lfm2_vl.TextConfig(
                    model_type="lfm2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    layer_types=["full_attention"],
                    block_dim=16,
                    block_ff_dim=32,
                    conv_dim=16,
                    conv_dim_out=16,
                ),
                vision_config=lfm2_vl.VisionConfig(
                    model_type="lfm2_vl",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="lfm2-vl",
                projector_hidden_size=16,
            )
        )
        self._check_returns_input_embeddings_features(model, "lfm2_vl")

    def test_lfm2_vl_disabled_projector_layernorm_weights_load(self):
        from mlx_vlm.models import lfm2_vl

        model = lfm2_vl.Model(
            lfm2_vl.ModelConfig(
                text_config=lfm2_vl.TextConfig(
                    model_type="lfm2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    layer_types=["full_attention"],
                    block_dim=16,
                    block_ff_dim=32,
                    conv_dim=16,
                    conv_dim_out=16,
                ),
                vision_config=lfm2_vl.VisionConfig(
                    model_type="lfm2_vl",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="lfm2-vl",
                projector_hidden_size=16,
                projector_use_layernorm=False,
            )
        )

        self.assertFalse(model.multi_modal_projector.projector_use_layernorm)

        # Real LFM2.5-VL checkpoints ship no projector layer_norm weights.
        model.multi_modal_projector.load_weights(
            [
                ("linear_1.weight", mx.ones((16, 64))),
                ("linear_1.bias", mx.zeros((16,))),
                ("linear_2.weight", mx.ones((16, 16))),
                ("linear_2.bias", mx.zeros((16,))),
            ]
        )

        # Conversions from when the projector always created the layer_norm
        # can carry stale weights; sanitize must drop them when disabled.
        sanitized = model.sanitize(
            {
                "model.multi_modal_projector.layer_norm.weight": mx.ones((64,)),
                "model.multi_modal_projector.layer_norm.bias": mx.zeros((64,)),
                "model.multi_modal_projector.linear_1.weight": mx.ones((16, 64)),
            }
        )
        self.assertNotIn("multi_modal_projector.layer_norm.weight", sanitized)
        self.assertNotIn("multi_modal_projector.layer_norm.bias", sanitized)
        self.assertIn("multi_modal_projector.linear_1.weight", sanitized)

    def test_molmo2_input_embeddings(self):
        from mlx_vlm.models import molmo2

        model = molmo2.Model(
            molmo2.ModelConfig(
                text_config=molmo2.TextConfig(
                    model_type="molmo2",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    additional_vocab_size=4,
                ),
                vision_config=molmo2.VisionConfig(
                    vit_config=molmo2.VitConfig(
                        model_type="molmo2",
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                    ),
                    adapter_config=molmo2.AdapterConfig(
                        model_type="molmo2",
                        hidden_size=16,
                        intermediate_size=32,
                        text_hidden_size=16,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                        vit_layers=[-1],
                    ),
                ),
                model_type="molmo2",
            )
        )
        self._check_returns_input_embeddings_features(model, "molmo2")

    def test_paddleocr_vl_input_embeddings(self):
        from mlx_vlm.models import paddleocr_vl

        model = paddleocr_vl.Model(
            paddleocr_vl.ModelConfig(
                text_config=paddleocr_vl.TextConfig(
                    model_type="paddleocr_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                ),
                vision_config=paddleocr_vl.VisionConfig(
                    model_type="paddleocr_vl",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                ),
                model_type="paddleocr_vl",
            )
        )
        self._check_returns_input_embeddings_features(model, "paddleocr_vl")

    def test_paddleocr_vl_text_only_clears_mrope_state(self):
        from mlx_vlm.models import paddleocr_vl

        model = paddleocr_vl.Model(
            paddleocr_vl.ModelConfig(
                text_config=paddleocr_vl.TextConfig(
                    model_type="paddleocr_vl",
                    hidden_size=4,
                    num_hidden_layers=1,
                    intermediate_size=8,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    vocab_size=10,
                    head_dim=2,
                ),
                vision_config=paddleocr_vl.VisionConfig(
                    model_type="paddleocr_vl",
                    hidden_size=4,
                    intermediate_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                ),
                model_type="paddleocr_vl",
                image_token_id=9,
                vision_start_token_id=8,
            )
        )

        model.language_model._position_ids = mx.array([[[0, 1]]], dtype=mx.int32)
        model.language_model._rope_deltas = mx.array([[4]], dtype=mx.int32)

        model.get_input_embeddings(mx.array([[1, 2]], dtype=mx.int32))

        self.assertIsNone(model.language_model._position_ids)
        self.assertIsNone(model.language_model._rope_deltas)

    def test_paddleocr_vl_prefill_recomputes_stale_position_ids(self):
        from mlx_vlm.models import paddleocr_vl

        text_config = paddleocr_vl.TextConfig(
            model_type="paddleocr_vl",
            hidden_size=4,
            num_hidden_layers=1,
            intermediate_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=10,
            head_dim=2,
        )
        config = paddleocr_vl.ModelConfig(
            text_config=text_config,
            vision_config=paddleocr_vl.VisionConfig(
                model_type="paddleocr_vl",
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
            ),
            model_type="paddleocr_vl",
            image_token_id=9,
            vision_start_token_id=8,
        )
        lm = paddleocr_vl.LanguageModel(text_config, config)

        captured = {}

        class _CapturingInnerModel:
            class _Embed:
                def __call__(self, inputs):
                    return mx.zeros((inputs.shape[0], inputs.shape[1], 4))

            embed_tokens = _Embed()

            def __call__(self, inputs, inputs_embeds=None, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], 4))

        lm.model = _CapturingInnerModel()
        lm.lm_head = lambda x: x
        lm._position_ids = mx.array([[[0, 1, 2]]], dtype=mx.int32)
        lm._rope_deltas = mx.array([[3]], dtype=mx.int32)

        class _Cache:
            _idx = 0
            offset = mx.array(0)

        lm(mx.array([[1, 2], [3, 4]], dtype=mx.int32), cache=[_Cache()])

        self.assertEqual(captured["position_ids"].shape, (3, 2, 2))
        self.assertEqual(lm._rope_deltas.tolist(), [[0], [0]])

    def test_phi3_v_input_embeddings(self):
        from mlx_vlm.models import phi3_v
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = phi3_v.Model(
            phi3_v.ModelConfig(
                text_config=phi3_v.TextConfig(),
                vision_config=phi3_v.VisionConfig(
                    model_type="phi3_v",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="phi3_v",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=32,
            )
        )
        # phi3_v uses 'inputs' as positional arg instead of 'input_ids'
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

        num_image_tokens = int((1 * 1 + 1) * 144 + 1 + (1 + 1) * 12)
        mm_input_ids = mx.array([[1, 2] + [-1] * num_image_tokens + [3]])
        image_sizes = mx.array([[336, 336]], dtype=mx.int64)

        model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))

        pixel_values = mx.random.normal(shape=(1, 2, 3, 336, 336))
        result_bf16 = model.get_input_embeddings(
            mm_input_ids, pixel_values=pixel_values, image_sizes=image_sizes
        )
        self.assertIsInstance(result_bf16, InputEmbeddingsFeatures)
        self.assertIsNotNone(result_bf16.inputs_embeds)
        self.assertEqual(result_bf16.inputs_embeds.dtype, mx.bfloat16)

    def test_phi4mm_input_embeddings(self):
        from mlx_vlm.models import phi4mm

        model = phi4mm.Model(
            phi4mm.ModelConfig(
                text_config=phi4mm.TextConfig(
                    model_type="phi4mm",
                    max_position_embeddings=2048,
                ),
                vision_config=phi4mm.VisionConfig(
                    model_type="siglip2_vision_model",
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    patch_size=14,
                    image_size=28,
                    num_channels=3,
                    layer_norm_eps=1e-6,
                ),
                model_type="phi4mm",
                vocab_size=256,
                hidden_size=64,
                num_hidden_layers=2,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                rms_norm_eps=1e-5,
                mm_hidden_size=32,
                image_token_index=-200,
                audio_token_index=200,
                audio_processor={
                    "config": {
                        "attention_dim": 32,
                        "attention_heads": 4,
                        "num_blocks": 2,
                        "linear_units": 64,
                        "input_size": 80,
                        "time_reduction": 8,
                        "kernel_size": 3,
                        "conv_channels": 32,
                        "ext_pw_out_channel": 32,
                        "depthwise_seperable_out_channel": 32,
                    }
                },
            )
        )
        self._check_returns_input_embeddings_features(model, "phi4mm")

    def test_qwen3_vl_moe_input_embeddings(self):
        from mlx_vlm.models import qwen3_vl_moe

        model = qwen3_vl_moe.Model(
            qwen3_vl_moe.ModelConfig(
                text_config=qwen3_vl_moe.TextConfig(
                    model_type="qwen3_vl_moe",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_experts=2,
                    num_experts_per_tok=1,
                    decoder_sparse_step=1,
                    mlp_only_layers=[],
                    moe_intermediate_size=16,
                    rms_norm_eps=1e-5,
                    vocab_size=32,
                    num_key_value_heads=2,
                    head_dim=8,
                    rope_theta=10000.0,
                    max_position_embeddings=128,
                ),
                vision_config=qwen3_vl_moe.VisionConfig(
                    model_type="qwen3_vl_moe",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    out_hidden_size=16,
                    num_heads=2,
                ),
                model_type="qwen3_vl_moe",
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_vl_moe")
        self._assert_qwen_request_owned_mrope_kwargs(model)

    def test_qwen3_5_input_embeddings_owns_mrope_kwargs(self):
        from mlx_vlm.models import qwen3_5

        model = qwen3_5.Model(
            qwen3_5.ModelConfig(
                text_config=qwen3_5.TextConfig(
                    model_type="qwen3_5",
                    hidden_size=16,
                    intermediate_size=32,
                    linear_num_value_heads=2,
                    linear_num_key_heads=2,
                    linear_key_head_dim=8,
                    linear_value_head_dim=8,
                    linear_conv_kernel_dim=3,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    rms_norm_eps=1e-5,
                    vocab_size=32,
                    num_key_value_heads=2,
                    max_position_embeddings=128,
                    head_dim=8,
                ),
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
        )
        self._assert_qwen_request_owned_mrope_kwargs(model)

    def test_qwen3_omni_moe_input_embeddings(self):
        from mlx_vlm.models import qwen3_omni_moe

        text_config = qwen3_omni_moe.TextConfig(
            num_hidden_layers=1,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            moe_intermediate_size=16,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            max_position_embeddings=128,
        )
        model = qwen3_omni_moe.Model(
            qwen3_omni_moe.ModelConfig(
                thinker_config=qwen3_omni_moe.ThinkerConfig(
                    text_config=text_config,
                    vision_config=qwen3_omni_moe.VisionConfig(
                        model_type="qwen3_omni_moe_vision_encoder",
                        depth=1,
                        hidden_size=16,
                        intermediate_size=32,
                        out_hidden_size=16,
                        num_heads=2,
                    ),
                    audio_config=qwen3_omni_moe.AudioConfig(
                        d_model=16,
                        encoder_layers=1,
                        encoder_attention_heads=2,
                        encoder_ffn_dim=32,
                        num_hidden_layers=1,
                        output_dim=16,
                    ),
                ),
                talker_config=qwen3_omni_moe.TalkerConfig(
                    text_config=text_config,
                    code_predictor_config=qwen3_omni_moe.CodePredictorConfig(
                        num_hidden_layers=1,
                        hidden_size=16,
                        intermediate_size=32,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                    ),
                    thinker_hidden_size=16,
                ),
                code2wav_config=qwen3_omni_moe.Code2WavConfig(),
                model_type="qwen3_omni_moe",
                enable_audio_output=False,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_omni_moe")

    def test_internvl_chat_input_embeddings(self):
        from mlx_vlm.models import internvl_chat

        model = internvl_chat.Model(
            internvl_chat.ModelConfig(
                text_config=internvl_chat.TextConfig(
                    model_type="qwen2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    max_window_layers=1,
                    hidden_act="silu",
                ),
                vision_config=internvl_chat.VisionConfig(
                    model_type="intern_vit_6b",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="internvl_chat",
                image_token_index=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "internvl_chat")

    def test_glm_ocr_input_embeddings(self):
        from mlx_vlm.models import glm_ocr

        model = glm_ocr.Model(
            glm_ocr.ModelConfig(
                text_config=glm_ocr.TextConfig(
                    model_type="glm_ocr_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    head_dim=8,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=1000,
                    rope_parameters={
                        "rope_type": "default",
                        "mrope_section": [2, 3, 3],
                        "partial_rotary_factor": 1.0,
                        "rope_theta": 10000,
                    },
                ),
                vision_config=glm_ocr.VisionConfig(
                    model_type="glm_ocr_vision",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    out_hidden_size=16,
                    patch_size=14,
                    in_channels=3,
                    rms_norm_eps=1e-5,
                ),
                model_type="glm_ocr",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "glm_ocr")

    def test_phi4_siglip_input_embeddings(self):
        from mlx_vlm.models import phi4_siglip
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = phi4_siglip.Model(
            phi4_siglip.ModelConfig(
                text_config=phi4_siglip.TextConfig(
                    model_type="phi4-siglip",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=phi4_siglip.VisionConfig(
                    model_type="siglip2_vision_model",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    patch_size=14,
                    num_patches=256,
                ),
                model_type="phi4-siglip",
                mm_hidden_size=16,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

    def test_moondream3_input_embeddings(self):
        from mlx_vlm.models import moondream3
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = moondream3.Model(
            moondream3.ModelConfig(
                text_config=moondream3.TextConfig(
                    model_type="moondream3",
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=16,
                    vocab_size=256,
                    rope_dim=8,
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=32,
                    moe_start_layer=2,
                ),
                vision_config=moondream3.VisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    patch_size=14,
                    crop_size=28,
                    proj_inner_dim=64,
                    proj_out_dim=64,
                ),
                model_type="moondream3",
            )
        )
        # Text-only
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)
        self.assertIsNone(result.attention_mask_4d)

        # With vision: should return prefix attention mask
        num_patches = (28 // 14) ** 2  # 4 patches
        input_ids_vis = mx.zeros((1, 1 + num_patches + 2), dtype=mx.int32)
        pixel_values = mx.random.uniform(shape=(1, 28, 28, 3))
        result_vis = model.get_input_embeddings(
            input_ids_vis,
            pixel_values=pixel_values,
            num_crops=[1],
            crop_layouts=[(1, 1)],
        )
        self.assertIsInstance(result_vis, InputEmbeddingsFeatures)
        self.assertIsNotNone(result_vis.inputs_embeds)
        self.assertIsNotNone(result_vis.attention_mask_4d)

    def test_granite_vision_input_embeddings(self):
        from mlx_vlm.models import granite_vision
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = granite_vision.Model(
            granite_vision.ModelConfig(
                text_config=granite_vision.TextConfig(
                    model_type="granite",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=granite_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="granite_vision",
                vision_feature_layer=-1,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

    def test_granite4_vision_input_embeddings(self):
        from mlx_vlm.models import granite4_vision
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = granite4_vision.Model(
            granite4_vision.ModelConfig(
                text_config=granite4_vision.TextConfig(
                    model_type="granitemoehybrid",
                    hidden_size=64,
                    num_hidden_layers=1,
                    intermediate_size=128,
                    shared_intermediate_size=128,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=granite4_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=16,
                ),
                model_type="granite4_vision",
                deepstack_layer_map=[[-1, 0]],
                use_spatial_sampling=False,
                downsample_rate="2/2",
                use_image_newline_parameter=False,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)


class TestChunkedPrefillRoPE(unittest.TestCase):
    """Test chunked prefill RoPE position ID generation for vision-language models."""

    def test_ernie_chunked_prefill_rope(self):
        """Test ERNIE 4.5 MoE VL chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import ernie4_5_moe_vl

        text_config = ernie4_5_moe_vl.TextConfig(
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
        )
        vision_config = ernie4_5_moe_vl.VisionConfig(
            embed_dim=256,
            hidden_size=256,
            num_heads=8,
            patch_size=14,
            spatial_merge_size=2,
        )
        model_config = ernie4_5_moe_vl.ModelConfig(
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            vision_config=vision_config,
            text_config=text_config,
            im_patch_id=100,
            image_token_id=100,
            video_token_id=101,
            image_start_token_id=99,
            vision_start_token_id=99,
        )
        lm = ernie4_5_moe_vl.LanguageModel(text_config, model_config)

        input_ids = mx.array([[1, 2, 3, 99, 100, 100, 100, 100, 5, 6, 7]])
        image_grid_thw = mx.array([[1, 4, 4]])
        position_ids, _ = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[1], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        full_input = [1, 2, 3, 99, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10]
        chunked_input = full_input[:8]
        chunked_input_ids = mx.array([chunked_input])
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[1], len(chunked_input))

        # Position IDs have correct 3D shape for MRoPE
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 1)  # batch size
        self.assertEqual(position_ids.shape[2], 3)  # 3D positions (T, H, W)

    def test_glm4v_chunked_prefill_rope(self):
        """Test GLM4V chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v_text",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=256,
        )
        vision_config = glm4v.VisionConfig(
            model_type="glm4v_vision",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        model_config = glm4v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v",
            vocab_size=64,
            image_token_id=61,
            image_token_index=61,
            video_token_id=62,
            video_token_index=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
            pad_token_id=0,
        )
        lm = glm4v.LanguageModel(text_config, model_config)

        input_ids = mx.array([[10, 60, 61, 11, 12, 13, 14, 15]], dtype=mx.int32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        position_ids, rope_deltas = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[2], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        chunked_input_ids = input_ids[:, :4]
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[2], chunked_input_ids.shape[1])

        # Position IDs have expected MRoPE shape
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 3)  # MRoPE dimensions
        self.assertEqual(position_ids.shape[1], 1)  # batch size

        # Regression guard: full-length mask with chunked inputs should not fail
        full_mask = mx.ones((1, input_ids.shape[1]), dtype=mx.int32)
        lm._position_ids = position_ids
        lm._rope_deltas = rope_deltas
        outputs = lm(chunked_input_ids, mask=full_mask, image_grid_thw=image_grid_thw)
        self.assertEqual(
            outputs.logits.shape,
            (1, chunked_input_ids.shape[1], text_config.vocab_size),
        )

    def test_glm4v_get_rope_index_per_row_deltas(self):
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v_text",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=256,
        )
        vision_config = glm4v.VisionConfig(
            model_type="glm4v_vision",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        lm = glm4v.LanguageModel(
            text_config,
            glm4v.ModelConfig(
                text_config=text_config,
                vision_config=vision_config,
                model_type="glm4v",
                vocab_size=64,
                image_token_id=61,
                image_token_index=61,
                video_token_id=62,
                video_token_index=62,
                vision_start_token_id=60,
                vision_end_token_id=59,
                pad_token_id=0,
            ),
        )

        input_ids = mx.array(
            [
                [10, 60, 61, 61, 61, 61, 11, 12],
                [10, 11, 12, 13, 14, 15, 16, 17],
            ],
            dtype=mx.int32,
        )
        _, rope_deltas = lm.get_rope_index(
            input_ids, mx.array([[1, 4, 4]], dtype=mx.int32)
        )
        self.assertEqual(rope_deltas.shape, (2, 1))
        self.assertEqual(rope_deltas[1, 0].item(), 0)
        self.assertNotEqual(rope_deltas[0, 0].item(), rope_deltas[1, 0].item())

        input_ids = mx.array(
            [[0, 0, 10, 11, 12, 13], [10, 11, 12, 13, 14, 15]], dtype=mx.int32
        )
        attention_mask = mx.array(
            [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=mx.int32
        )
        position_ids, rope_deltas = lm.get_rope_index(
            input_ids, image_grid_thw=None, attention_mask=attention_mask
        )
        self.assertEqual(rope_deltas.shape, (2, 1))
        self.assertEqual(position_ids.shape, (3, 2, 6))
        self.assertEqual(rope_deltas[0, 0].item(), -2)
        self.assertEqual(rope_deltas[1, 0].item(), 0)

    def test_glm4v_moe_chunked_prefill_rope(self):
        """Test GLM4V-MoE chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import glm4v_moe

        text_config = glm4v_moe.TextConfig(
            model_type="glm4v_moe_text",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            max_position_embeddings=256,
            moe_intermediate_size=16,
            norm_topk_prob=True,
            num_attention_heads=2,
            n_group=1,
            head_dim=8,
            topk_group=1,
            n_shared_experts=1,
            n_routed_experts=2,
            routed_scaling_factor=1.0,
            num_experts_per_tok=1,
            first_k_dense_replace=0,
            num_hidden_layers=1,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            use_qk_norm=False,
            attention_bias=False,
            partial_rotary_factor=0.5,
            rope_theta=10000.0,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            tie_word_embeddings=False,
        )
        vision_config = glm4v_moe.VisionConfig(
            model_type="glm4v_moe",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        model_config = glm4v_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v_moe",
            vocab_size=64,
            image_token_id=61,
            image_token_index=61,
            video_token_id=62,
            video_token_index=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
            pad_token_id=0,
        )
        lm = glm4v_moe.LanguageModel(text_config, model_config)

        input_ids = mx.array([[10, 60, 61, 11, 12, 13, 14, 15]], dtype=mx.int32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        position_ids, rope_deltas = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[2], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        chunked_input_ids = input_ids[:, :4]
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[2], chunked_input_ids.shape[1])

        # Position IDs have expected MRoPE shape
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 3)  # MRoPE dimensions
        self.assertEqual(position_ids.shape[1], 1)  # batch size

        # Regression guard: full-length mask with chunked inputs should not fail
        full_mask = mx.ones((1, input_ids.shape[1]), dtype=mx.int32)
        lm._position_ids = position_ids
        lm._rope_deltas = rope_deltas
        outputs = lm(chunked_input_ids, mask=full_mask, image_grid_thw=image_grid_thw)
        self.assertEqual(
            outputs.logits.shape,
            (1, chunked_input_ids.shape[1], text_config.vocab_size),
        )


class TestMultiImageMRoPE(unittest.TestCase):
    """Regression tests for multi-image prompts in ``get_rope_index``.

    The original M-RoPE port summed all vision-start indices into a single
    scalar (``mx.sum(mx.where(...))``), so prompts with two or more images
    mis-counted the image tokens and assigned flat text positions to every
    image after the first.  qwen3_vl/qwen3_5 received the corrected token
    scan in the MRoPE refactor; these tests pin the same behavior across
    every family that shares the implementation.
    """

    _FAMILIES = [
        "glm4v",
        "glm4v_moe",
        "paddleocr_vl",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_omni_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
    ]

    # Prompt layout: text(3 incl. vision_start) | 4 image tokens | text(2 incl.
    # vision_start) | 4 image tokens | text(1) — two (1, 4, 4) grids at
    # spatial_merge_size 2, i.e. four merged tokens per image.
    _INPUT_IDS = [1, 2, 99, 100, 100, 100, 100, 5, 99, 100, 100, 100, 100, 7]
    _IMAGE_GRID_THW = [[1, 4, 4], [1, 4, 4]]

    # Expected positions follow the HF reference: a text block advances all
    # three dims together; each image block expands t/h/w over the merged
    # (1, 2, 2) grid starting one past the previous maximum.
    _EXPECTED_T = [0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 7, 7, 7, 9]
    _EXPECTED_H = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9]
    _EXPECTED_W = [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 9]
    _EXPECTED_DELTA = 9 + 1 - len(_INPUT_IDS)

    @staticmethod
    def _rope_index(family, input_ids, image_grid_thw, attention_mask=None):
        module = importlib.import_module(f"mlx_vlm.models.{family}.language")
        stub = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
                image_token_id=100,
                video_token_id=101,
                vision_start_token_id=99,
            )
        )
        return module.LanguageModel.get_rope_index(
            stub,
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    def test_two_image_prompt_positions(self):
        input_ids = mx.array([self._INPUT_IDS])
        image_grid_thw = mx.array(self._IMAGE_GRID_THW)

        for family in self._FAMILIES:
            with self.subTest(model=family):
                position_ids, deltas = self._rope_index(
                    family, input_ids, image_grid_thw
                )
                self.assertEqual(position_ids.shape, (3, 1, len(self._INPUT_IDS)))
                self.assertEqual(position_ids[0, 0].tolist(), self._EXPECTED_T)
                self.assertEqual(position_ids[1, 0].tolist(), self._EXPECTED_H)
                self.assertEqual(position_ids[2, 0].tolist(), self._EXPECTED_W)
                self.assertEqual(
                    int(deltas.reshape(-1)[0].item()), self._EXPECTED_DELTA
                )

    def test_two_image_prompt_with_left_padding(self):
        pad = 2
        input_ids = mx.array([[0] * pad + self._INPUT_IDS])
        attention_mask = mx.array([[0] * pad + [1] * len(self._INPUT_IDS)])
        image_grid_thw = mx.array(self._IMAGE_GRID_THW)

        for family in self._FAMILIES:
            with self.subTest(model=family):
                position_ids, deltas = self._rope_index(
                    family, input_ids, image_grid_thw, attention_mask
                )
                valid = position_ids[:, 0, pad:]
                self.assertEqual(valid[0].tolist(), self._EXPECTED_T)
                self.assertEqual(valid[1].tolist(), self._EXPECTED_H)
                self.assertEqual(valid[2].tolist(), self._EXPECTED_W)
                self.assertEqual(
                    int(deltas.reshape(-1)[0].item()), self._EXPECTED_DELTA
                )


class TestMRoPETrainingVJP(unittest.TestCase):
    """Regression tests for issue #1474: LoRA/fine-tuning any M-RoPE VL model
    crashed on the first backward with ``[Primitive::vjp] Not implemented for
    CustomKernel`` because the M-RoPE apply used a raw ``metal_kernel`` (which
    has no VJP). The kernel forward is now wrapped in ``mx.custom_function`` with
    a VJP routed through the pure-MLX equivalent, so ``value_and_grad`` works
    while inference stays byte-identical.
    """

    _DIM = 64
    _HEADS = 2
    _SEQ = 8
    _BSZ = 1

    def _qk(self):
        mx.random.seed(0)
        q = mx.random.normal((self._BSZ, self._HEADS, self._SEQ, self._DIM))
        k = mx.random.normal((self._BSZ, self._HEADS, self._SEQ, self._DIM))
        return q, k

    @staticmethod
    def _sum_loss(fn):
        def loss(q, k):
            qo, ko = fn(q, k)
            return qo.astype(mx.float32).sum() + ko.astype(mx.float32).sum()

        return loss

    def _grads(self, fn, q, k):
        return mx.value_and_grad(self._sum_loss(fn), argnums=(0, 1))(q, k)

    def test_fused_apply_rotary_vjp(self):
        """Fused MRoPE apply (``MRoPERotaryEmbedding.apply_rotary``) is
        differentiable and matches the known-good pure-MLX gradient."""
        from mlx_vlm.models import rope_utils

        pos_2d = mx.arange(self._SEQ, dtype=mx.int32)[None, :]
        pos_3d = mx.broadcast_to(
            mx.arange(self._SEQ, dtype=mx.int32)[None, None, :],
            (3, self._BSZ, self._SEQ),
        )

        for style in ("interleaved", "chunked"):
            for pos in (pos_2d, pos_3d):
                with self.subTest(style=style, pos_ndim=pos.ndim):
                    q, k = self._qk()

                    rope = rope_utils.MRoPERotaryEmbedding(dim=self._DIM, style=style)
                    _, (dq, dk) = self._grads(
                        lambda q, k: rope.apply_rotary(q, k, pos), q, k
                    )
                    mx.eval(dq, dk)
                    self.assertTrue(bool(mx.all(mx.isfinite(dq))))
                    self.assertTrue(bool(mx.all(mx.isfinite(dk))))

                    saved = rope_utils._HAS_METAL
                    try:
                        rope_utils._HAS_METAL = False
                        ref = rope_utils.MRoPERotaryEmbedding(
                            dim=self._DIM, style=style
                        )
                        _, (dq2, dk2) = self._grads(
                            lambda q, k: ref.apply_rotary(q, k, pos), q, k
                        )
                        mx.eval(dq2, dk2)
                    finally:
                        rope_utils._HAS_METAL = saved

                    self.assertTrue(bool(mx.allclose(dq, dq2, atol=1e-4, rtol=1e-4)))
                    self.assertTrue(bool(mx.allclose(dk, dk2, atol=1e-4, rtol=1e-4)))

    def test_precomputed_rotary_vjp_matches_pure_mlx(self):
        """Precomputed-cos/sin apply (even/odd and sectioned kernel paths) is
        differentiable and matches the known-good pure-MLX gradient."""
        from mlx_vlm.models import rope_utils

        mx.random.seed(1)
        rotary_dim = self._DIM
        ang3 = mx.random.normal((self._BSZ, self._SEQ, rotary_dim))
        cos3, sin3 = mx.cos(ang3), mx.sin(ang3)
        ang4 = mx.random.normal((3, self._BSZ, self._SEQ, rotary_dim))
        cos4, sin4 = mx.cos(ang4), mx.sin(ang4)
        mrope_section = [4, 6, 6]  # sums to rotary_dim // 2 == 32

        cases = [
            (
                "even_odd_half",
                lambda q, k: rope_utils.apply_rotary_pos_emb_even_odd(
                    q, k, cos3, sin3, cos_layout="half"
                ),
            ),
            (
                "even_odd_full",
                lambda q, k: rope_utils.apply_rotary_pos_emb_even_odd(
                    q, k, cos3, sin3, cos_layout="full"
                ),
            ),
            (
                "sectioned_half_split",
                lambda q, k: rope_utils.apply_multimodal_rotary_pos_emb(
                    q,
                    k,
                    cos4,
                    sin4,
                    mrope_section=mrope_section,
                    style="sectioned_half_split",
                ),
            ),
            (
                "sectioned_even_odd",
                lambda q, k: rope_utils.apply_multimodal_rotary_pos_emb(
                    q,
                    k,
                    cos4,
                    sin4,
                    mrope_section=mrope_section,
                    style="sectioned_even_odd",
                ),
            ),
        ]

        for name, fn in cases:
            with self.subTest(case=name):
                q, k = self._qk()
                _, (dq, dk) = self._grads(fn, q, k)
                mx.eval(dq, dk)
                self.assertTrue(bool(mx.all(mx.isfinite(dq))))
                self.assertTrue(bool(mx.all(mx.isfinite(dk))))

                saved = rope_utils._HAS_METAL
                try:
                    rope_utils._HAS_METAL = False
                    rope_utils._compiled_rotary_apply.cache_clear()
                    _, (dq2, dk2) = self._grads(fn, q, k)
                    mx.eval(dq2, dk2)
                finally:
                    rope_utils._HAS_METAL = saved
                    rope_utils._compiled_rotary_apply.cache_clear()

                self.assertTrue(bool(mx.allclose(dq, dq2, atol=1e-4, rtol=1e-4)))
                self.assertTrue(bool(mx.allclose(dk, dk2, atol=1e-4, rtol=1e-4)))


class TestMiniCPMV4_6(unittest.TestCase):
    @staticmethod
    def _tiny_text_config():
        from mlx_vlm.models import minicpmv4_6

        return minicpmv4_6.TextConfig(
            model_type="qwen3_5_text",
            hidden_size=16,
            intermediate_size=32,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            num_hidden_layers=4,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
        )

    def test_minicpmv4_6_language_uses_text_only_rope(self):
        from mlx_vlm.models import minicpmv4_6

        model_config = SimpleNamespace(vision_config=SimpleNamespace())
        lm = minicpmv4_6.LanguageModel(self._tiny_text_config(), model_config)

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        position_ids, rope_deltas = lm.get_rope_index(input_ids)

        self.assertEqual(position_ids.shape, (3, 1, input_ids.shape[1]))
        self.assertEqual(rope_deltas.tolist(), [[0]])

    def test_minicpmv4_6_language_rejects_qwen_vl_grid_rope(self):
        from mlx_vlm.models import minicpmv4_6

        model_config = SimpleNamespace(vision_config=SimpleNamespace())
        lm = minicpmv4_6.LanguageModel(self._tiny_text_config(), model_config)

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        with self.assertRaisesRegex(ValueError, "does not support Qwen3.5-VL"):
            lm.get_rope_index(input_ids, image_grid_thw=image_grid_thw)


class TestMiniCPMO(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        from mlx_vlm.models import minicpmo

        text_config = minicpmo.TextConfig(
            model_type="minicpmo",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=4,
            head_dim=16,
            rope_theta=10000.0,
            max_position_embeddings=2048,
        )
        vision_config = minicpmo.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=28,
            patch_size=14,
        )
        return minicpmo.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            query_num=4,
        )

    def test_minicpmo_config_from_root_fields(self):
        from mlx_vlm.models import minicpmo

        cfg = {
            "model_type": "minicpmo",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960,
            "query_num": 64,
            "vision_config": {
                "model_type": "siglip",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_channels": 3,
                "image_size": 448,
                "patch_size": 14,
            },
        }
        model_config = minicpmo.ModelConfig.from_dict(cfg)
        self.assertEqual(model_config.text_config.hidden_size, 4096)
        self.assertEqual(model_config.vision_config.model_type, "siglip_vision_model")
        self.assertEqual(model_config.query_num, 64)

    def test_minicpmo_sanitize_key_mapping_and_qkv_split(self):
        from mlx_vlm.models import minicpmo

        model = minicpmo.Model(self._tiny_config())
        weights = {
            "llm.model.embed_tokens.weight": mx.zeros((10, 10)),
            "llm.lm_head.weight": mx.zeros((10, 10)),
            "vpm.embeddings.patch_embedding.weight": mx.zeros((8, 3, 14, 14)),
            "resampler.attn.in_proj_weight": mx.zeros((192, 64)),
            "resampler.attn.in_proj_bias": mx.zeros((192,)),
            "apm.conv1.weight": mx.zeros((1, 1)),
        }

        sanitized = model.sanitize(weights)
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)
        self.assertIn("vision_tower.embeddings.patch_embedding.weight", sanitized)
        self.assertNotIn("apm.conv1.weight", sanitized)

        self.assertIn("resampler.attn.q_proj.weight", sanitized)
        self.assertIn("resampler.attn.k_proj.weight", sanitized)
        self.assertIn("resampler.attn.v_proj.weight", sanitized)
        self.assertIn("resampler.attn.q_proj.bias", sanitized)
        self.assertIn("resampler.attn.k_proj.bias", sanitized)
        self.assertIn("resampler.attn.v_proj.bias", sanitized)

    def test_minicpmo_sanitize_audio_conv_layout(self):
        from mlx_vlm.models import minicpmo

        model = minicpmo.Model(self._tiny_config())
        weights = {
            "apm.conv1.weight": mx.zeros((8, 80, 3)),
            "apm.conv2.weight": mx.zeros((8, 8, 3)),
        }

        sanitized = model.sanitize(weights)
        self.assertEqual(sanitized["audio_tower.conv1.weight"].shape, (8, 3, 80))
        self.assertEqual(sanitized["audio_tower.conv2.weight"].shape, (8, 3, 8))

    def test_minicpmo_vision_embedding_uses_floating_pixel_dtype(self):
        from mlx_vlm.models import minicpmo

        model = minicpmo.Model(self._tiny_config())
        model.language_model.model.embed_tokens.weight = mx.zeros(
            model.language_model.model.embed_tokens.weight.shape,
            dtype=mx.uint32,
        )
        pixel_values = [[mx.ones((3, 28, 28), dtype=mx.uint32)]]
        tgt_sizes = [mx.array([[2, 2]], dtype=mx.int32)]

        vision_hidden_states = model.get_vision_embedding(pixel_values, tgt_sizes)

        self.assertEqual(len(vision_hidden_states), 1)
        self.assertIsInstance(vision_hidden_states[0], mx.array)
        self.assertEqual(vision_hidden_states[0].shape, (1, 4, 64))


class TestPhi4MM(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        from mlx_vlm.models.phi4mm.config import ModelConfig, TextConfig, VisionConfig

        text_config = TextConfig(
            model_type="phi4mm",
            max_position_embeddings=2048,
        )
        vision_config = VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            patch_size=14,
            image_size=28,
            num_channels=3,
            layer_norm_eps=1e-6,
        )
        return ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="phi4mm",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            mm_hidden_size=32,
            image_token_index=-200,
            audio_token_index=200,
            vision_lora={"r": 4, "lora_alpha": 8},
            speech_lora={"r": 4, "lora_alpha": 8},
            audio_processor={
                "config": {
                    "attention_dim": 32,
                    "attention_heads": 4,
                    "num_blocks": 2,
                    "linear_units": 64,
                    "input_size": 80,
                    "time_reduction": 8,
                    "kernel_size": 3,
                    "conv_channels": 32,
                    "ext_pw_out_channel": 32,
                    "depthwise_seperable_out_channel": 32,
                }
            },
        )

    def test_phi4mm_sanitize_lora_keys(self):
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        hidden = config.hidden_size
        qkv_size = config.num_attention_heads * (
            hidden // config.num_attention_heads
        ) + 2 * config.num_key_value_heads * (hidden // config.num_attention_heads)
        lora_r = 4

        weights = {
            # Base layer weight
            "model.layers.0.self_attn.qkv_proj.base_layer.weight": mx.zeros(
                (qkv_size, hidden)
            ),
            # Vision LoRA
            "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight": mx.zeros(
                (lora_r, hidden)
            ),
            "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight": mx.zeros(
                (qkv_size, lora_r)
            ),
            # Speech LoRA
            "model.layers.0.self_attn.qkv_proj.lora_A.speech.weight": mx.zeros(
                (lora_r, hidden)
            ),
            "model.layers.0.self_attn.qkv_proj.lora_B.speech.weight": mx.zeros(
                (qkv_size, lora_r)
            ),
            # Embed tokens
            "model.embed_tokens.weight": mx.zeros((config.vocab_size, hidden)),
        }

        sanitized = model.sanitize(weights)

        # Base layer should be merged with vision LoRA by default
        self.assertIn(
            "language_model.model.layers.0.self_attn.qkv_proj.weight", sanitized
        )
        # LoRA keys should not appear in sanitized output
        for k in sanitized:
            self.assertNotIn("lora_A", k)
            self.assertNotIn("lora_B", k)
            self.assertNotIn("base_layer", k)

        # Speech LoRA should be stored for runtime switching
        self.assertTrue(len(model._speech_lora_a) > 0)
        self.assertTrue(len(model._speech_lora_b) > 0)
        self.assertTrue(len(model._base_weights) > 0)

    def test_phi4mm_quant_predicate_skips_multimodal(self):
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        predicate = model.quant_predicate

        # Language model layers should be quantized
        self.assertTrue(
            predicate(
                "language_model.model.layers.0.self_attn.qkv_proj", nn.Linear(4, 4)
            )
        )

        # Multimodal modules should NOT be quantized
        self.assertFalse(
            predicate("audio_encoder.encoders.0.attn.linear_q", nn.Linear(4, 4))
        )
        self.assertFalse(predicate("audio_projection.speech.proj_0", nn.Linear(4, 4)))
        self.assertFalse(predicate("mm_projector.0", nn.Linear(4, 4)))
        self.assertFalse(
            predicate("vision_tower.vision_tower.encoder.layers.0", nn.Linear(4, 4))
        )

    def test_phi4mm_quant_predicate_clears_lora(self):
        """Accessing quant_predicate should merge LoRAs and clear stored copies."""
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        hidden = config.hidden_size
        head_dim = hidden // config.num_attention_heads
        qkv_size = (
            config.num_attention_heads * head_dim
            + 2 * config.num_key_value_heads * head_dim
        )
        key = "language_model.model.layers.0.self_attn.qkv_proj.weight"
        lora_r = 4

        # Set up LoRA weights with correct shapes
        model._base_weights = {key: mx.ones((qkv_size, hidden))}
        model._speech_lora_a = {key: mx.zeros((lora_r, hidden))}
        model._speech_lora_b = {key: mx.zeros((qkv_size, lora_r))}
        model._speech_lora_scale = 1.0
        model._vision_lora_a = {key: mx.zeros((lora_r, hidden))}
        model._vision_lora_b = {key: mx.zeros((qkv_size, lora_r))}
        model._vision_lora_scale = 1.0
        model._active_lora = "vision"

        # Accessing the property triggers merge and clears LoRA dicts
        _ = model.quant_predicate

        self.assertEqual(len(model._base_weights), 0)
        self.assertEqual(len(model._speech_lora_a), 0)
        self.assertEqual(len(model._speech_lora_b), 0)
        self.assertEqual(len(model._vision_lora_a), 0)
        self.assertEqual(len(model._vision_lora_b), 0)

    def test_phi4mm_set_modality_skips_when_no_lora(self):
        """set_modality should no-op when _base_weights is empty (quantized model)."""
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)
        model._base_weights = {}

        # Should not raise even with modality flags set
        model.set_modality(has_image=True, has_audio=True)


class TestSam3(unittest.TestCase):
    # ─── SAM3 Tests ────────────────────────────────────────────

    def test_sam3_config(self):
        """Config parses the nested detector/tracker structure."""
        from mlx_vlm.models import sam3

        config = sam3.ModelConfig()
        self.assertEqual(config.model_type, "sam3_video")
        self.assertEqual(
            config.detector_config.vision_config.backbone_config.hidden_size, 1024
        )
        self.assertEqual(config.detector_config.text_config.hidden_size, 1024)
        self.assertEqual(config.detector_config.detr_encoder_config.num_layers, 6)
        self.assertEqual(config.detector_config.detr_decoder_config.num_queries, 200)
        self.assertEqual(config.tracker_config.memory_attention_num_layers, 4)

    def test_sam3_vision_encoder(self):
        """ViT backbone + FPN produce correct shapes."""
        from mlx_vlm.models.sam3.config import VisionEncoderConfig, ViTConfig
        from mlx_vlm.models.sam3.vision import VisionEncoder

        vit_cfg = ViTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            image_size=112,
            patch_size=14,
            window_size=4,
            global_attn_indexes=[1],
            pretrain_image_size=56,
        )
        vision_cfg = VisionEncoderConfig(
            backbone_config=vit_cfg,
            fpn_hidden_size=32,
            scale_factors=[2.0, 1.0],
        )
        encoder = VisionEncoder(vision_cfg)

        x = mx.random.normal((1, 112, 112, 3))
        fpn_out = encoder(x)
        self.assertIsInstance(fpn_out, list)
        self.assertEqual(len(fpn_out), 2)
        # 1x scale = 8x8 (112/14), 2x scale = 16x16
        self.assertEqual(fpn_out[1].shape, (1, 8, 8, 32))
        self.assertEqual(fpn_out[0].shape, (1, 16, 16, 32))

    def test_sam3_text_encoder(self):
        """CLIP text encoder produces correct shapes."""
        from mlx_vlm.models.sam3.config import TextEncoderConfig
        from mlx_vlm.models.sam3.text_encoder import TextEncoder

        cfg = TextEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=16,
            projection_dim=32,
        )
        encoder = TextEncoder(cfg, d_model=32)

        input_ids = mx.array([[1, 2, 3, 4, 0, 0]])
        out = encoder(input_ids)
        self.assertEqual(out.shape, (1, 6, 64))

    def test_sam3_detr_encoder(self):
        """DETR encoder with text cross-attention."""
        from mlx_vlm.models.sam3.config import DETREncoderConfig
        from mlx_vlm.models.sam3.encoder import DETREncoder

        cfg = DETREncoderConfig(
            hidden_size=64, num_layers=2, num_attention_heads=2, intermediate_size=128
        )
        encoder = DETREncoder(cfg)

        src = mx.random.normal((1, 16, 64))
        pos = mx.random.normal((1, 16, 64))
        text = mx.random.normal((1, 4, 64))

        out = encoder(src, pos, text)
        self.assertEqual(out.shape, (1, 16, 64))

    def test_sam3_detr_decoder(self):
        """DETR decoder with box refinement and presence token."""
        from mlx_vlm.models.sam3.config import DETRDecoderConfig
        from mlx_vlm.models.sam3.decoder import DETRDecoder

        cfg = DETRDecoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=2,
            num_queries=10,
            intermediate_size=128,
        )
        decoder = DETRDecoder(cfg)

        memory = mx.random.normal((1, 16, 64))
        text = mx.random.normal((1, 4, 64))
        pos = mx.random.normal((1, 16, 64))

        hs, boxes, presence = decoder(memory, text, pos, spatial_shape=(4, 4))
        self.assertEqual(hs.shape, (2, 1, 10, 64))  # (L, B, Q, D)
        self.assertEqual(boxes.shape, (2, 1, 10, 4))  # (L, B, Q, 4)
        self.assertEqual(presence.shape, (2, 1, 1))  # (L, B, 1)

    def test_sam3_dot_product_scoring(self):
        """DotProductScoring with scale and clamp."""
        from mlx_vlm.models.sam3.segmentation import DotProductScoring

        scorer = DotProductScoring(64)
        hs = mx.random.normal((2, 1, 10, 64))  # (L, B, Q, D)
        text = mx.random.normal((1, 4, 64))
        mask = mx.array([[1, 1, 1, 0]])

        scores = scorer(hs, text, mask)
        self.assertEqual(scores.shape, (2, 1, 10, 1))
        # Scores should be clamped to [-12, 12]
        scores_np = scores.tolist()
        for layer in scores_np:
            for batch in layer:
                for query in batch:
                    for val in query:
                        self.assertGreaterEqual(val, -12.0)
                        self.assertLessEqual(val, 12.0)

    def test_sam3_mask_decoder(self):
        """Mask decoder produces correct mask resolution."""
        from mlx_vlm.models.sam3.config import DetectorMaskDecoderConfig
        from mlx_vlm.models.sam3.segmentation import MaskDecoder

        cfg = DetectorMaskDecoderConfig(hidden_size=32, num_upsampling_stages=2)
        decoder = MaskDecoder(cfg)

        queries = mx.random.normal((1, 10, 32))
        # 2 FPN levels: 8x8, 4x4
        features = [mx.random.normal((1, 8, 8, 32)), mx.random.normal((1, 4, 4, 32))]
        encoder_hs = mx.random.normal((1, 16, 32))

        out = decoder(queries, features, encoder_hidden_states=encoder_hs)
        self.assertIn("pred_masks", out)
        self.assertIn("semantic_seg", out)
        self.assertEqual(out["pred_masks"].shape[0], 1)
        self.assertEqual(out["pred_masks"].shape[1], 10)

    def test_sam3_full_model(self):
        """Full SAM3 model instantiation and forward pass."""
        from mlx_vlm.models import sam3

        config = sam3.ModelConfig(
            detector_config=sam3.DetectorConfig(
                vision_config=sam3.VisionEncoderConfig(
                    backbone_config=sam3.ViTConfig(
                        hidden_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        intermediate_size=128,
                        image_size=112,
                        patch_size=14,
                        window_size=4,
                        global_attn_indexes=[1],
                        pretrain_image_size=56,
                    ),
                    fpn_hidden_size=32,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                ),
                text_config=sam3.TextEncoderConfig(
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    intermediate_size=128,
                    vocab_size=100,
                    max_position_embeddings=16,
                    projection_dim=32,
                ),
                detr_encoder_config=sam3.DETREncoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                ),
                detr_decoder_config=sam3.DETRDecoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    num_queries=10,
                    intermediate_size=64,
                ),
                geometry_encoder_config=sam3.GeometryEncoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                ),
                mask_decoder_config=sam3.DetectorMaskDecoderConfig(
                    hidden_size=32,
                    num_upsampling_stages=3,
                ),
            ),
        )

        model = sam3.Model(config)

        pixel_values = mx.random.normal((1, 112, 112, 3))
        input_ids = mx.array([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        attention_mask = mx.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        outputs = model.detect(pixel_values, input_ids, attention_mask)
        mx.eval(outputs)

        self.assertIn("pred_logits", outputs)
        self.assertIn("pred_boxes", outputs)
        self.assertIn("pred_masks", outputs)
        self.assertIn("presence_logits", outputs)
        self.assertEqual(outputs["pred_logits"].shape, (1, 10))
        self.assertEqual(outputs["pred_boxes"].shape, (1, 10, 4))

    def test_sam3_sanitize(self):
        """Sanitize transposes conv weights correctly."""
        from mlx_vlm.models.sam3.sam3 import Model

        weights = {
            "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight": mx.zeros(
                (64, 3, 14, 14)
            ),
            "detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.0.weight": mx.zeros(
                (128, 64, 2, 2)
            ),
            "tracker_model.memory_temporal_positional_encoding": mx.zeros(
                (7, 1, 1, 32)
            ),
            "detector_model.detr_encoder.layers.0.self_attn.q_proj.weight": mx.zeros(
                (64, 64)
            ),
        }

        sanitized = Model.sanitize(weights)

        # Conv2d: (out, in, H, W) -> (out, H, W, in)
        self.assertEqual(
            sanitized[
                "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
            ].shape,
            (64, 14, 14, 3),
        )
        # ConvTranspose2d: (in, out, H, W) -> (out, H, W, in)
        self.assertEqual(
            sanitized[
                "detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.0.weight"
            ].shape,
            (64, 2, 2, 128),
        )
        # Non-conv 4D param: unchanged
        self.assertEqual(
            sanitized["tracker_model.memory_temporal_positional_encoding"].shape,
            (7, 1, 1, 32),
        )
        # 2D weight: unchanged
        self.assertEqual(
            sanitized[
                "detector_model.detr_encoder.layers.0.self_attn.q_proj.weight"
            ].shape,
            (64, 64),
        )

    def test_sam3_quant_predicate(self):
        """quant_predicate skips convs, small embeddings, and odd dimensions."""
        from mlx_vlm.models.sam3.sam3 import Model

        class FakeModule:
            def __init__(self, shape):
                self.weight = mx.zeros(shape)

        # Should quantize: large linear in DETR
        self.assertTrue(
            Model.quant_predicate(
                "detector_model.detr_encoder.layers.0.self_attn.q_proj",
                FakeModule((256, 256)),
            )
        )
        # Should skip: conv layers
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.vision_encoder.neck.fpn_layers.0.proj1",
                FakeModule((256, 256)),
            )
        )
        # Should skip: small embedding
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.detr_decoder.query_embed", FakeModule((200, 256))
            )
        )
        # Should quantize: vision encoder linear (not skipped for better compression)
        self.assertTrue(
            Model.quant_predicate(
                "detector_model.vision_encoder.backbone.layers.0.attention.q_proj",
                FakeModule((1024, 1024)),
            )
        )
        # Should skip: patch_embeddings (conv)
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection",
                FakeModule((1024, 1024)),
            )
        )
        # Should skip: odd dimension
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.geometry_encoder.boxes_pos_enc_project",
                FakeModule((256, 258)),
            )
        )

    def test_sam3_position_encoding(self):
        """Sinusoidal position encoding and 2D RoPE produce correct shapes."""
        from mlx_vlm.models.sam3.position import (
            PositionEmbeddingSine,
            compute_axial_cis,
        )

        pos_enc = PositionEmbeddingSine(num_pos_feats=32)
        x = mx.random.normal((1, 8, 8, 64))
        pos = pos_enc(x)
        self.assertEqual(pos.shape, (1, 8, 8, 64))

        cos, sin = compute_axial_cis(64, 8, 8)
        self.assertEqual(cos.shape, (64, 64))
        self.assertEqual(sin.shape, (64, 64))


class TestRTDetrV2(unittest.TestCase):
    def test_config_from_dict(self):
        """Flat HF config dict resolves into nested sub-configs."""
        from mlx_vlm.models.rt_detr_v2 import ModelConfig, RTDetrResNetConfig

        hf_like = {
            "model_type": "rt_detr_v2",
            "image_size": 64,
            "num_labels": 10,
            "backbone_config": {
                "model_type": "rt_detr_resnet",
                "depths": [1, 1, 2, 1],
                "hidden_sizes": [10, 20, 30, 40],
            },
            "encoder_hidden_dim": 32,
            "encoder_in_channels": [20, 30, 40],
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "d_model": 32,
            "num_queries": 30,
            "decoder_layers": 2,
            "decoder_attention_heads": 2,
            "decoder_ffn_dim": 64,
            "decoder_in_channels": [32, 32, 32],
        }
        cfg = ModelConfig.from_dict(hf_like)
        self.assertEqual(cfg.model_type, "rt_detr_v2")
        self.assertEqual(cfg.num_labels, 10)
        self.assertIsInstance(cfg.backbone_config, RTDetrResNetConfig)
        self.assertEqual(cfg.backbone_config.depths, [1, 1, 2, 1])
        # __post_init__ rebuilds the sub-configs from flat fields
        self.assertEqual(cfg._hybrid_encoder_config.encoder_hidden_dim, 32)
        self.assertEqual(cfg._transformer_config.d_model, 32)
        self.assertEqual(cfg._transformer_config.num_queries, 30)
        self.assertEqual(cfg._transformer_config.num_labels, 10)
        # framework-compat hooks
        self.assertIsNone(cfg.text_config)
        self.assertIsNone(cfg.vision_config)

    def test_forward_shapes(self):
        """End-to-end forward on a tiny config returns the documented shapes."""
        from mlx_vlm.models.rt_detr_v2 import Model, ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "model_type": "rt_detr_v2",
                "image_size": 64,
                "num_labels": 10,
                "backbone_config": {
                    "model_type": "rt_detr_resnet",
                    "depths": [1, 1, 2, 1],
                    "hidden_sizes": [10, 20, 30, 40],
                },
                "encoder_hidden_dim": 32,
                "encoder_in_channels": [20, 30, 40],
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 64,
                "d_model": 32,
                "num_queries": 30,
                "decoder_layers": 2,
                "decoder_attention_heads": 2,
                "decoder_ffn_dim": 64,
                "decoder_in_channels": [32, 32, 32],
            }
        )
        model = Model(cfg)
        model.eval()

        pixel = mx.random.normal((2, 64, 64, 3))
        out = model(pixel)
        mx.eval(out["pred_logits"], out["pred_boxes"])

        self.assertEqual(out["pred_logits"].shape, (2, 30, 10))
        self.assertEqual(out["pred_boxes"].shape, (2, 30, 4))
        # intermediate trajectories cover all decoder layers
        self.assertEqual(out["intermediate_logits"].shape, (2, 2, 30, 10))
        self.assertEqual(out["intermediate_reference_points"].shape, (2, 2, 30, 4))

    def test_sanitize_renames_and_transposes(self):
        """`Model.sanitize` rewrites HF keys and transposes NCHW conv weights."""
        from mlx_vlm.models.rt_detr_v2 import Model

        raw = {
            # NCHW conv weight under the HF backbone prefix -> NHWC under
            # vision.backbone after sanitize
            "model.backbone.model.embedder.embedder.0.convolution.weight": mx.zeros(
                (8, 3, 7, 7)
            ),
            # encoder body: `normalization` -> `bn`
            "model.encoder.0.normalization.weight": mx.ones((16,)),
            # vd downsampling shortcut: Sequential index 1 -> proj
            "model.backbone.model.encoder.stages.1.layers.0.shortcut.1.convolution.weight": mx.zeros(
                (16, 8, 1, 1)
            ),
            # num_batches_tracked must be dropped
            "model.backbone.model.embedder.embedder.0.normalization.num_batches_tracked": mx.array(
                0
            ),
        }
        sanitized = Model.sanitize(raw)

        self.assertNotIn(
            "model.backbone.model.embedder.embedder.0.normalization.num_batches_tracked",
            sanitized,
        )
        self.assertIn("vision.backbone.embedder.embedder.0.conv.weight", sanitized)
        # NCHW (8, 3, 7, 7) -> NHWC (8, 7, 7, 3)
        self.assertEqual(
            sanitized["vision.backbone.embedder.embedder.0.conv.weight"].shape,
            (8, 7, 7, 3),
        )
        # `.normalization.` rename under the encoder body
        self.assertIn("vision.hybrid_encoder.0.bn.weight", sanitized)
        # vd shortcut Sequential[1] -> proj
        self.assertIn(
            "vision.backbone.encoder.stages.1.layers.0.shortcut.proj.conv.weight",
            sanitized,
        )


if __name__ == "__main__":
    unittest.main()


class TestKimiK25VideoVision(unittest.TestCase):
    """kimi_k25 MoonViT tower: a 3-col (t, h, w) video grid must run end to end
    (temporal pos-emb, t-tiled RoPE, t*h*w cu_seqlens, block-diagonal attention,
    temporal pooling) while the 2-col (h, w) image path stays unchanged."""

    P, C, EMBED, HEADS = 4, 3, 64, 4

    @classmethod
    def _tower(cls, seed=0):
        from mlx_vlm.models.kimi_k25.config import VisionConfig
        from mlx_vlm.models.kimi_k25.vision import VisionModel

        cfg = VisionConfig(
            depth=2,
            embed_dim=cls.EMBED,
            hidden_size=cls.EMBED,
            num_heads=cls.HEADS,
            patch_size=cls.P,
            num_channels=cls.C,
            init_pos_emb_height=8,
            intermediate_size=128,
            spatial_merge_size=2,
        )
        cfg.merge_kernel_size = (2, 2)
        mx.random.seed(seed)
        vm = VisionModel(cfg)
        mx.eval(vm.parameters())
        return vm

    @classmethod
    def _pixels(cls, n, seed=123):
        mx.random.seed(seed)
        return mx.random.normal((n, cls.P, cls.P, cls.C))

    def test_image_path_unchanged(self):
        # New block-diagonal attention must equal the old full [1, seq, seq]-mask
        # attention for a single segment (the image case).
        from mlx_vlm.models.kimi_k25.vision import Attention, apply_rope

        vm = self._tower()
        att = Attention(self.EMBED, self.HEADS)
        mx.eval(att.parameters())
        n = 16
        mx.random.seed(7)
        x = mx.random.normal((n, self.EMBED))
        rope = vm.rope_pos_emb.get_freqs_cis(mx.array([[4, 4]]))
        cu = mx.array([0, n], dtype=mx.int32)

        def old_full_mask(att, x, cu, rope):
            seq = x.shape[0]
            qkv = att.wqkv(x).reshape(seq, 3, att.num_heads, att.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q, k = apply_rope(q, k, rope)
            mask = mx.zeros((1, seq, seq), dtype=x.dtype)
            for i in range(1, len(cu)):
                s, e = int(cu[i - 1]), int(cu[i])
                mask[..., s:e, s:e] = 1
            q, k, v = (z.transpose(1, 0, 2) for z in (q, k, v))
            aw = q @ k.swapaxes(-2, -1) / mx.sqrt(q.shape[-1])
            aw = mx.softmax(aw + mask, axis=-1).astype(q.dtype)
            return att.wo((aw @ v).transpose(1, 0, 2).reshape(seq, -1))

        self.assertTrue(
            mx.allclose(att(x, cu, rope), old_full_mask(att, x, cu, rope), atol=1e-5)
        )

    def test_video_grid_runs_and_image_still_works(self):
        vm = self._tower()
        img = vm(self._pixels(4 * 4), mx.array([[4, 4]]))
        self.assertEqual(img[0].shape, (4, 4, self.EMBED))
        vid = vm(self._pixels(2 * 4 * 4), mx.array([[2, 4, 4]]))
        mx.eval(vid[0])
        self.assertEqual(vid[0].shape, (4, 4, self.EMBED))
        self.assertFalse(bool(mx.any(mx.isnan(vid[0]))))

    def test_temporal_pool_collapses_frames(self):
        # Merged token count must be independent of t (sd2_tpool over frames).
        vm = self._tower()
        counts = {
            t: vm(self._pixels(t * 4 * 4), mx.array([[t, 4, 4]]))[0].shape[0]
            for t in (1, 2, 4)
        }
        self.assertEqual(set(counts.values()), {(4 // 2) * (4 // 2)})

    def test_block_diagonal_no_cross_chunk_leakage(self):
        from mlx_vlm.models.kimi_k25.vision import Attention

        vm = self._tower()
        att = Attention(self.EMBED, self.HEADS)
        mx.eval(att.parameters())
        n_a = n_b = 8
        cu = mx.array([0, n_a, n_a + n_b], dtype=mx.int32)
        rope = vm.rope_pos_emb.get_freqs_cis(mx.array([[2, 4], [2, 4]]))
        mx.random.seed(9)
        x = mx.random.normal((n_a + n_b, self.EMBED))
        out_b = att(x, cu, rope)[n_a:]
        xp = mx.array(x)
        xp[:n_a] = mx.random.normal((n_a, self.EMBED))
        out_b_perturbed = att(xp, cu, rope)[n_a:]
        self.assertTrue(mx.allclose(out_b, out_b_perturbed, atol=1e-6))


class TestDeepseekV4HISA(unittest.TestCase):
    """HISA hierarchical indexer (deepseek_v4): block-coarse -> token-fine
    selection must match the flat top-k scan in the keep-all limit and recover
    most of the top-k when relevance is block-clustered."""

    @staticmethod
    def _indexer(
        index_block,
        index_keep,
        index_topk=32,
        n_heads=8,
        head_dim=64,
        hidden=256,
        q_lora_rank=64,
        seed=0,
    ):
        from mlx_vlm.models.deepseek_v4.config import ModelConfig
        from mlx_vlm.models.deepseek_v4.language import Indexer

        mx.random.seed(seed)
        cfg = ModelConfig(
            hidden_size=hidden,
            q_lora_rank=q_lora_rank,
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=index_topk,
            index_block=index_block,
            index_keep=index_keep,
        )
        ix = Indexer(cfg, compress_ratio=4)
        mx.eval(ix.parameters())
        return ix

    @staticmethod
    def _flat_select(ix, q, pooled, x, k):
        f32 = mx.float32
        s = mx.maximum(q.astype(f32) @ pooled[:, None].swapaxes(-1, -2).astype(f32), 0)
        s = s * ix.scale
        w = ix.weights_proj(x).astype(f32) * (ix.n_heads**-0.5)
        s = (s * w.swapaxes(-1, -2)[..., None]).sum(axis=1)
        return mx.argpartition(-s, kth=k - 1, axis=-1)[..., :k]

    @staticmethod
    def _recall(a, b, k):
        a, b = a.reshape(-1, k).tolist(), b.reshape(-1, k).tolist()
        return sum(len(set(x) & set(y)) for x, y in zip(a, b)) / (len(a) * k)

    def test_hisa_equals_flat_when_all_blocks_kept(self):
        Np, block = 512, 64
        ix = self._indexer(index_block=block, index_keep=Np // block, index_topk=32)
        mx.random.seed(1)
        q = mx.random.normal((2, ix.n_heads, 1, ix.head_dim))
        pooled = mx.random.normal((2, Np, ix.head_dim))
        x = mx.random.normal((2, 1, 256))
        k = min(ix.index_topk, Np)
        hisa = ix._hisa_select(q, pooled, x, k)
        flat = self._flat_select(ix, q, pooled, x, k)
        self.assertEqual(self._recall(hisa, flat, k), 1.0)

    def test_hisa_shape_and_valid_indices(self):
        Np = 2048
        ix = self._indexer(index_block=64, index_keep=8, index_topk=32)
        mx.random.seed(1)
        q = mx.random.normal((2, ix.n_heads, 1, ix.head_dim))
        pooled = mx.random.normal((2, Np, ix.head_dim))
        x = mx.random.normal((2, 1, 256))
        k = min(ix.index_topk, Np)
        out = ix._hisa_select(q, pooled, x, k)
        self.assertEqual(out.shape, (2, 1, k))
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLess(int(out.max()), Np)

    def test_hisa_high_recall_on_clustered_prefix(self):
        Np, block = 4096, 64
        ix = self._indexer(index_block=block, index_keep=8, index_topk=64)
        mx.random.seed(1)
        nb = Np // block
        dc = mx.random.normal((nb, 1, ix.head_dim)) * 2.0
        pooled = (dc + mx.random.normal((nb, block, ix.head_dim))).reshape(
            1, Np, ix.head_dim
        )
        q = mx.random.normal((1, ix.n_heads, 1, ix.head_dim))
        x = mx.random.normal((1, 1, 256))
        k = ix.index_topk
        r = self._recall(
            ix._hisa_select(q, pooled, x, k), self._flat_select(ix, q, pooled, x, k), k
        )
        self.assertGreaterEqual(r, 0.7)

    def test_hisa_batched_l_gt_1_matches_flat(self):
        # L>1 batched HISA (hisa_kernel.hisa_select): keep-all must return the
        # exact flat per-query top-k for every query position.
        from mlx_vlm.models.deepseek_v4.hisa_kernel import hisa_select

        mx.random.seed(2)
        H, Dh, Np, k, block = 8, 64, 512, 16, 64
        scale = Dh**-0.5
        q = mx.random.normal((1, H, 4, Dh))  # L = 4 queries
        pooled = mx.random.normal((1, Np, Dh))
        weights = mx.random.normal((1, 4, H)) * (H**-0.5)
        out = hisa_select(q, pooled, weights, scale, k, block, index_keep=Np // block)
        wk_h = (weights * scale).transpose(0, 2, 1)[..., None]
        flat = (mx.maximum(q @ pooled[:, None].swapaxes(-1, -2), 0) * wk_h).sum(1)
        ftk = mx.argpartition(-flat, kth=k - 1, axis=-1)[..., :k]
        self.assertTrue(bool(mx.array_equal(mx.sort(out, -1), mx.sort(ftk, -1))))


class TestQuantizedKVCacheMask(unittest.TestCase):
    """Mask-length checks must handle quantized KV caches, whose
    ``update_and_fetch`` returns packed tuples instead of arrays (#1481)."""

    def test_kv_sequence_length(self):
        from mlx_vlm.models import gemma3
        from mlx_vlm.models.base import kv_sequence_length
        from mlx_vlm.models.cache import KVCache, QuantizedKVCache

        keys = mx.zeros((1, 2, 5, 64))
        self.assertEqual(kv_sequence_length(keys), 5)
        quantized = mx.quantize(keys, group_size=64, bits=8)
        self.assertEqual(kv_sequence_length(quantized), 5)

        config = gemma3.TextConfig(
            model_type="gemma3",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=64,
            sliding_window=8,
            sliding_window_pattern=2,
        )
        model = gemma3.LanguageModel(config)

        # With --kv-bits the server quantizes global-attention layers from
        # token 0; sliding layers keep their RotatingKVCache.
        cache = [
            QuantizedKVCache(group_size=64, bits=8) if isinstance(c, KVCache) else c
            for c in model.make_cache()
        ]

        prompt = mx.arange(12)[None]  # longer than the sliding window
        logits = model(prompt, cache=cache).logits
        self.assertEqual(logits.shape, (1, 12, config.vocab_size))

        decode = model(mx.array([[3]]), cache=cache).logits
        chunk = model(mx.arange(4)[None], cache=cache).logits  # chunked prefill
        mx.eval(logits, decode, chunk)
        self.assertTrue(mx.isfinite(chunk).all().item())

        # The quantized path must actually have been exercised.
        self.assertIsInstance(cache[1].keys, tuple)


class TestQwen35NormSanitization(unittest.TestCase):
    _HF_VL_KEY = "model.language_model.layers.0.input_layernorm.weight"
    _HF_TEXT_KEY = "model.layers.0.input_layernorm.weight"
    _MLX_KEY = "language_model.model.layers.0.input_layernorm.weight"

    def _sanitize(self, module, key):
        stub = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(
                    tie_word_embeddings=False, num_hidden_layers=0
                )
            ),
        )
        return module.Model.sanitize(stub, {key: mx.zeros(4)})

    def test_qwen3_5_shifts_hf_vl_norm_weights(self):
        from mlx_vlm.models import qwen3_5

        out = self._sanitize(qwen3_5, self._HF_VL_KEY)
        self.assertTrue(mx.allclose(out[self._MLX_KEY], mx.ones(4)).item())

    def test_qwen3_5_shifts_hf_text_norm_weights(self):
        from mlx_vlm.models import qwen3_5

        out = self._sanitize(qwen3_5, self._HF_TEXT_KEY)
        self.assertTrue(mx.allclose(out[self._HF_TEXT_KEY], mx.ones(4)).item())

    def test_qwen3_5_preserves_mlx_norm_weights(self):
        from mlx_vlm.models import qwen3_5

        out = self._sanitize(qwen3_5, self._MLX_KEY)
        self.assertTrue(mx.allclose(out[self._MLX_KEY], mx.zeros(4)).item())

    def test_qwen3_5_moe_shifts_hf_vl_norm_weights(self):
        from mlx_vlm.models import qwen3_5_moe

        out = self._sanitize(qwen3_5_moe, self._HF_VL_KEY)
        self.assertTrue(mx.allclose(out[self._MLX_KEY], mx.ones(4)).item())

    def test_qwen3_5_moe_preserves_mlx_norm_weights(self):
        from mlx_vlm.models import qwen3_5_moe

        out = self._sanitize(qwen3_5_moe, self._MLX_KEY)
        self.assertTrue(mx.allclose(out[self._MLX_KEY], mx.zeros(4)).item())


class TestQwenMRoPEDecodeContinuation(unittest.TestCase):
    """Regression tests for MRoPE decode positions in plain generation.

    ``generate()`` forwards ``position_ids``/``rope_deltas`` (computed by
    ``get_input_embeddings``) only with the prefill call; decode steps reach
    the language model with no position kwargs. The language model must
    persist the prefill ``rope_deltas`` so decode positions continue in the
    compressed MRoPE space instead of restarting from a single-token
    recalculation (#1505, #1526).
    """

    def _tiny_qwen3_vl(self):
        from mlx_vlm.models import qwen3_vl

        text_config = qwen3_vl.TextConfig(
            model_type="qwen3_vl_text",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            head_dim=16,
            vocab_size=1000,
            rope_theta=1000,
            max_position_embeddings=1000,
            tie_word_embeddings=False,
            norm_topk_prob=True,
            rope_scaling={"rope_type": "mrope", "mrope_section": [4, 2, 2]},
        )
        vision_config = qwen3_vl.VisionConfig(
            model_type="qwen3_vl",
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            out_hidden_size=64,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=144,
            deepstack_visual_indexes=[],
        )
        config = qwen3_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="qwen3_vl",
            image_token_id=998,
            video_token_id=999,
            vocab_size=1000,
        )
        return qwen3_vl.Model(config).language_model

    def _decode_logits(self, lm, with_delta_kwarg):
        from mlx_vlm.models import cache as cache_mod

        lm._rope_deltas = None
        lm._position_ids = None
        prompt_cache = cache_mod.make_prompt_cache(lm)

        ids = mx.array([[3, 5, 7, 11, 13, 17]])
        # Vision-compressed prompt positions: max position 2 for 6 tokens,
        # so rope_delta = (2 + 1) - 6 = -3.
        pos_row = mx.array([0, 1, 1, 2, 2, 2])
        positions = mx.broadcast_to(pos_row[None, None, :], (3, 1, 6))
        delta = mx.array([[-3]])

        prefill_kwargs = {"rope_deltas": delta} if with_delta_kwarg else {}
        lm(ids, cache=prompt_cache, position_ids=positions, **prefill_kwargs)

        step = mx.array([[19]])
        if with_delta_kwarg:
            # The generate() decode flow: no position kwargs at all.
            out = lm(step, cache=prompt_cache)
        else:
            # Ground truth: cache offset (6) + delta (-3) = position 3.
            decode_positions = mx.full((3, 1, 1), 3, dtype=positions.dtype)
            out = lm(step, cache=prompt_cache, position_ids=decode_positions)
        mx.eval(out.logits)
        return out.logits

    def test_qwen3_vl_decode_continues_prefill_rope_deltas(self):
        lm = self._tiny_qwen3_vl()
        mx.eval(lm.parameters())
        reference = self._decode_logits(lm, with_delta_kwarg=False)
        subject = self._decode_logits(lm, with_delta_kwarg=True)
        self.assertTrue(mx.allclose(reference, subject, atol=1e-5).item())
