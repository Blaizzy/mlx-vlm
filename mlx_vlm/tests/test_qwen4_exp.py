import unittest

import mlx.core as mx

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models import qwen4_exp
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import (
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpNGramEmbedding,
)
from mlx_vlm.prompt_utils import MessageFormat, MessageFormatter


def tiny_config():
    text_config = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision_config = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    return qwen4_exp.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )


class Qwen4ExpTests(unittest.TestCase):
    def test_config_normalizes_reference_layer_type(self):
        config = tiny_config()
        self.assertEqual(
            config.text_config.layer_types,
            ["linear_attention", "qwen_sparse_attention"],
        )
        self.assertEqual(config.text_config.rope_parameters["type"], "default")
        self.assertEqual(
            MessageFormatter("qwen4_exp").format_type,
            MessageFormat.LIST_WITH_IMAGE_FIRST,
        )

    def test_ngram_embedding_chunked_cache_matches_full_sequence(self):
        config = tiny_config().text_config
        embedding = Qwen4ExpNGramEmbedding(config, 32, 0, 0)

        # Make each row encode its global row id, independent of shard layout.
        for shard, start, end in zip(
            embedding.ngram_embedding.shards,
            embedding.ngram_embedding.shard_offsets[:-1],
            embedding.ngram_embedding.shard_offsets[1:],
        ):
            values = mx.arange(start, end, dtype=mx.float32)[:, None]
            shard.weight = mx.broadcast_to(values, shard.weight.shape)

        input_ids = mx.array([[1, 2, 3, 1, 4, 5]], dtype=mx.int32)
        full = embedding(input_ids, cache=None)
        cache = ArraysCache(size=4)
        prefix = embedding(input_ids[:, :4], cache=cache)
        suffix = embedding(input_ids[:, 4:], cache=cache)
        mx.eval(full, prefix, suffix)

        self.assertTrue(mx.array_equal(full[:, :4], prefix).item())
        self.assertTrue(mx.array_equal(full[:, 4:], suffix).item())
        self.assertEqual(cache[3].shape, (1, config.ngram_size - 1))

    def test_gated_delta_uses_reference_l2_normalization(self):
        layer = Qwen4ExpGatedDeltaNet(tiny_config().text_config)
        query = mx.random.normal((1, 3, 2, 8))
        key = mx.random.normal((1, 3, 2, 8))

        normalized_query, normalized_key = layer._normalize_qk(query, key)
        expected_query = (
            query
            * mx.rsqrt(mx.sum(mx.square(query), axis=-1, keepdims=True) + 1e-6)
            / mx.sqrt(mx.array(8.0))
        )
        expected_key = key * mx.rsqrt(
            mx.sum(mx.square(key), axis=-1, keepdims=True) + 1e-6
        )
        mx.eval(normalized_query, normalized_key, expected_query, expected_key)

        self.assertTrue(mx.allclose(normalized_query, expected_query).item())
        self.assertTrue(mx.allclose(normalized_key, expected_key).item())

    def test_forward_sparse_attention_and_cached_decode(self):
        model = qwen4_exp.Model(tiny_config())
        input_ids = mx.arange(12, dtype=mx.int32)[None]
        prefill_positions = mx.broadcast_to(mx.arange(10)[None, None], (3, 1, 10))

        full_logits = model.language_model(input_ids).logits
        cache = model.language_model.make_cache()
        prefix_logits = model.language_model(
            input_ids[:, :10], cache=cache, position_ids=prefill_positions
        ).logits
        decode_logits = model.language_model(
            input_ids[:, 10:],
            cache=cache,
            position_ids=mx.arange(10, 12)[None],
        ).logits
        mx.eval(full_logits, prefix_logits, decode_logits)

        self.assertEqual(full_logits.shape, (1, 12, 64))
        self.assertEqual(decode_logits.shape, (1, 2, 64))
        self.assertLess(
            mx.max(mx.abs(full_logits[:, 10:] - decode_logits)).item(), 1e-3
        )
        self.assertEqual(cache[1].index_keys.shape, (1, 12, 8))
        self.assertEqual(cache[1].index_position_ids.shape, (3, 1, 12))
        self.assertEqual(cache[0][2].shape[1], 6)
        self.assertEqual(cache[0][3].shape[1], 2)

    def test_multimodal_forward_uses_qwen3_vision_encoder(self):
        model = qwen4_exp.Model(tiny_config())
        input_ids = mx.array([[58, 60, 59, 1]], dtype=mx.int32)
        pixels = mx.zeros((4, 3 * 2 * 14 * 14), dtype=mx.float32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

        logits = model(
            input_ids,
            pixel_values=pixels,
            image_grid_thw=image_grid_thw,
        ).logits
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 4, 64))

    def test_uniform_kv_quantization_preserves_qsa_indexer_state(self):
        cache = QSAKVCache()
        cache.update_and_fetch(mx.zeros((1, 2, 10, 32)), mx.zeros((1, 2, 10, 32)))
        cache.update_indexer(mx.zeros((1, 10, 8)), mx.arange(10, dtype=mx.int32)[None])
        prompt_cache = [cache]
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=8,
        )
        quantized = prompt_cache[0]
        quantized.update_and_fetch(mx.zeros((1, 2, 2, 32)), mx.zeros((1, 2, 2, 32)))
        quantized.update_indexer(
            mx.zeros((1, 2, 8)), mx.arange(10, 12, dtype=mx.int32)[None]
        )

        self.assertIsInstance(quantized, QSAQuantizedKVCache)
        self.assertEqual(quantized.index_keys.shape, (1, 12, 8))
        self.assertEqual(quantized.offset, 12)

    def test_sanitize_maps_packed_experts_and_ngram_shards(self):
        model = qwen4_exp.Model(tiny_config())
        prefix = "model.language_model.layers.0"
        weights = {
            f"{prefix}.mlp.experts.gate_up_proj": mx.zeros((4, 32, 32)),
            f"{prefix}.mlp.experts.down_proj": mx.zeros((4, 32, 16)),
            f"{prefix}.ple.ple_embedding.ngram_embedding.shard_0.weight": (
                mx.zeros((25, 8))
            ),
            f"{prefix}.ple.conv1d.weight": mx.zeros((64, 1, 3)),
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((1, 1)),
        }

        sanitized = model.sanitize(weights)
        mapped = "language_model.model.layers.0"
        self.assertIn(f"{mapped}.mlp.switch_mlp.gate_proj.weight", sanitized)
        self.assertIn(f"{mapped}.mlp.switch_mlp.up_proj.weight", sanitized)
        self.assertIn(f"{mapped}.mlp.switch_mlp.down_proj.weight", sanitized)
        self.assertIn(
            f"{mapped}.ple.ple_embedding.ngram_embedding.shards.0.weight",
            sanitized,
        )
        self.assertEqual(sanitized[f"{mapped}.ple.conv1d.weight"].shape, (64, 3, 1))
        self.assertFalse(any(key.startswith("mtp.") for key in sanitized))


if __name__ == "__main__":
    unittest.main()
