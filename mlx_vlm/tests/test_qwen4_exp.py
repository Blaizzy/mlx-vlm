import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import qwen4_exp
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpGatedDeltaNet,
    _create_qwen4_exp_attention_mask,
)
from mlx_vlm.models.qwen4_exp.qsa_kernel import (
    QSAExecutionPlan,
    select_qsa_execution_plan,
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
    def test_quantized_decode_uses_the_batch_invariant_path(self):
        model = qwen4_exp.Model(tiny_config())
        self.assertTrue(model.language_model._supports_batch_invariant_decode())

        model.language_model.lm_head = nn.QuantizedLinear.from_linear(
            model.language_model.lm_head, group_size=32, bits=4
        )

        self.assertFalse(model.language_model._supports_batch_invariant_decode())

        supported_head = nn.Linear(512, 64, bias=False)
        supported_head.set_dtype(mx.bfloat16)
        model.language_model.lm_head = nn.QuantizedLinear.from_linear(
            supported_head, group_size=32, bits=4
        )

        self.assertTrue(model.language_model._supports_batch_invariant_decode())

    def test_qsa_short_prefix_selection_uses_sentinels(self):
        model = qwen4_exp.Model(tiny_config())
        indexer = model.language_model.model.layers[1].self_attn.indexer
        selection = indexer.select(
            mx.random.normal((1, 6, 32)), cache=QSAKVCache(), position_ids=None
        )
        mx.eval(selection.selected_blocks, selection.complete_counts)

        self.assertEqual(selection.selected_blocks.shape, (1, 6, 4))
        for position in range(6):
            complete = int(selection.complete_counts[0, position].item())
            blocks = sorted(selection.selected_blocks[0, position].tolist())
            self.assertEqual(blocks.count(-1), 4 - complete)
            self.assertEqual(
                [block for block in blocks if block >= 0], list(range(complete))
            )

    def test_ragged_qsa_batch_retains_explicit_mask(self):
        hidden = mx.zeros((2, 32, 64))
        cache = BatchQSAKVCache([3, 0])

        mask = _create_qwen4_exp_attention_mask(hidden, cache)

        self.assertIsInstance(mask, mx.array)
        self.assertEqual(mask.shape, (2, 1, 32, 32))

    def test_qsa_dispatch_is_explicit_for_decode_and_masked_attention(self):
        queries = mx.zeros((2, 4, 1, 32), dtype=mx.bfloat16)
        keys = mx.zeros((2, 2, 64, 32), dtype=mx.bfloat16)
        values = mx.zeros_like(keys)
        blocks = mx.zeros((2, 1, 4), dtype=mx.int32)
        query_ends = mx.full((2, 1), 64, dtype=mx.int32)

        decode = select_qsa_execution_plan(
            queries, keys, values, blocks, query_ends, block_size=2, causal=True
        )
        masked = select_qsa_execution_plan(
            queries, keys, values, blocks, query_ends, block_size=2, causal=False
        )

        self.assertIs(decode, QSAExecutionPlan.DENSE_DECODE)
        self.assertIs(masked, QSAExecutionPlan.DENSE_MASKED)

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
            input_ids[:, 10:], cache=cache, position_ids=mx.arange(10, 12)[None]
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

    def _assert_batched_qsa_decode_matches_singleton_rows(self, model):
        prompts = mx.array([list(range(2, 12)), list(range(12, 22))], dtype=mx.int32)
        decode = mx.array([[22], [23]], dtype=mx.int32)
        prompt_positions = mx.broadcast_to(mx.arange(10)[None], (2, 10))
        decode_positions = mx.full((2, 1), 10, dtype=mx.int32)

        batch_cache = _make_cache(model.language_model, [0, 0])
        model.language_model(prompts, cache=batch_cache, position_ids=prompt_positions)
        batch_logits = model.language_model(
            decode, cache=batch_cache, position_ids=decode_positions
        ).logits

        row_logits = []
        for row in range(2):
            row_cache = model.language_model.make_cache()
            model.language_model(
                prompts[row : row + 1],
                cache=row_cache,
                position_ids=prompt_positions[row : row + 1],
            )
            row_logits.append(
                model.language_model(
                    decode[row : row + 1],
                    cache=row_cache,
                    position_ids=decode_positions[row : row + 1],
                ).logits
            )
        row_logits = mx.concatenate(row_logits, axis=0)
        mx.eval(batch_logits, row_logits)

        self.assertTrue(mx.array_equal(batch_logits, row_logits).item())
        self.assertTrue(
            mx.array_equal(
                mx.argmax(batch_logits, axis=-1), mx.argmax(row_logits, axis=-1)
            ).item()
        )

    def test_batched_qsa_cache_decode_matches_singleton_rows(self):
        mx.random.seed(47)
        model = qwen4_exp.Model(tiny_config())
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        self._assert_batched_qsa_decode_matches_singleton_rows(model)

    def test_quantized_batched_qsa_decode_matches_singleton_rows(self):
        mx.random.seed(48)
        model = qwen4_exp.Model(tiny_config())
        model.set_dtype(mx.bfloat16)

        def quantizable(path, module):
            weight = getattr(module, "weight", None)
            return (
                path != "lm_head"
                and hasattr(module, "to_quantized")
                and weight is not None
                and weight.shape[-1] % 32 == 0
            )

        nn.quantize(
            model.language_model,
            group_size=32,
            bits=5,
            mode="affine",
            class_predicate=quantizable,
        )
        mx.eval(model.parameters())

        self._assert_batched_qsa_decode_matches_singleton_rows(model)

    def test_chunked_ragged_prefill_handles_an_all_padding_row_chunk(self):
        mx.random.seed(29)
        model = qwen4_exp.Model(tiny_config())
        row_prompts = [list(range(2, 12)), list(range(12, 26))]
        prompts = mx.array(
            [[0, 0, 0, 0, *row_prompts[0]], row_prompts[1]], dtype=mx.int32
        )
        positions = mx.array(
            [[-4, -3, -2, -1, *range(10)], list(range(14))], dtype=mx.int32
        )
        decode = mx.array([[26], [27]], dtype=mx.int32)
        decode_positions = mx.array([[10], [14]], dtype=mx.int32)

        batch_cache = _make_cache(model.language_model, [4, 0])
        model.language_model(
            prompts[:, :2], cache=batch_cache, position_ids=positions[:, :2]
        )
        model.language_model(
            prompts[:, 2:], cache=batch_cache, position_ids=positions[:, 2:]
        )
        batch_logits = model.language_model(
            decode, cache=batch_cache, position_ids=decode_positions
        ).logits

        row_logits = []
        for row, prompt in enumerate(row_prompts):
            row_cache = model.language_model.make_cache()
            model.language_model(
                mx.array(prompt, dtype=mx.int32)[None],
                cache=row_cache,
                position_ids=mx.arange(len(prompt), dtype=mx.int32)[None],
            )
            row_logits.append(
                model.language_model(
                    decode[row : row + 1],
                    cache=row_cache,
                    position_ids=decode_positions[row : row + 1],
                ).logits
            )
        row_logits = mx.concatenate(row_logits, axis=0)
        mx.eval(batch_logits, row_logits)

        self.assertTrue(
            mx.array_equal(
                mx.argmax(batch_logits, axis=-1), mx.argmax(row_logits, axis=-1)
            ).item()
        )

    def test_multimodal_forward_uses_qwen3_vision_encoder(self):
        model = qwen4_exp.Model(tiny_config())
        input_ids = mx.array([[58, 60, 59, 1]], dtype=mx.int32)
        pixels = mx.zeros((4, 3 * 2 * 14 * 14), dtype=mx.float32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

        logits = model(
            input_ids, pixel_values=pixels, image_grid_thw=image_grid_thw
        ).logits
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 4, 64))

    def test_uniform_kv_quantization_preserves_qsa_indexer_state(self):
        cache = QSAKVCache()
        cache.update_and_fetch(mx.zeros((1, 2, 10, 32)), mx.zeros((1, 2, 10, 32)))
        cache.update_indexer(mx.zeros((1, 10, 8)), mx.arange(10, dtype=mx.int32)[None])
        prompt_cache = [cache]
        maybe_quantize_kv_cache(
            prompt_cache, quantized_kv_start=0, kv_group_size=32, kv_bits=8
        )
        quantized = prompt_cache[0]
        quantized.update_and_fetch(mx.zeros((1, 2, 2, 32)), mx.zeros((1, 2, 2, 32)))
        quantized.update_indexer(
            mx.zeros((1, 2, 8)), mx.arange(10, 12, dtype=mx.int32)[None]
        )

        self.assertIsInstance(quantized, QSAQuantizedKVCache)
        self.assertEqual(quantized.index_keys.shape, (1, 12, 8))
        self.assertEqual(quantized.offset, 12)

    def test_qsa_cache_merges_ragged_rows_and_round_trips_extract(self):
        rows = []
        for length in (3, 1):
            cache = QSAKVCache()
            cache.update_and_fetch(
                mx.ones((1, 2, length, 8)) * length,
                mx.ones((1, 2, length, 8)) * (length + 1),
            )
            cache.update_indexer(
                mx.arange(length * 8).reshape(1, length, 8),
                mx.arange(length, dtype=mx.int32)[None],
            )
            rows.append(cache)

        batch = QSAKVCache.merge(rows)
        self.assertIsInstance(batch, BatchQSAKVCache)
        self.assertEqual(batch.index_keys.shape, (2, 3, 8))
        self.assertEqual(batch.index_position_ids.shape, (2, 3))
        self.assertEqual(batch.left_padding.tolist(), [0, 2])

        restored = batch.extract(1)
        mx.eval(*restored.state)
        self.assertEqual(restored.offset, 1)
        self.assertTrue(mx.array_equal(restored.index_keys, rows[1].index_keys).item())
        self.assertTrue(
            mx.array_equal(
                restored.index_position_ids, rows[1].index_position_ids
            ).item()
        )

        cloned = BatchQSAKVCache.from_state(batch.state, batch.meta_state)
        cloned_row = cloned.extract(1)
        mx.eval(*cloned_row.state)
        self.assertEqual(cloned.offset.tolist(), [3, 1])
        self.assertTrue(
            mx.array_equal(cloned_row.index_keys, rows[1].index_keys).item()
        )

    def test_qsa_batch_cache_promotes_mixed_text_and_mrope_positions(self):
        text = QSAKVCache()
        text.update_and_fetch(mx.zeros((1, 2, 2, 8)), mx.zeros((1, 2, 2, 8)))
        text_positions = mx.arange(2, dtype=mx.int32)[None]
        text.update_indexer(mx.zeros((1, 2, 8)), text_positions)

        multimodal = QSAKVCache()
        multimodal.update_and_fetch(mx.zeros((1, 2, 1, 8)), mx.zeros((1, 2, 1, 8)))
        mrope_positions = mx.array([[[7]], [[8]], [[9]]], dtype=mx.int32)
        multimodal.update_indexer(mx.zeros((1, 1, 8)), mrope_positions)

        batch = QSAKVCache.merge([text, multimodal])
        restored_text = batch.extract(0)
        restored_multimodal = batch.extract(1)
        mx.eval(
            restored_text.index_position_ids, restored_multimodal.index_position_ids
        )

        self.assertEqual(batch.index_position_ids.shape, (3, 2, 2))
        self.assertEqual(restored_text.index_position_ids.shape, (3, 1, 2))
        for axis in range(3):
            self.assertTrue(
                mx.array_equal(
                    restored_text.index_position_ids[axis], text_positions
                ).item()
            )
        self.assertTrue(
            mx.array_equal(
                restored_multimodal.index_position_ids, mrope_positions
            ).item()
        )

        extended = QSAKVCache.merge([text])
        extended.extend(QSAKVCache.merge([multimodal]))
        extended_text = extended.extract(0)
        extended_multimodal = extended.extract(1)
        mx.eval(
            extended_text.index_position_ids, extended_multimodal.index_position_ids
        )

        self.assertEqual(extended.index_position_ids.shape, (3, 2, 2))
        for axis in range(3):
            self.assertTrue(
                mx.array_equal(
                    extended_text.index_position_ids[axis], text_positions
                ).item()
            )
        self.assertTrue(
            mx.array_equal(
                extended_multimodal.index_position_ids, mrope_positions
            ).item()
        )

    def test_qsa_batch_cache_extends_empty_rows_without_duplicating_them(self):
        empty = BatchQSAKVCache([0, 0])
        filled = QSAKVCache()
        filled.update_and_fetch(mx.ones((1, 2, 1, 8)), mx.ones((1, 2, 1, 8)))
        filled.update_indexer(mx.ones((1, 1, 8)), mx.array([[0]], dtype=mx.int32))

        empty.extend(QSAKVCache.merge([filled]))

        self.assertEqual(empty.offset.shape, (3,))
        self.assertEqual(empty.index_keys.shape, (3, 1, 8))
        self.assertEqual(empty.index_position_ids.shape, (3, 1))
        self.assertEqual(empty.extract(0).offset, 0)
        self.assertEqual(empty.extract(2).offset, 1)

    def test_empty_qsa_batch_cache_state_round_trip(self):
        batch = BatchQSAKVCache([0, 2])
        state = batch.state
        restored = BatchQSAKVCache([0])
        restored.state = state

        self.assertIsNone(state[0][0])
        self.assertEqual(restored.left_padding.tolist(), [0, 2])
        self.assertEqual(restored.offset.tolist(), [0, -2])
        self.assertTrue(restored.empty())

    def test_generation_cache_factory_rejects_qsa_batch_quantization(self):
        model = qwen4_exp.Model(tiny_config())

        with self.assertRaisesRegex(
            NotImplementedError,
            "QSAKVCache does not support quantized continuous batching",
        ):
            _make_cache(model.language_model, [0], kv_bits=8)

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
            f"{mapped}.ple.ple_embedding.ngram_embedding.shards.0.weight", sanitized
        )
        self.assertEqual(sanitized[f"{mapped}.ple.conv1d.weight"].shape, (64, 3, 1))
        self.assertFalse(any(key.startswith("mtp.") for key in sanitized))

    def test_sanitize_restores_official_fp8_experts_and_ple(self):
        model = qwen4_exp.Model(tiny_config())
        weights = {}
        prefix = "model.language_model.layers.0.mlp"
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"{prefix}.experts.{expert}.{projection}.weight"
                weights[key] = mx.to_fp8(mx.ones((128, 128)) * (expert + 1))
                weights[f"{key}_scale_inv"] = mx.ones((1, 1))
        ple = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
        weights[f"{ple}.shard_0.weight"] = mx.to_fp8(mx.ones((4, 8)))
        weights[f"{ple}.weight_scale"] = mx.array([0.5], dtype=mx.bfloat16)

        sanitized = model.sanitize(weights)
        mapped = "language_model.model.layers.0"
        gate = sanitized[f"{mapped}.mlp.switch_mlp.gate_proj.weight"]
        up = sanitized[f"{mapped}.mlp.switch_mlp.up_proj.weight"]
        down = sanitized[f"{mapped}.mlp.switch_mlp.down_proj.weight"]
        ple_weight = sanitized[
            f"{mapped}.ple.ple_embedding.ngram_embedding.shards.0.weight"
        ]
        mx.eval(gate, up, down, ple_weight)
        self.assertEqual(gate.shape, (2, 128, 128))
        self.assertEqual(up.shape, (2, 128, 128))
        self.assertEqual(down.shape, (2, 128, 128))
        self.assertTrue(mx.allclose(ple_weight, mx.ones((4, 8)) * 0.5).item())
        self.assertFalse(any("scale" in key for key in sanitized))

    def test_quantization_uses_compatible_group_for_ple_shards(self):
        model = qwen4_exp.Model(tiny_config())
        predicate = model.quant_predicate
        path = (
            "language_model.model.layers.0.ple.ple_embedding."
            "ngram_embedding.shards.0"
        )
        self.assertEqual(predicate(path, None), {"fallback_group_size": 32})


if __name__ == "__main__":
    unittest.main()
