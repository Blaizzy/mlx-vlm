"""Qwen4 experimental models, PLE storage, and MTP drafting."""

import json
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import qwen4_exp
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    LanguageModel,
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpNGramEmbedding,
    _create_qwen4_exp_attention_mask,
)
from mlx_vlm.models.qwen4_exp.ple_storage import (
    QuantizedMMapNGramEmbedding,
    build_quantized_ple_manifest,
    materialize_interleaved_ple_store,
    prepare_external_ple_model,
)
from mlx_vlm.models.qwen4_exp.qsa_kernel import (
    QSAExecutionPlan,
    select_qsa_execution_plan,
)
from mlx_vlm.prompt_utils import MessageFormat, MessageFormatter
from mlx_vlm.speculative.common import _dflash_block_total
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter, get_mtp_splitter
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import split_qwen4_exp_mtp
from mlx_vlm.speculative.mtp import _mtp_next_block_size

# Language and vision models


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


# External PLE storage


def _write_quantized_store(tmp_path, table, *, cache_rows=0):
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    scales = scales.astype(mx.bfloat16)
    biases = biases.astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    arrays = {
        "weight": np.asarray(weight),
        "scales": np.asarray(scales.view(mx.uint16)),
        "biases": np.asarray(biases.view(mx.uint16)),
    }
    offset = 0
    tensors = {}
    data_path = tmp_path / "rows.bin"
    with data_path.open("wb") as stream:
        for name, dtype in (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")):
            values = arrays[name]
            stream.write(values.tobytes())
            tensors[name] = {
                "file": data_path.name,
                "offset": offset,
                "dtype": dtype,
                "shape": list(values.shape),
            }
            offset += values.nbytes
    manifest = {
        "version": 2,
        "source_root": str(tmp_path),
        "row_width": table.shape[1],
        "row_count": table.shape[0],
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "cache_rows": cache_rows,
        "shards": [{"row_start": 0, "row_count": table.shape[0], **tensors}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    expected = mx.dequantize(weight, scales, biases, group_size=32, bits=4)
    return path, expected


def _write_nvfp4_store(tmp_path, table, *, cache_rows=0):
    weight, scales = mx.quantize(table, group_size=16, bits=4, mode="nvfp4")
    mx.eval(weight, scales)
    arrays = {"weight": np.asarray(weight), "scales": np.asarray(scales)}
    offset = 0
    tensors = {}
    data_path = tmp_path / "rows.bin"
    with data_path.open("wb") as stream:
        for name, dtype in (("weight", "U32"), ("scales", "U8")):
            values = arrays[name]
            stream.write(values.tobytes())
            tensors[name] = {
                "file": data_path.name,
                "offset": offset,
                "dtype": dtype,
                "shape": list(values.shape),
            }
            offset += values.nbytes
    manifest = {
        "version": 2,
        "source_root": ".",
        "row_width": table.shape[1],
        "row_count": table.shape[0],
        "quantization": {"bits": 4, "group_size": 16, "mode": "nvfp4"},
        "cache_rows": cache_rows,
        "shards": [{"row_start": 0, "row_count": table.shape[0], **tensors}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    expected = mx.dequantize(weight, scales, group_size=16, bits=4, mode="nvfp4")
    return path, expected


def test_nvfp4_mmap_lookup_matches_resident_dequantization(tmp_path):
    table = mx.arange(16 * 160).reshape(16, 160).astype(mx.float32) / 100
    path, expected = _write_nvfp4_store(tmp_path, table, cache_rows=4)
    store = QuantizedMMapNGramEmbedding(path)
    ids = np.array([[1, 7, 1], [15, 7, 2]])
    actual = store(ids)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32))[ids],
    )
    assert store.stats.bytes_read == 4 * (20 * 4 + 10)


def test_external_ple_preserves_batched_incremental_history(tmp_path):
    config = qwen4_exp.TextConfig(
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
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=128,
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
    )
    resident = Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, 0)
    row_count = resident.ngram_embedding.shard_offsets[-1]
    row_width = config.ple_embed_dim // resident.ngram_heads
    table = mx.arange(row_count * row_width, dtype=mx.float32).reshape(
        row_count, row_width
    )
    manifest, _ = _write_quantized_store(tmp_path, table)
    config.ple_storage = {"manifest": str(manifest)}
    external = Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, 0)

    prompts = mx.array([[2, 3, 4, 5], [1, 6, 7, 8]], dtype=mx.int32)
    continuation = mx.array([[9, 10], [11, 12]], dtype=mx.int32)
    batch_cache = ArraysCache(size=4)
    external(prompts, cache=batch_cache)
    batch_output = external(continuation, cache=batch_cache)

    row_outputs = []
    for row in range(prompts.shape[0]):
        row_cache = ArraysCache(size=4)
        external(prompts[row : row + 1], cache=row_cache)
        row_outputs.append(external(continuation[row : row + 1], cache=row_cache))
    expected = mx.concatenate(row_outputs, axis=0)
    mx.eval(batch_output, expected)

    np.testing.assert_array_equal(
        np.asarray(batch_output.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(batch_cache[3]), np.asarray(continuation[:, -external.context_len :])
    )


def test_quantized_store_fails_closed_for_truncation_and_escape(tmp_path):
    table = mx.ones((4, 160))
    path, _ = _write_quantized_store(tmp_path, table)
    (tmp_path / "rows.bin").write_bytes(b"short")
    with pytest.raises(ValueError, match="byte range exceeds"):
        QuantizedMMapNGramEmbedding(path)

    manifest = json.loads(path.read_text())
    manifest["shards"][0]["weight"]["file"] = "../rows.bin"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must be filenames"):
        QuantizedMMapNGramEmbedding(path)


def test_prepare_external_model_indexes_existing_q4_ranges(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    table = mx.arange(8 * 160).reshape(8, 160).astype(mx.float32) / 100
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    scales = scales.astype(mx.bfloat16)
    biases = biases.astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.scales": scales,
        f"{prefix}.biases": biases,
        "language_model.embed_tokens.weight": mx.ones((2, 2)),
    }
    file_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "text_config": {},
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    prefix: {"bits": 4, "group_size": 32, "mode": "affine"},
                },
            }
        )
    )
    (source / "tokenizer.json").write_text("{}")

    prepare_external_ple_model(source, target)
    range_manifest = json.loads((target / "ple-store.json").read_text())
    assert range_manifest["source_root"] == "../source"
    target_config = json.loads((target / "config.json").read_text())
    assert target_config["text_config"]["ple_storage"]["manifest"] == "ple-store.json"
    materialize_interleaved_ple_store(source, target / "ple-store.json")
    interleaved_manifest = json.loads((target / "ple-store.json").read_text())
    assert interleaved_manifest["source_root"] == "../source"

    assert (target / file_name).stat().st_ino == (source / file_name).stat().st_ino
    target_index = json.loads((target / "model.safetensors.index.json").read_text())
    assert list(target_index["weight_map"]) == ["language_model.embed_tokens.weight"]
    assert (target / "ple-q4.rows").stat().st_size == 8 * 100
    store = QuantizedMMapNGramEmbedding(target / "ple-store.json")
    actual = store(np.array([0, 7]))
    expected = mx.dequantize(weight, scales, biases, group_size=32, bits=4)[[0, 7]]
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )


def test_build_manifest_rejects_non_q4_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": mx.ones((8, 20), dtype=mx.uint32),
        f"{prefix}.scales": mx.ones((8, 5), dtype=mx.bfloat16),
        f"{prefix}.biases": mx.ones((8, 5), dtype=mx.bfloat16),
    }
    file_name = "model.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {"quantization": {prefix: {"bits": 8, "group_size": 32, "mode": "affine"}}}
        )
    )
    manifest_path = tmp_path / "output" / "ple-store.json"

    with pytest.raises(ValueError, match="unsupported PLE quantization"):
        build_quantized_ple_manifest(source, manifest_path)

    assert not manifest_path.exists()


def test_materialize_checks_existing_data_before_rewriting_manifest(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    table = mx.ones((8, 160))
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.scales": scales.astype(mx.bfloat16),
        f"{prefix}.biases": biases.astype(mx.bfloat16),
        "language_model.embed_tokens.weight": mx.ones((2, 2)),
    }
    file_name = "model.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "text_config": {},
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    prefix: {"bits": 4, "group_size": 32, "mode": "affine"},
                },
            }
        )
    )
    prepare_external_ple_model(source, target)
    manifest_path = target / "ple-store.json"
    original_manifest = manifest_path.read_text()
    (target / "ple-q4.rows").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="row store already exists"):
        materialize_interleaved_ple_store(source, manifest_path)

    assert manifest_path.read_text() == original_manifest


# MTP drafting


def _tiny_text_config(with_ple=False):
    config = {
        "model_type": "qwen4_exp_text",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 16,
        "moe_intermediate_size": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 64,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "hc_count": 2,
        "hc_lowrank": 8,
        "head_dim": 16,
        "layer_types": ["linear_attention", "full_attention"],
        "ple_layer_ids": [1] if with_ple else [],
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [1, 1, 0],
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.25,
        },
        "mtp_num_hidden_layers": 1,
    }
    if with_ple:
        config.update(
            {
                "ple_embed_dim": 32,
                "ple_conv_kernel_size": 3,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 17,
                "make_ngram_vocab_size_divisible_by": 4,
                "split_ngram_parts": 4,
                "eos_token_id": 1,
            }
        )
    return TextConfig.from_dict(config)


def _outer_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=59,
    )


def test_qwen4_mtp_uses_shared_adaptive_policy_with_three_draft_ceiling():
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=_tiny_text_config()))

    assert _dflash_block_total(drafter, None) == 4
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2
    drafter.accept_lens.extend([1] * 8)
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 4
    assert _dflash_block_total(drafter, 3) == 3

    drafter.accept_lens[:] = [1] * 5 + [0] * 3
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2


def test_qwen4_mtp_draft_block_uses_hyper_connection_hidden():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=nn.Linear(32, 64, bias=False),
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    tokens = drafter.draft_block(
        7,
        mx.zeros((1, 1, 64)),
        None,
        2,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    mx.eval(tokens)

    assert tokens.shape == (1, 1)
    assert drafter._cache[0].offset == 1


@pytest.mark.parametrize("accepted", range(6))
def test_qwen4_target_exposes_pre_mixer_hidden_and_rolls_back_rejection_exactly(
    accepted,
):
    config = _tiny_text_config(with_ple=True)
    language = LanguageModel(config, _outer_config())
    prompt = mx.arange(1, 17, dtype=mx.int32)[None]
    verify = mx.array([[17, 18, 19, 20, 21, 22]], dtype=mx.int32)

    speculative_cache = language.make_cache()
    prefill = language(prompt, cache=speculative_cache, return_hidden=True)
    hidden, _, rollback = language.speculative_verify_hidden(verify, speculative_cache)
    language.rollback_speculative_cache(
        speculative_cache, rollback, accepted=accepted, block_size=6
    )

    reference_cache = language.make_cache()
    language(prompt, cache=reference_cache)
    for index in range(accepted + 1):
        language(verify[:, index : index + 1], cache=reference_cache)
    probe = mx.array([[7]], dtype=mx.int32)
    speculative_logits = language(probe, cache=speculative_cache).logits
    reference_logits = language(probe, cache=reference_cache).logits
    mx.eval(prefill.hidden_states, hidden, speculative_logits, reference_logits)

    assert prefill.hidden_states[-1].shape == (1, 16, 64)
    assert hidden.shape == (1, 6, 64)
    assert mx.allclose(speculative_logits, reference_logits, rtol=0, atol=1e-6).item()
    assert mx.array_equal(
        mx.argmax(speculative_logits, axis=-1), mx.argmax(reference_logits, axis=-1)
    ).item()


def test_qwen4_batched_qsa_rollback_restores_exact_offsets():
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5]], dtype=mx.int32)

    speculative_cache = _make_cache(language, [0])
    language(prompt, cache=speculative_cache)
    _, _, rollback = language.speculative_verify_hidden(verify, speculative_cache)
    language.rollback_speculative_cache(
        speculative_cache, rollback, accepted=[0], block_size=2
    )

    reference_cache = _make_cache(language, [0])
    language(prompt, cache=reference_cache)
    language(verify[:, :1], cache=reference_cache)

    speculative_offsets = [
        entry.offset for entry in speculative_cache if hasattr(entry, "offset")
    ]
    reference_offsets = [
        entry.offset for entry in reference_cache if hasattr(entry, "offset")
    ]
    mx.eval(speculative_offsets, reference_offsets)

    assert all(
        mx.array_equal(actual, expected).item()
        for actual, expected in zip(speculative_offsets, reference_offsets)
    )


def test_qwen4_speculative_verifier_matches_tokenwise_hidden_and_logits():
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5, 6]], dtype=mx.int32)

    batched_cache = language.make_cache()
    language(prompt, cache=batched_cache)
    batched_hidden, _, _, batched_logits = language.speculative_verify_logits(
        verify, batched_cache, lambda logits: logits
    )

    tokenwise_cache = language.make_cache()
    language(prompt, cache=tokenwise_cache)
    tokenwise_hidden = []
    tokenwise_logits = []
    for index in range(verify.shape[1]):
        output = language(
            verify[:, index : index + 1], cache=tokenwise_cache, return_hidden=True
        )
        tokenwise_hidden.append(output.hidden_states[-1])
        tokenwise_logits.append(output.logits)
    tokenwise_hidden = mx.concatenate(tokenwise_hidden, axis=1)
    tokenwise_logits = mx.concatenate(tokenwise_logits, axis=1)
    mx.eval(batched_hidden, batched_logits, tokenwise_hidden, tokenwise_logits)

    assert mx.allclose(batched_hidden, tokenwise_hidden, rtol=0, atol=1e-6).item()
    assert mx.allclose(batched_logits, tokenwise_logits, rtol=0, atol=1e-6).item()
    assert mx.array_equal(
        mx.argmax(batched_logits, axis=-1), mx.argmax(tokenwise_logits, axis=-1)
    ).item()


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("prefix_length", [512, 2050, 2051, 2052])
@pytest.mark.parametrize("block_size", [2, 4])
def test_qwen4_qsa_verifier_matches_decode_across_sparse_boundary(
    batch, prefix_length, block_size
):
    from mlx_vlm.models.qwen4_exp.language import (
        BatchQSAKVCache,
        Qwen4ExpAttention,
        Qwen4ExpBatchInvariantForward,
    )

    mx.random.seed(2127)
    config = _tiny_text_config()
    config.hidden_size = 512
    config.num_attention_heads = 4
    config.head_dim = 128
    config.indexer_head_dim = 32
    config.indexer_budget = 2048
    attention = Qwen4ExpAttention(config)
    attention.set_dtype(mx.bfloat16)
    nn.quantize(attention, group_size=32, bits=4)
    verifier = Qwen4ExpBatchInvariantForward()
    caches = [BatchQSAKVCache([0] * batch) for _ in range(2)]
    hidden = mx.random.normal(
        (batch, prefix_length + block_size, config.hidden_size)
    ).astype(mx.bfloat16)
    for cache in caches:
        mx.eval(attention(hidden[:, :prefix_length], cache=cache, mask="causal"))

    # Budget 2048 and compression 4 first select sparse attention at length 2052.
    # Test fully dense blocks, blocks spanning the transition, and sparse blocks.
    proposal = hidden[:, prefix_length:]
    expected = mx.concatenate(
        [
            verifier._qsa_attention(
                attention, proposal[:, index : index + 1], caches[0], None, None
            )
            for index in range(block_size)
        ],
        axis=1,
    )
    actual = verifier._qsa_attention(attention, proposal, caches[1], None, "causal")
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()
    assert caches[0].index_offset == caches[1].index_offset
    assert mx.array_equal(caches[0].index_keys, caches[1].index_keys).item()


def test_qwen4_fused_greedy_mixes_captured_hyper_state_before_lm_head(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as qwen4_language

    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    verifier = qwen4_language._QWEN4_EXACT_SPECULATIVE_VERIFIER
    monkeypatch.setattr(verifier, "can_quantized_head", lambda linear: True)
    monkeypatch.setattr(
        verifier,
        "quantized_argmax",
        lambda linear, hidden, token_mask=None: mx.argmax(linear(hidden), axis=-1),
    )
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = mx.argmax(language(inputs, cache=language.make_cache()).logits, axis=-1)
    language._position_ids = None
    language._rope_deltas = None
    actual = language.fused_greedy_decode(inputs, cache=language.make_cache())
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_splitter_maps_fused_experts_and_quantizes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": text_config})
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((64,)),
            "mtp.fc_hidden.weight": mx.ones((32, 32)),
            "mtp.layers.0.mlp.experts.gate_up_proj": mx.ones((4, 32, 32)),
            "mtp.layers.0.mlp.experts.down_proj": mx.ones((4, 32, 16)),
            "mtp.layers.0.mlp.gate.weight": mx.ones((4, 32)),
        },
    )

    splitter = detect_mtp_splitter(source)
    assert splitter is not None
    assert splitter.output_model_type == "qwen4_exp_mtp"
    assert get_mtp_splitter("qwen4_exp").output_model_type == "qwen4_exp_mtp"

    split_qwen4_exp_mtp(str(source), str(output), q_bits=3, q_group_size=32)
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())

    assert "layers.0.mlp.switch_mlp.gate_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.up_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.down_proj.weight" in weights
    assert "fc_hidden.scales" in weights
    assert "layers.0.mlp.gate.scales" not in weights
    assert config["model_type"] == "qwen4_exp_mtp"
    assert config["block_size"] == 2
    assert config["quantization"] == {"group_size": 32, "bits": 3, "mode": "affine"}


def test_qwen4_mtp_splitter_converts_official_fp8_experts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": text_config,
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    weights = {"mtp.pre_fc_norm_hidden.weight": mx.zeros((64,))}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
            weights[key] = mx.to_fp8(mx.ones((128, 128)) * (expert + 1))
            weights[f"{key}_scale_inv"] = mx.ones((1, 1))
    mx.save_safetensors(str(source / "model.safetensors"), weights)

    split_qwen4_exp_mtp(str(source), str(output))

    split_weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    gate = split_weights["layers.0.mlp.switch_mlp.gate_proj.weight"]
    up = split_weights["layers.0.mlp.switch_mlp.up_proj.weight"]
    down = split_weights["layers.0.mlp.switch_mlp.down_proj.weight"]
    mx.eval(gate, up, down)
    assert gate.shape == (2, 128, 32)
    assert up.shape == (2, 128, 32)
    assert down.shape == (2, 128, 32)
    assert gate.dtype == mx.uint32
    assert "layers.0.mlp.switch_mlp.gate_proj.scales" in split_weights
    assert "layers.0.mlp.switch_mlp.up_proj.scales" in split_weights
    assert "layers.0.mlp.switch_mlp.down_proj.scales" in split_weights
    assert not any(key.endswith("weight_scale_inv") for key in split_weights)
    assert config["quantization"] == {"group_size": 32, "bits": 8, "mode": "mxfp8"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
