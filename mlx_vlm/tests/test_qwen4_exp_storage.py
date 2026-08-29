import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models import qwen4_exp
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpNGramEmbedding
from mlx_vlm.models.qwen4_exp.ple_storage import (
    QuantizedMMapNGramEmbedding,
    build_quantized_ple_manifest,
    materialize_interleaved_ple_store,
    prepare_external_ple_model,
)


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
        for name, dtype in (
            ("weight", "U32"),
            ("scales", "BF16"),
            ("biases", "BF16"),
        ):
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
        "shards": [
            {
                "row_start": 0,
                "row_count": table.shape[0],
                **tensors,
            }
        ],
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


def test_quantized_mmap_lookup_matches_resident_dequantization(tmp_path):
    table = mx.arange(64 * 160).reshape(64, 160).astype(mx.float32) / 100
    path, expected = _write_quantized_store(tmp_path, table, cache_rows=4)
    store = QuantizedMMapNGramEmbedding(path)
    ids = np.array([[1, 7, 1], [63, 7, 2]])
    actual = store(ids)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32))[ids],
    )
    assert store.stats.rows == 6
    assert store.stats.cache_hits == 2
    assert store.stats.cache_misses == 4
    assert store.stats.bytes_read == 4 * (20 * 4 + 5 * 2 + 5 * 2)


def test_quantized_mmap_lookup_preserves_unsorted_unique_ids(tmp_path):
    table = mx.arange(8 * 160).reshape(8, 160).astype(mx.float32) / 100
    path, expected = _write_quantized_store(tmp_path, table)
    store = QuantizedMMapNGramEmbedding(path)
    ids = np.array([7, 1, 5, 0])
    actual = store(ids)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32))[ids],
    )


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
        np.asarray(batch_cache[3]),
        np.asarray(continuation[:, -external.context_len :]),
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
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )


def test_prepare_external_model_supports_nvfp4(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    table = mx.arange(8 * 160).reshape(8, 160).astype(mx.float32) / 100
    weight, scales = mx.quantize(table, group_size=16, bits=4, mode="nvfp4")
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.scales": scales,
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
                    "group_size": 16,
                    "mode": "nvfp4",
                },
            }
        )
    )

    prepare_external_ple_model(source, target)
    range_manifest = json.loads((target / "ple-store.json").read_text())
    assert range_manifest["source_root"] == "../source"
    assert range_manifest["quantization"] == {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
    }
    materialize_interleaved_ple_store(source, target / "ple-store.json")
    assert (target / "ple-q4.rows").stat().st_size == 8 * 90
    store = QuantizedMMapNGramEmbedding(target / "ple-store.json")
    actual = store(np.array([0, 7]))
    expected = mx.dequantize(weight, scales, group_size=16, bits=4, mode="nvfp4")[
        [0, 7]
    ]
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
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


def test_materialize_does_not_publish_partial_store(tmp_path, monkeypatch):
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
    original_memmap = np.memmap

    def fail_output_mapping(*args, **kwargs):
        if kwargs.get("mode") == "r+":
            raise OSError("simulated write failure")
        return original_memmap(*args, **kwargs)

    monkeypatch.setattr(np, "memmap", fail_output_mapping)

    with pytest.raises(OSError, match="simulated write failure"):
        materialize_interleaved_ple_store(source, manifest_path)

    assert manifest_path.read_text() == original_manifest
    assert not (target / "ple-q4.rows").exists()
    assert list(target.glob(".ple-q4.rows.*.partial")) == []
