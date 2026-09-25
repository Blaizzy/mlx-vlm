import json
import struct
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.convert import convert
from mlx_vlm.models.deepseek_v41 import Model, ModelConfig
from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight
from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding
from mlx_vlm.utils import load_model


@pytest.fixture(autouse=True)
def _seed():
    mx.random.seed(0)


def _write_source(path, weights, dtypes):
    header, buffers, offset = {}, [], 0
    for key, value in weights.items():
        data = np.asarray(value.view(mx.uint8)).tobytes()
        dtype = {mx.float32: "F32", mx.bfloat16: "BF16", mx.uint8: "U8"}[value.dtype]
        header[key] = {
            "dtype": dtypes.get(key, dtype),
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        buffers.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(buffers))


def test_nested_config_preserves_text_and_vision_settings():
    config = ModelConfig.from_dict(
        {
            "model_type": "deepseek_v41",
            "text_config": {
                "model_type": "deepseek_v41_text",
                "hidden_size": 64,
                "rope_scaling": {"factor": 16},
            },
            "vision_config": {
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "hidden_size": 32,
                "max_wh_ratio": 5,
            },
        }
    )
    assert config.model_type == "deepseek_v41"
    assert config.hidden_size == 64
    assert config.rope_scaling == {"factor": 16}
    assert (config.vision_num_layers, config.vision_num_heads) == (3, 4)
    assert config.vision_hidden_size == 32
    assert config.vision_max_wh_ratio == 5
    assert ModelConfig.from_dict(asdict(config)) == config


def _processor(config):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from mlx_vlm.models.deepseek_v41.processing_deepseek_v41 import DeepseekV41Processor

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    return DeepseekV41Processor(tokenizer, config=config)


def test_processor_saves_model_config(tmp_path):
    config = ModelConfig(rope_scaling={"factor": 16}, vision_max_wh_ratio=5)
    processor = _processor(config)
    processor.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "processor_config.json").read_text())
    assert saved["config"] == asdict(config)
    assert (tmp_path / "tokenizer.json").is_file()


@pytest.mark.parametrize("rowwise", [False, True])
def test_fp8_scale_layouts_decode_exactly(rowwise):
    raw = mx.full((64, 64), 56, dtype=mx.uint8)  # E4M3 encoding of 1.0.
    scale_rows = 64 if rowwise else 2
    scales = (mx.arange(scale_rows * 2).reshape(scale_rows, 2) % 4 + 125).astype(
        mx.uint8
    )
    packed, expanded, mode = _pack_source_weight(raw, scales)
    decoded = mx.dequantize(packed, expanded, group_size=32, bits=8, mode=mode)
    expected = mx.power(2.0, scales.astype(mx.float32) - 127)
    expected = mx.repeat(expected, 32, axis=-1)
    if not rowwise:
        expected = mx.repeat(expected, 32, axis=0)
    assert mx.array_equal(decoded, expected)


def test_native_mixed_checkpoint_conversion(tmp_path):
    config = ModelConfig(
        hidden_size=64,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=64,
        qk_rope_head_dim=8,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        moe_intermediate_size=64,
        n_routed_experts=2,
        num_experts_per_tok=1,
        compress_ratios=[0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        num_nextn_predict_layers=0,
        engram_layer_ids=[0],
        engram_num_embeddings=[17],
        engram_head_dim=256,
        engram_n_heads=1,
        engram_max_ngram_size=2,
        engram_vocab_size=16,
        vision_num_layers=1,
        vision_hidden_size=64,
        vision_num_heads=4,
        vision_intermediate_size=64,
        vision_patch_size=2,
    )
    source = tmp_path / "source"
    source.mkdir()
    weights, dtypes = {}, {}
    for key, value in tree_flatten(Model(config).parameters()):
        key = key.removeprefix("language_model.")
        key = key.replace("embed_tokens.", "embed.")
        if ".switch_mlp." in key:
            projection = key.split(".switch_mlp.")[1].split(".")[0]
            src = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[projection]
            prefix = key.split(".switch_mlp.")[0]
            for expert in range(2):
                name = f"{prefix}.experts.{expert}.{src}"
                packed, scales = mx.quantize(
                    value[expert], group_size=32, bits=4, mode="mxfp4"
                )
                weights[name + ".weight"] = packed.view(mx.uint8)
                weights[name + ".scale"] = scales
                dtypes[name + ".weight"] = "I8"
                dtypes[name + ".scale"] = "F8_E8M0"
        elif key.endswith(("attn.wq_a.weight", "engram.embed.weight")):
            rows, cols = value.shape
            name = key[: -len(".weight")]
            weights[key] = mx.full((rows, cols), 56, dtype=mx.uint8)
            scale_rows = rows if ".engram." in key else rows // 32
            weights[name + ".scale"] = mx.full(
                (scale_rows, cols // 32), 124, dtype=mx.uint8
            )
            dtypes[key], dtypes[name + ".scale"] = "F8_E4M3", "F8_E8M0"
        else:
            weights[key] = value
    shard = source / "model.safetensors"
    _write_source(shard, weights, dtypes)
    original = shard.read_bytes()
    raw_config = asdict(config)
    raw_config["quantization_config"] = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }
    (source / "config.json").write_text(json.dumps(raw_config))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard.name for key in weights}})
    )
    _processor(config).save_pretrained(source)

    native = load_model(source, lazy=True)
    layer = native.language_model.layers[0]
    assert hasattr(layer.ffn.switch_mlp.gate_proj, "to_quantized")
    assert isinstance(layer.attn.wq_a, nn.Linear)
    assert isinstance(layer.engram.embed, QuantizedEngramEmbedding)
    assert layer.engram.embed.mode == "mxfp8"
    assert mx.array_equal(
        layer.engram.embed(mx.array([0, 16])), mx.full((2, 256), 0.125)
    )
    assert shard.read_bytes() == original

    output = tmp_path / "converted"
    convert(str(source), str(output), quantize=True, q_group_size=64, q_bits=4)
    reloaded = load_model(output, lazy=True, strict=True)
    layer = reloaded.language_model.layers[0]
    assert layer.ffn.switch_mlp.gate_proj.bits == 4
    assert layer.attn.wq_a.bits == 4
    assert layer.engram.embed.bits == 4
    assert layer.ffn.switch_mlp.gate_proj.mode == "affine"
    assert mx.array_equal(
        layer.engram.embed(mx.array([0, 16])), mx.full((2, 256), 0.125)
    )
    for name in ("vision", "aligner"):
        before = dict(tree_flatten(getattr(native, name).parameters()))
        after = dict(tree_flatten(getattr(reloaded, name).parameters()))
        assert before.keys() == after.keys()
        assert all(mx.array_equal(before[key], after[key]) for key in before)
        assert all(mx.issubdtype(value.dtype, mx.floating) for value in after.values())

    patches = mx.random.normal((9, 3 * config.vision_patch_size**2))
    assert mx.array_equal(
        native.encode_image(patches, 3, 3), reloaded.encode_image(patches, 3, 3)
    )
    assert shard.read_bytes() == original


def test_engram_chunked_requantization(monkeypatch):
    monkeypatch.setattr(QuantizedEngramEmbedding, "_quantize_chunk_rows", 3)
    source = QuantizedEngramEmbedding(
        7, 256, group_size=32, bits=8, mode="mxfp8", scale_dtype=mx.uint8
    )
    source.weight, source.scales = mx.quantize(
        mx.random.normal((7, 256)).astype(mx.bfloat16),
        group_size=32,
        bits=8,
        mode="mxfp8",
    )
    expected = mx.quantize(
        mx.dequantize(
            source.weight, source.scales, group_size=32, bits=8, mode="mxfp8"
        ),
        group_size=64,
        bits=4,
    )
    converted = source.to_quantized(group_size=64, bits=4)
    for actual, reference in zip(
        (converted.weight, converted.scales, converted.biases), expected
    ):
        assert mx.array_equal(actual, reference)
    assert converted.to_quantized(group_size=64, bits=4) is converted
