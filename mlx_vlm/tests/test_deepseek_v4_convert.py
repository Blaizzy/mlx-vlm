import json
from pathlib import Path

import mlx.core as mx

import mlx_vlm.models.deepseek_v4.convert as convert_module
from mlx_vlm.models.deepseek_v4.convert import (
    configure_parser,
    convert_deepseek_v4_vision,
)


def _write_shard(path, weights):
    mx.save_safetensors(str(path), weights, metadata={"format": "pt"})
    return path.name


def _make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "compress_ratios": [0],
        "o_groups": 2,
        "o_lora_rank": 2,
        "vision_n_layers": 1,
        "vision_dim": 4,
        "vision_n_heads": 1,
        "vision_inter_dim": 8,
        "vision_patch_size": 2,
        "quantization_config": {"quant_method": "fp8"},
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("{}")

    shard1 = {
        "model.embed_tokens.weight": mx.zeros((32, 32), dtype=mx.bfloat16),
        "model.vision.patch_embed.proj.weight": mx.ones((4, 12), dtype=mx.bfloat16),
        "model.image_start": mx.ones((32,), dtype=mx.bfloat16),
    }
    shard2 = {
        "model.layers.0.self_attn.wq_a.weight": mx.zeros((128, 128), dtype=mx.uint8),
        "model.layers.0.self_attn.wq_a.weight_scale_inv": mx.ones(
            (1, 1), dtype=mx.uint8
        ),
    }
    projection_names = ("gate_proj", "down_proj", "up_proj")
    for expert, shard in ((0, shard1), (1, shard2)):
        for projection in projection_names:
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            shard[f"{prefix}.weight"] = mx.zeros((32, 16), dtype=mx.int8)
            shard[f"{prefix}.weight_scale_inv"] = mx.ones((32, 1), dtype=mx.uint8)
    mtp = {"mtp.layers.0.weight": mx.ones((2, 2), dtype=mx.bfloat16)}

    shard_names = [
        _write_shard(source / "model-00001-of-00003.safetensors", shard1),
        _write_shard(source / "model-00002-of-00003.safetensors", shard2),
        _write_shard(source / "model-00003-of-00003.safetensors", mtp),
    ]
    weight_map = {}
    for shard_name, weights in zip(shard_names, (shard1, shard2, mtp)):
        weight_map.update({key: shard_name for key in weights})
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123456},
                "weight_map": weight_map,
            }
        )
    )
    return source


def test_streaming_converter_maps_and_packs_mixed_checkpoint(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "output"

    convert_deepseek_v4_vision(source, output)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["source_total_size"] == 123456
    assert len(set(index["weight_map"].values())) == 1
    assert not any(key.startswith("mtp.") for key in index["weight_map"])
    weights = mx.load(str(next(output.glob("model-*.safetensors"))))

    assert "vision.patch_embed.proj.weight" in weights
    assert weights["vision.patch_embed.proj.weight"].dtype == mx.bfloat16
    assert "language_model.model.embed_tokens.weight" in weights
    expert_prefix = "language_model.model.layers.0.ffn.switch_mlp"
    for projection in ("gate_proj", "down_proj", "up_proj"):
        assert weights[f"{expert_prefix}.{projection}.weight"].shape == (2, 32, 4)
        assert weights[f"{expert_prefix}.{projection}.scales"].shape == (2, 32, 1)
    attention = "language_model.model.layers.0.attn.wq_a"
    assert weights[f"{attention}.weight"].shape == (128, 32)
    assert weights[f"{attention}.scales"].shape == (128, 4)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"][f"{expert_prefix}.gate_proj"] == {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }
    assert config["quantization"][attention] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert (output / "tokenizer.json").exists()


def test_streaming_converter_refuses_nonempty_output(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep")

    try:
        convert_deepseek_v4_vision(source, output)
    except ValueError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("Expected non-empty output to be rejected")


def test_dedicated_converter_resolves_source_and_extracts_dspark(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    output = tmp_path / "output"
    drafter = tmp_path / "drafter"
    calls = {}

    monkeypatch.setattr(
        convert_module,
        "get_model_path",
        lambda model, revision=None: calls.update(model=model, revision=revision)
        or source,
    )

    def fake_convert(model_path, output_path):
        calls.update(model_path=model_path, output_path=Path(output_path))
        Path(output_path).mkdir()
        return Path(output_path)

    class Splitter:
        def split(self, model_path, output_path):
            calls.update(dspark_source=model_path, dspark_output=output_path)

    from mlx_vlm.speculative.drafters import mtp_split

    monkeypatch.setattr(convert_module, "convert_deepseek_v4_vision", fake_convert)
    monkeypatch.setattr(mtp_split, "detect_mtp_splitter", lambda path: Splitter())
    monkeypatch.setattr(
        convert_module,
        "create_model_card",
        lambda path, source_id: calls.update(card=(path, source_id)),
    )
    monkeypatch.setattr(
        convert_module,
        "upload_to_hub",
        lambda path, repo: calls.update(upload=(path, repo)),
    )

    result = convert_module.convert(
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        output,
        revision="test-revision",
        mtp=True,
        mtp_output=drafter,
        upload_repo="mlx-community/test-model",
    )

    assert result == output
    assert calls["model"] == "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    assert calls["revision"] == "test-revision"
    assert calls["model_path"] == source
    assert calls["output_path"] == output
    assert calls["dspark_source"] == str(source)
    assert calls["dspark_output"] == str(drafter)
    assert calls["card"] == (
        output,
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    )
    assert calls["upload"] == (output, "mlx-community/test-model")


def test_dedicated_converter_parser_uses_model_specific_options():
    args = configure_parser().parse_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            "--mlx-path",
            "converted",
            "--mtp",
        ]
    )

    assert args.model == "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    assert args.output_path == "converted"
    assert args.mtp is True
