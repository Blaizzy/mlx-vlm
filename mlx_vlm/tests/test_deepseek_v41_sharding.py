"""Released-weight loading and two-rank DeepSeek V4.1 inference checks."""

import copy
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.deepseek_v41 import Model, ModelConfig
from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight
from mlx_vlm.models.deepseek_v41.engram import NgramHashState
from mlx_vlm.models.deepseek_v41.language import LanguageModel, make_quantization_config
from mlx_vlm.tests.test_models import DATA
from mlx_vlm.utils import load_model


def small_config():
    values = copy.deepcopy(
        next(c["config"] for c in DATA["cases"] if c["module"] == "deepseek_v41")
    )
    values.update(
        hidden_size=64,
        moe_intermediate_size=128,
        q_lora_rank=64,
        head_dim=64,
        o_lora_rank=64,
    )
    return ModelConfig.from_dict(values)


def quantize_experts(model):
    def predicate(path, module):
        if hasattr(module, "to_quantized") and any(
            p in path
            for p in ("switch_mlp.", "shared_experts.", "attn.wq_b", "attn.wo_a")
        ):
            bits = 4 if "switch_mlp." in path else 8
            return dict(group_size=32, bits=bits, mode=f"mxfp{bits}")
        return False

    nn.quantize(model, class_predicate=predicate)


NATIVE_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


@pytest.mark.parametrize("bits", [4, 8])
def test_native_weight_repacking_preserves_bytes_and_block_scales(bits):
    rows, dims = 33, 64
    weight = mx.arange(rows * dims * bits // 8, dtype=mx.uint8).reshape(rows, -1)
    scale_rows = rows if bits == 4 else 2
    scales = mx.arange(scale_rows * 2, dtype=mx.uint8).reshape(scale_rows, 2)
    packed, expanded, mode = _pack_source_weight(weight, scales)
    assert packed.dtype == mx.uint32
    assert packed.shape == (rows, dims * bits // 32)
    assert mx.array_equal(packed.view(mx.uint8), weight).item()
    expected = scales if bits == 4 else mx.repeat(scales, 32, axis=0)[:rows]
    assert mx.array_equal(expanded, expected).item()
    assert mode == f"mxfp{bits}"


@pytest.mark.parametrize("nested_config", [False, True])
def test_native_checkpoint_keeps_packed_weights(tmp_path, nested_config):
    mx.random.seed(9)
    model = Model(small_config())
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    model.language_model.head.weight = mx.random.normal(
        model.language_model.head.weight.shape
    )
    quantization = make_quantization_config(model)
    nn.quantize(model, class_predicate=lambda p, m: quantization.get(p, False))
    mx.eval(model.parameters())
    expected = dict(tree_flatten(model.parameters()))
    weights = {}
    for name, value in expected.items():
        if (
            name.endswith(".scales")
            or name.endswith(".weight")
            and name[:-6] + "scales" in expected
        ):
            suffix = "scale" if name.endswith(".scales") else "weight"
            prefix = name.rsplit(".", 1)[0]
            if suffix == "weight":
                value = value.view(mx.uint8)
            if ".switch_mlp." in prefix:
                prefix, projection = prefix.rsplit(".", 1)
                prefix = prefix.replace(".switch_mlp", ".experts")
                projection = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[
                    projection
                ]
                for i in range(value.shape[0]):
                    weights[f"{prefix}.{i}.{projection}.{suffix}"] = value[i]
            else:
                if ".attn.wo_a" in prefix:
                    value = value.flatten(0, 1)
                weights[f"{prefix}.{suffix}"] = value
        else:
            weights[name] = value
    config = model.config.to_dict()
    if nested_config:
        config["text_config"] = {"quantization_config": NATIVE_QUANTIZATION}
    else:
        config["quantization_config"] = NATIVE_QUANTIZATION
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded = load_model(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    for name, value in expected.items():
        assert name in actual, name
        assert mx.array_equal(value, actual[name]).item(), name
    assert loaded.layers[0].ffn.switch_mlp.down_proj.mode == "mxfp4"
    assert loaded.layers[0].ffn.shared_experts.down_proj.mode == "mxfp8"
    assert loaded.layers[0].attn.wq_a.mode == "mxfp8"
    assert loaded.layers[1].engram.wkv.mode == "mxfp8"
    assert not hasattr(loaded, "_source_quantization")
    assert not hasattr(loaded, "_preserve_source_quantization")

    # The converter must describe the already-native modules when it writes an
    # MLX checkpoint, even if the requested default for other layers is affine4.
    from mlx_vlm.convert import _preserve_existing_deepseek_v4_quantization

    _preserve_existing_deepseek_v4_quantization(config, loaded, 64, 4, "affine")
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), actual)
    reloaded = dict(tree_flatten(load_model(tmp_path).parameters()))
    assert actual.keys() == reloaded.keys()
    for name, value in actual.items():
        assert mx.array_equal(value, reloaded[name]).item(), name


@pytest.mark.parametrize("bits", [4, 8])
def test_converted_checkpoint_keeps_declared_quantization(tmp_path, bits):
    model = Model(small_config())
    quantization = dict(group_size=64, bits=bits, mode="affine")
    nn.quantize(
        model,
        **quantization,
        class_predicate=lambda p, m: ".switch_mlp." in p and hasattr(m, "to_quantized"),
    )
    expected = dict(tree_flatten(model.parameters()))
    config = model.config.to_dict()
    config["quantization"] = quantization
    # Converted checkpoints can retain the source metadata; the explicit MLX
    # quantization config must win over the native checkpoint format.
    config["quantization_config"] = NATIVE_QUANTIZATION
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), expected)
    loaded = load_model(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert mx.array_equal(value, actual[name]).item(), name
    projection = loaded.layers[0].ffn.switch_mlp.down_proj
    assert (projection.mode, projection.bits, projection.group_size) == (
        "affine",
        bits,
        64,
    )


def test_invalid_shard_preserves_parameters():
    model = LanguageModel(small_config())
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="Expert count"):
        model.shard(SimpleNamespace(size=lambda: 3, rank=lambda: 0))
    after = dict(tree_flatten(model.parameters()))
    assert all(after[k] is v for k, v in before.items())


def distributed_worker():
    group = mx.distributed.init(strict=True, backend="ring")
    records = []
    for dtype in (mx.float32, mx.bfloat16):
        mx.random.seed(19)
        cfg = small_config()
        reference = LanguageModel(cfg)
        reference.head.weight = mx.random.normal(reference.head.weight.shape) * 0.05
        reference.update(tree_map(lambda p: p.astype(dtype), reference.parameters()))
        reference.head.weight = reference.head.weight.astype(mx.float32)
        quantize_experts(reference)
        reference.engram_hash = NgramHashState(
            cfg, reference.layout, token_map=[i % 7 for i in range(cfg.vocab_size)]
        )
        mx.eval(reference.parameters())
        sharded = LanguageModel(copy.deepcopy(cfg))
        quantize_experts(sharded)
        sharded.engram_hash = NgramHashState(
            cfg, sharded.layout, token_map=[i % 7 for i in range(cfg.vocab_size)]
        )
        sharded.load_weights(tree_flatten(reference.parameters()))
        sharded.shard(group)
        with pytest.raises(ValueError, match="already sharded"):
            sharded.shard(group)
        mx.eval(sharded.parameters())
        assert sharded.head.weight.size * 2 == reference.head.weight.size
        for layer, original in zip(sharded.layers, reference.layers):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assert (
                    getattr(layer.ffn.switch_mlp, projection).weight.size * 2
                    == getattr(original.ffn.switch_mlp, projection).weight.size
                )
        for batch in (1, 4):
            actual_cache, expected_cache = sharded.make_cache(), reference.make_cache()
            for step, tokens in enumerate(
                ([1, 7, 9, 3, 11, 15, 2, 6], [13], [17], [19])
            ):
                ids = mx.array(
                    [
                        [(t + row) % cfg.vocab_size for t in tokens]
                        for row in range(batch)
                    ]
                )
                expected = reference(ids, cache=expected_cache).logits
                actual = sharded(ids, cache=actual_cache).logits
                mx.eval(actual, expected)
                atol = 0.06 if dtype == mx.bfloat16 else 2e-4
                error = mx.max(mx.abs(actual - expected)).item()
                agrees = mx.distributed.all_gather(actual, group=group)
                mx.eval(agrees)
                ranks_agree = mx.array_equal(agrees[:batch], agrees[batch:]).item()
                records.append(
                    dict(
                        dtype=str(dtype),
                        batch=batch,
                        step=step,
                        max_error=error,
                        ranks_agree=ranks_agree,
                        passed=mx.allclose(
                            actual, expected, atol=atol, rtol=atol
                        ).item(),
                    )
                )
    passed = mx.distributed.all_sum(
        mx.array(int(all(r["passed"] and r["ranks_agree"] for r in records))),
        group=group,
    ).item()
    print(
        json.dumps(dict(rank=group.rank(), passed_ranks=passed, checks=records)),
        flush=True,
    )
    assert passed == 2, records


@pytest.mark.skipif(not mx.distributed.is_available("ring"), reason="ring required")
def test_two_rank_prefill_decode_and_batch():
    repo = Path(__file__).resolve().parents[2]
    while True:
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            break
    command = [
        sys.executable,
        "-c",
        "from mlx._distributed_utils.launch import main; main()",
        "--backend",
        "ring",
        "-n",
        "2",
        "--starting-port",
        str(port),
        "--env",
        f"PYTHONPATH={repo}",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--distributed-worker",
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(output)
    assert process.returncode == 0, output
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert {r["rank"] for r in records if r["passed_ranks"] == 2} == {0, 1}, output


if __name__ == "__main__":
    distributed_worker()
