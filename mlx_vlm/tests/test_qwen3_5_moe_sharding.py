"""Two-process numerical checks for Qwen3.5/3.6 MoE tensor parallelism."""

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

from mlx_vlm.models.qwen3_5_moe.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel


def small_config(kv_heads=2):
    return TextConfig(
        model_type="qwen3_5_moe",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        full_attention_interval=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=128,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 1, 2],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )


def test_invalid_group_does_not_modify_weights():
    model = LanguageModel(small_config())
    before = dict(tree_flatten(model.parameters()))
    group = SimpleNamespace(size=lambda: 3, rank=lambda: 0)
    with pytest.raises(ValueError, match="linear_num_key_heads"):
        model.shard(group)
    after = dict(tree_flatten(model.parameters()))
    assert before.keys() == after.keys()
    assert all(after[key] is value for key, value in before.items())


def test_single_rank_is_a_noop():
    model = LanguageModel(small_config())
    before = dict(tree_flatten(model.parameters()))
    model.shard(SimpleNamespace(size=lambda: 1, rank=lambda: 0))
    after = dict(tree_flatten(model.parameters()))
    assert all(after[key] is value for key, value in before.items())


def test_quantization_group_must_fit_inside_shard():
    config = small_config()
    config.moe_intermediate_size = 64
    model = LanguageModel(config)
    nn.quantize(model, group_size=64, bits=4)
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="quantization groups"):
        model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    after = dict(tree_flatten(model.parameters()))
    assert all(after[key] is value for key, value in before.items())


def distributed_worker():
    group = mx.distributed.init(strict=True, backend="ring")
    assert group.size() == 2
    records = []
    for quantized, dtype, kv_heads in (
        (False, mx.float32, 2),
        (False, mx.float32, 1),
        (True, mx.float32, 2),
        (True, mx.bfloat16, 2),
    ):
        mx.random.seed(17)
        config = small_config(kv_heads)
        model_config = ModelConfig(
            text_config=config,
            vision_config=VisionConfig(deepstack_visual_indexes=[]),
            model_type="qwen3_5_moe",
        )
        reference = LanguageModel(copy.deepcopy(config), model_config)
        reference.update(tree_map(lambda p: p.astype(dtype), reference.parameters()))
        if quantized:
            nn.quantize(reference, group_size=32, bits=4)
        mx.eval(reference.parameters())
        sharded = LanguageModel(copy.deepcopy(config), model_config)
        if quantized:
            nn.quantize(sharded, group_size=32, bits=4)
        sharded.load_weights(tree_flatten(reference.parameters()))
        reference.eval()
        sharded.eval()
        sharded.shard(group)
        mx.eval(sharded.parameters())
        with pytest.raises(ValueError, match="already sharded"):
            sharded.shard(group)
        with pytest.raises(NotImplementedError, match="Speculative verification"):
            sharded(mx.array([[1]]), speculative_verify=True)
        reference_bytes = sum(p.nbytes for _, p in tree_flatten(reference.parameters()))
        shard_bytes = sum(p.nbytes for _, p in tree_flatten(sharded.parameters()))
        assert shard_bytes < reference_bytes

        reference_cache, sharded_cache = reference.make_cache(), sharded.make_cache()
        for step, tokens in enumerate(([1, 5, 9, 13, 2, 7], [11], [3], [8])):
            ids = mx.array([tokens])
            expected = reference(ids, cache=reference_cache).logits
            actual = sharded(ids, cache=sharded_cache).logits
            mx.eval(expected, actual)
            # BF16 rounds each partial projection before reduction, unlike the
            # full-width reference matmul. Allow eight BF16 epsilons for the
            # two-layer network; FP32 still checks the split to 2e-5.
            atol = 0.0625 if dtype == mx.bfloat16 else 2e-5
            records.append(
                {
                    "quantized": quantized,
                    "dtype": str(dtype),
                    "kv_heads": kv_heads,
                    "step": step,
                    "max_error": mx.max(mx.abs(actual - expected)).item(),
                    "passed": mx.allclose(
                        actual, expected, rtol=atol, atol=atol
                    ).item(),
                }
            )

    # All ranks reach the final collective even if a numerical comparison fails.
    passed = mx.distributed.all_sum(
        mx.array([int(all(r["passed"] for r in records))]), group=group
    ).item()
    print(
        json.dumps({"rank": group.rank(), "checks": records, "passed_ranks": passed}),
        flush=True,
    )
    assert passed == 2, records


@pytest.mark.skipif(
    not mx.distributed.is_available("ring"), reason="ring backend required"
)
def test_two_rank_prefill_and_decode_match_reference():
    repo = Path(__file__).resolve().parents[2]
    # Reserve two adjacent free ports for the launcher's ring listeners.
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
    env = dict(os.environ, PYTHONPATH=str(repo))
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
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"Distributed worker timed out:\n{output}")
    assert process.returncode == 0, output
    results = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert {r["rank"] for r in results if r["passed_ranks"] == 2} == {0, 1}, output


if __name__ == "__main__":
    if sys.argv[1:] != ["--distributed-worker"]:
        raise SystemExit("Run with pytest, or use --distributed-worker via mlx.launch")
    distributed_worker()
