"""MoE checkpoint repacking, expert offload, and output parity."""

import dataclasses
import json
import shutil
import threading
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models import deepseek_v3, laguna, minimax
from mlx_vlm.models.laguna.language import LagunaPackedSwitchGLU
from mlx_vlm.moe_offload import ExpertStore, patch_model, plan, repack
from mlx_vlm.utils import load_model, save_weights


def _deepseek_config():
    return deepseek_v3.ModelConfig(
        model_type="deepseek_v3",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        attention_bias=False,
    )


def _build_and_repack(root, model=None, config=None):
    if model is None:
        config = _deepseek_config()
        model = deepseek_v3.Model(config)
    mx.eval(model.parameters())
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: "switch_mlp" in path
        and hasattr(module, "to_quantized"),
    )
    mx.eval(model.parameters())
    config = dict(config) if isinstance(config, dict) else dataclasses.asdict(config)
    config["quantization"] = dict(group_size=32, bits=4, mode="affine")
    build, offload = root / "build", root / "offload"
    save_weights(str(build), model)
    (build / "config.json").write_text(json.dumps(config))
    repack(str(build), str(offload))
    return build, offload


def _assert_offload_parity(resident, offloaded):
    # Batched versus looped matmul has a different GEMM reduction order.
    mx.eval(resident, offloaded)
    relative_error = float(mx.abs(resident - offloaded).max()) / float(
        mx.abs(resident).max()
    )
    assert (
        relative_error < 0.02
    ), f"offloaded output diverged: {relative_error:.4f} relative"


@pytest.mark.parametrize("family", ["deepseek_v3", "laguna"])
def test_offload_loads_with_resident_parity(tmp_path, family):
    # Separate projections and fused gate_up_proj both go through the real loader.
    if family == "deepseek_v3":
        build, offload = _build_and_repack(tmp_path)
    else:
        config = laguna.ModelConfig(
            model_type="laguna",
            num_hidden_layers=2,
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=256,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
        )
        build, offload = _build_and_repack(tmp_path, laguna.Model(config), config)
    prompt = mx.array([[1, 2, 3, 4, 5, 6]])
    resident = load_model(build)(prompt).logits
    mx.eval(resident)
    model = load_model(offload)
    store = getattr(model, "moe_offload_store", None)
    assert store is not None, "load_model did not auto-patch the offload directory"
    assert store.swapped == 2
    _assert_offload_parity(resident, model(prompt).logits)


def test_patch_model_raises_on_empty_offload_dir(tmp_path):
    (tmp_path / "experts").mkdir()
    (tmp_path / "offload_index.json").write_text(
        json.dumps({"layers": [], "num_experts": 4})
    )
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        patch_model(nn.Module(), str(tmp_path))


def test_repack_raises_on_insufficient_disk_headroom(tmp_path):
    build, offload = tmp_path / "build", tmp_path / "offload"
    build.mkdir()
    (build / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 1024)
    fake_usage = shutil.disk_usage(tmp_path)._replace(free=0)
    with patch("mlx_vlm.moe_offload.shutil.disk_usage", return_value=fake_usage):
        with pytest.raises(ValueError):
            repack(str(build), str(offload))
    assert not (offload / "experts" / "layer_0000.safetensors").exists()


def test_plan_partitions_stacked_and_shared_experts():
    names = [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales",
        "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight",
        "language_model.model.layers.2.mlp.experts.3.gate_proj.weight",
    ]
    result = plan(names)
    assert names[0] in result["resident"]
    assert names[3] in result["resident"]
    assert result["layers"] == [1, 2]
    assert result["experts"][1][0][2] == "STACK"
    assert result["experts"][2][0][0] == "e3.gate_proj.weight"


def test_expert_store_get_is_thread_safe(tmp_path):
    _, offload = _build_and_repack(tmp_path)
    store = ExpertStore(str(offload))
    reference = {(layer, j): store.get(layer, j) for layer in (1, 2) for j in range(4)}
    n_threads, n_iters = 8, 200
    errors = []
    results = [[] for _ in range(n_threads)]

    # mmap arrays must be evaluated on their loading thread. Workers collect
    # references; the main thread checks every returned tensor after joining.
    def worker(tid):
        for i in range(n_iters):
            layer, j = 1 + (i % 2), (i + tid) % 4
            try:
                val = store.get(layer, j)
                assert len(val) == 3
                results[tid].append(((layer, j), val))
            except Exception as error:
                errors.append(error)
                return

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum(map(len, results)) == n_threads * n_iters
    for thread_results in results:
        for key, val in thread_results:
            for got, want in zip(val, reference[key]):
                for g, w in zip(got, want):
                    assert (g is None) == (w is None)
                    if g is not None:
                        assert mx.array_equal(g, w)


def test_fused_switch_layer_offloads_correctly(tmp_path):
    class FusedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = LagunaPackedSwitchGLU(32, 64, 4)

    class InnerModel(nn.Module):
        """Named ``model`` so paths read ``model.layers.N...``, matching
        the ``\\.layers\\.`` regex repack()/patch_model() rely on."""

        def __init__(self):
            super().__init__()
            self.router = nn.Linear(32, 4, bias=False)
            self.layers = [FusedMLP()]

    class FusedTestModel(nn.Module):
        """One MoE layer, fused switch_mlp, top-2-of-4 routing."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model_type = "fused_test_model"
            self.model = InnerModel()

        def __call__(self, x):
            g = self.model.router(x)
            indices = mx.argsort(-g, axis=-1)[..., :2].astype(mx.uint32)
            weights = mx.softmax(mx.take_along_axis(g, indices, axis=-1), axis=-1)
            y = self.model.layers[0].switch_mlp(x, indices)
            return (y * weights[..., None]).sum(-2)

    model = FusedTestModel({"model_type": "fused_test_model"})
    _, offload = _build_and_repack(tmp_path, model, model.config)
    x = mx.random.normal((3, 32))
    mx.eval(x)
    resident = model(x)
    mx.eval(resident)
    # Direct patching remains separate from registry-based loading above.
    store = patch_model(model, str(offload))
    assert store.swapped == 1
    _assert_offload_parity(resident, model(x))


def test_patch_model_raises_on_missing_expert_layer_file(tmp_path):
    _, offload = _build_and_repack(tmp_path)
    files = sorted((offload / "experts").iterdir())
    assert len(files) >= 2
    files[0].unlink()
    model = deepseek_v3.Model(_deepseek_config())
    mx.eval(model.parameters())
    with pytest.raises(ValueError, match="(?i)missing"):
        patch_model(model, str(offload))


def test_repack_sanitizes_raw_mixtral_style_expert_naming(tmp_path):
    config = minimax.ModelConfig(
        model_type="minimax_m2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        num_experts_per_tok=2,
        num_local_experts=4,
        shared_intermediate_size=128,
        num_hidden_layers=2,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        rotary_dim=16,
        vocab_size=256,
        head_dim=16,
    )
    model = minimax.Model(config)
    mx.eval(model.parameters())
    build_prefixed = tmp_path / "build_prefixed"
    build_raw, offload_raw = tmp_path / "build_raw", tmp_path / "offload_raw"
    save_weights(build_prefixed, model)
    weights = {}
    for path in sorted(build_prefixed.glob("*.safetensors")):
        weights.update(mx.load(str(path)))
    # Invert LanguageModel.sanitize()'s w1/w2/w3 -> switch_mlp
    # stacking, to reconstruct MiniMax-M2's genuine raw (Mixtral-
    # style) upstream checkpoint from this already-canonical model.
    mapping = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    raw_weights = {}
    for k, v in weights.items():
        handled = False
        for new_name, orig_name in mapping.items():
            suffix = f".block_sparse_moe.switch_mlp.{new_name}.weight"
            if k.startswith("language_model.") and suffix in k:
                layer_prefix = k[len("language_model.") :].split(".block_sparse_moe")[0]
                for e in range(v.shape[0]):
                    raw_weights[
                        f"{layer_prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                    ] = v[e]
                handled = True
                break
        if not handled:
            k2 = k[len("language_model.") :] if k.startswith("language_model.") else k
            raw_weights[k2] = v
    build_raw.mkdir()
    mx.save_safetensors(str(build_raw / "model.safetensors"), raw_weights)
    (build_raw / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    repack(str(build_raw), str(offload_raw))
    index = json.loads((offload_raw / "offload_index.json").read_text())
    # Both layers must be sanitized together for the model's global layer-0 guard.
    assert index["layers"] == [0, 1]
    assert index["num_experts"] == 4
    prompt = mx.array([[1, 2, 3, 4, 5]])
    resident = model(prompt).logits
    mx.eval(resident)
    offloaded = load_model(offload_raw)
    assert getattr(offloaded, "moe_offload_store", None) is not None
    _assert_offload_parity(resident, offloaded(prompt).logits)
