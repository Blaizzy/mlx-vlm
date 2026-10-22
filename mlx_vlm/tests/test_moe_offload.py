"""MoE checkpoint repacking, expert offload, and output parity."""

import dataclasses
import json
import shutil
import threading
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models import deepseek_v3, diffusion_gemma, laguna, minimax
from mlx_vlm.models.diffusion_gemma.language import DecoderLayer as DiffusionGemmaLayer
from mlx_vlm.models.diffusion_gemma.language import Experts as DiffusionGemmaExperts
from mlx_vlm.models.hy_v4.fused_switch_glu import FusedSwitchGLU
from mlx_vlm.models.inkling.language import InklingSwitchGLU
from mlx_vlm.models.laguna.language import LagunaPackedSwitchGLU
from mlx_vlm.models.minimax_m3_vl.language import (
    MiniMaxPackedSwitchGLU,
    MiniMaxSwiGLUOAI,
)
from mlx_vlm.models.openai_privacy_filter.config import (
    ModelConfig as PrivacyFilterConfig,
)
from mlx_vlm.models.openai_privacy_filter.openai_privacy_filter import (
    Experts as PrivacyFilterExperts,
)
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.moe_offload import ExpertStore, patch_model, plan, repack
from mlx_vlm.utils import load_model, save_weights


@pytest.fixture(params=[None, mx.cpu], ids=["default", "cpu"])
def offload_device(request):
    with mx.stream(request.param or mx.default_device()):
        yield


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("scale_name", ["weight_scale_2", "weight_global_scale"])
@pytest.mark.parametrize("scale_layout", ["scalar", "expert", "expert_column", "row"])
@pytest.mark.parametrize("decode", [False, True])
def test_native_nvfp4_global_scales_survive_repack(
    tmp_path, fused, scale_name, scale_layout, decode, offload_device
):
    from mlx.utils import tree_flatten

    from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears

    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.switch_mlp = (LagunaPackedSwitchGLU if fused else SwitchGLU)(64, 64, 4)
    model.model.layers = [layer]
    nn.quantize(model, mode="nvfp4", group_size=16, bits=4)
    scales = {}
    for path, value in tree_flatten(model.parameters()):
        if path.endswith(".weight"):
            scale = mx.array([0.137, 0.348, 0.779, 0.29])
            if scale_layout == "scalar":
                scale = mx.array(0.137)
            elif scale_layout == "expert_column":
                scale = scale[:, None]
            elif scale_layout == "row":
                # Sanitizing separately scaled gate/up weights produces this
                # layout. Vary every row so copying or swapping halves fails.
                scale = scale[:, None] * mx.linspace(0.7, 1.3, value.shape[-2])
            scales[path.removesuffix("weight") + scale_name] = scale
    replace_scaled_quantized_linears(model, scales)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    indices = mx.array([[[0, 1], [2, 3], [0, 3]]])
    if decode:
        x, indices = x[:, :1], indices[:, :1]
    expected = layer.switch_mlp(x, indices)
    mx.eval(expected)
    build, out = tmp_path / "build", tmp_path / "out"
    save_weights(str(build), model)
    (build / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}})
    )
    repack(str(build), str(out))
    patch_model(model, str(out), expert_cache_gb=0.00001)
    actual = layer.switch_mlp(x, indices)
    _assert_offload_parity(expected, actual)
    # The same experts must still produce the same result after eviction/reload.
    _assert_offload_parity(expected, layer.switch_mlp(x, indices))


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


def _build_and_repack(root, model=None, config=None, quant_paths=("switch_mlp",)):
    if model is None:
        config = _deepseek_config()
        model = deepseek_v3.Model(config)
    mx.eval(model.parameters())
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: any(q in path for q in quant_paths)
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
    comparisons = []
    for thread_results in results:
        for key, val in thread_results:
            for got, want in zip(val, reference[key]):
                for g, w in zip(got, want):
                    assert (g is None) == (w is None)
                    if g is not None:
                        comparisons.append(mx.array_equal(g, w))
    # Check every returned tensor with one synchronization instead of thousands.
    assert mx.all(mx.stack(comparisons)).item()


def test_selected_experts_reload_once_and_survive_tiny_cache(tmp_path):
    _, offload = _build_and_repack(tmp_path)
    store = ExpertStore(str(offload), expert_cache_bytes=1)
    first = store.get_many(1, [0, 2, 0])
    mx.eval(first)
    assert store.stats()["resident_bytes"] <= 1
    # Both experts were evicted. Reloading them must parse the layer only once,
    # and eviction must not invalidate tensors returned to the current graph.
    with patch("mlx.core.load", wraps=mx.load) as load:
        second = store.get_many(1, [0, 2])
        assert load.call_count == 1
    for j in first:
        for got, want in zip(second[j], first[j]):
            for g, w in zip(got, want):
                if w is not None:
                    assert mx.array_equal(g, w).item()
    assert store.stats()["metadata_reloads"] == 1
    assert store.stats()["resident_bytes"] <= 1


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


def test_repack_rewrites_stale_source_weight_index(tmp_path):
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
    build, offload = tmp_path / "build", tmp_path / "offload"
    save_weights(str(build), model)
    config_dict = dataclasses.asdict(config)
    config_dict["quantization"] = dict(group_size=32, bits=4, mode="affine")
    (build / "config.json").write_text(json.dumps(config_dict))

    index_path = build / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weights = {
        k: mx.array(v) for k, v in mx.load(str(build / "model.safetensors")).items()
    }
    mx.eval(list(weights.values()))
    sidecar_key = "language_model.model.embed_tokens.weight"
    assert sidecar_key in weights
    (build / "sidecar").mkdir()
    mx.save_safetensors(
        str(build / "sidecar" / "extra.safetensors"),
        {sidecar_key: weights[sidecar_key]},
    )
    mx.save_safetensors(
        str(build / "model.safetensors"),
        {k: v for k, v in weights.items() if k != sidecar_key},
    )
    index["weight_map"][sidecar_key] = "sidecar/extra.safetensors"
    index_path.write_text(json.dumps(index))

    repack(str(build), str(offload))
    assert (offload / "sidecar" / "extra.safetensors").exists()
    rewritten = json.loads((offload / "model.safetensors.index.json").read_text())
    shards = set(rewritten["weight_map"].values())
    assert shards and all(s.startswith("resident-") for s in shards)
    assert sidecar_key in rewritten["weight_map"]
    assert not any(
        "switch_mlp.gate_proj.weight" in k for k in rewritten["weight_map"]
    ), "routed experts must not be named in the resident index"

    prompt = mx.array([[1, 2, 3, 4, 5, 6]])
    resident = load_model(build)(prompt).logits
    mx.eval(resident)
    offloaded = load_model(offload)
    assert getattr(offloaded, "moe_offload_store", None) is not None
    _assert_offload_parity(resident, offloaded(prompt).logits)


_H, _I, _E, _K = 32, 64, 4, 2

# Every switch layer patch_model can swap, so a differently shaped one cannot
# reach the swap without a parity test. Keyed by what builds it.
_SWAPPABLE_BLOCKS = {
    "switch_glu": lambda: SwitchGLU(_H, _I, _E),
    "inkling": lambda: InklingSwitchGLU(_H, _I, _E),
    "hy_v4": lambda: FusedSwitchGLU(_H, _I, _E),
    "laguna": lambda: LagunaPackedSwitchGLU(_H, _I, _E),
    "minimax_m3_vl": lambda: MiniMaxPackedSwitchGLU(_H, _I, _E, MiniMaxSwiGLUOAI()),
    "diffusion_gemma": lambda: DiffusionGemmaExperts(
        diffusion_gemma.TextConfig(
            hidden_size=_H,
            moe_intermediate_size=_I,
            num_experts=_E,
            top_k_experts=_K,
        )
    ),
    "openai_privacy_filter": lambda: PrivacyFilterExperts(
        PrivacyFilterConfig(
            hidden_size=_H,
            intermediate_size=_I,
            num_local_experts=_E,
            num_experts_per_tok=_K,
        )
    ),
}


def _wrap_block(block):
    """A model exposing ``block`` at ``model.layers.0.experts``."""

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = block

    class InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    class BlockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "switch_block_probe"}
            self.model_type = "switch_block_probe"
            self.model = InnerModel()

    return BlockModel()


@pytest.mark.parametrize("family", sorted(_SWAPPABLE_BLOCKS))
@pytest.mark.parametrize("tokens", [1, 5])
def test_every_swappable_block_matches_resident(tmp_path, family, tokens):
    model = _wrap_block(_SWAPPABLE_BLOCKS[family]())
    _, offload = _build_and_repack(
        tmp_path, model, model.config, quant_paths=("experts",)
    )
    x = mx.random.normal((tokens, _H)).astype(mx.bfloat16)
    indices = mx.random.randint(0, _E, (tokens, _K)).astype(mx.uint32)
    mx.eval(x, indices)
    resident = model.model.layers[0].experts(x, indices)
    mx.eval(resident)

    store = patch_model(model, str(offload))
    assert store.swapped == 1
    _assert_offload_parity(resident, model.model.layers[0].experts(x, indices))


@pytest.mark.parametrize("shape", [(1, 1), (2, 9)])
def test_nontrivial_expert_scales_preserve_bfloat16(tmp_path, shape, offload_device):
    mx.random.seed(53)
    block = InklingSwitchGLU(64, 128, 16, bias=True)
    for projection in (block.gate_proj, block.up_proj, block.down_proj):
        projection.bias = mx.random.normal(projection.bias.shape) * 0.1
    block.set_dtype(mx.bfloat16)
    block.gate_scale = mx.linspace(0.17, 1.7, 16)
    block.out_scale = mx.linspace(0.23, 1.2, 16)
    model = _wrap_block(block)
    _, offload = _build_and_repack(
        tmp_path, model, model.config, quant_paths=("experts",)
    )
    x = mx.random.normal((*shape, 64)).astype(mx.bfloat16)
    # Repeated routes, multiple batch axes, and more than eight unique experts.
    base = mx.arange(shape[0] * shape[1]).reshape(*shape, 1)
    indices = ((base + mx.array([0, 0, 5, 11])) % 16).astype(mx.uint32)
    resident = block(x, indices)
    mx.eval(resident)
    patch_model(model, str(offload), expert_cache_gb=1e-9)
    for _ in range(2):
        _assert_offload_parity(resident, model.model.layers[0].experts(x, indices))


@pytest.mark.parametrize("tokens", [1, 9])
def test_dense_bfloat16_experts_still_run_on_cpu(tmp_path, tokens):
    with mx.stream(mx.cpu):
        mx.random.seed(53)
        block = SwitchGLU(64, 128, 16)
        block.set_dtype(mx.bfloat16)
        model = _wrap_block(block)
        x = mx.random.normal((tokens, 64)).astype(mx.bfloat16)
        indices = (mx.arange(tokens * 2).reshape(tokens, 2) % 16).astype(mx.uint32)
        # CPU gather_mm cannot evaluate BF16. Use the original dense matmuls
        # as the reference, which supported CPU offloading before batching.
        expected = []
        for token, experts in enumerate(indices.tolist()):
            routed = []
            for expert in experts:
                gate = x[token : token + 1] @ block.gate_proj.weight[expert].T
                up = x[token : token + 1] @ block.up_proj.weight[expert].T
                routed.append(
                    block.activation(up, gate) @ block.down_proj.weight[expert].T
                )
            expected.append(mx.concatenate(routed))
        expected = mx.stack(expected)
        mx.eval(expected)
        build, offload = tmp_path / "build", tmp_path / "offload"
        save_weights(str(build), model)
        (build / "config.json").write_text(json.dumps(model.config))
        repack(str(build), str(offload))
        patch_model(model, str(offload), expert_cache_gb=1e-9)
        for _ in range(2):
            actual = model.model.layers[0].experts(x, indices)
            assert actual.dtype == mx.bfloat16
            _assert_offload_parity(expected, actual)


def test_patch_model_refuses_a_layer_that_declares_no_gating(tmp_path):
    model = _wrap_block(SwitchGLU(_H, _I, _E))
    _, offload = _build_and_repack(
        tmp_path, model, model.config, quant_paths=("experts",)
    )
    del model.model.layers[0].experts.activation
    with pytest.raises(ValueError, match="(?i)activation"):
        patch_model(model, str(offload))


def test_diffusion_gemma_layer_applies_router_weights():
    """The block returns [..., K, D]; its caller must still weight and sum."""
    config = diffusion_gemma.TextConfig(
        hidden_size=_H,
        moe_intermediate_size=_I,
        num_experts=_E,
        top_k_experts=_K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        intermediate_size=_I,
        vocab_size=128,
    )
    layer = DiffusionGemmaLayer(config, 0)
    mx.eval(layer.parameters())
    x = mx.random.normal((1, 4, _H))
    mx.eval(x)

    indices = mx.zeros((4, _K), dtype=mx.uint32)
    base = mx.full((4, _K), 0.25)

    def run(weights):
        layer.router = lambda _flat: (indices, weights)
        out = layer(x)
        mx.eval(out)
        return out

    once, twice = run(base), run(base * 2.0)
    # Doubling the router weights must change the layer's output; if the
    # combine were dropped, the expert branch would ignore them entirely.
    assert float(mx.abs(once - twice).max()) > 1e-4
