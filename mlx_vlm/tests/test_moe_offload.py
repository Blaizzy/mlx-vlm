"""MoE checkpoint repacking, expert offload, and output parity."""

import unittest

import mlx.core as mx
import mlx.nn as nn


class TestMoEOffload(unittest.TestCase):
    """Repacks a real (tiny) checkpoint via mlx-vlm's own save_weights/
    nn.quantize path and confirms the offloaded model matches resident.
    2% relative tolerance: batched vs. looped matmul differs by ~1e-3
    relative from GEMM reduction order alone, not a bug."""

    def _tiny_deepseek_v3_config(self):
        from mlx_vlm.models import deepseek_v3

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
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_bias=False,
        )

    def _build_and_repack(self, tmp_dir, quantize):
        import dataclasses
        import json
        import os
        from pathlib import Path

        from mlx_vlm.models import deepseek_v3
        from mlx_vlm.moe_offload import repack
        from mlx_vlm.utils import save_weights

        config = self._tiny_deepseek_v3_config()
        model = deepseek_v3.Model(config)
        mx.eval(model.parameters())

        group_size, bits = 32, 4
        if quantize:

            def only_switch_mlp(path, module):
                return "switch_mlp" in path and hasattr(module, "to_quantized")

            nn.quantize(
                model, group_size=group_size, bits=bits, class_predicate=only_switch_mlp
            )
            mx.eval(model.parameters())

        build = os.path.join(tmp_dir, "build")
        offload = os.path.join(tmp_dir, "offload")
        save_weights(build, model)

        cfg_dict = dataclasses.asdict(config)
        if quantize:
            cfg_dict["quantization"] = {
                "group_size": group_size,
                "bits": bits,
                "mode": "affine",
            }
        with open(os.path.join(build, "config.json"), "w") as f:
            json.dump(cfg_dict, f)

        repack(build, offload)
        return Path(build), Path(offload)

    def _assert_offload_matches_resident(self, quantize, seq_len=6):
        import shutil
        import tempfile

        from mlx_vlm.utils import load_model

        tmp_dir = tempfile.mkdtemp()
        try:
            build, offload = self._build_and_repack(tmp_dir, quantize)

            prompt = mx.array([[(i % 250) + 1 for i in range(seq_len)]])

            resident_model = load_model(build)
            logits_resident = resident_model(prompt).logits
            mx.eval(logits_resident)

            offload_model = load_model(offload)
            store = getattr(offload_model, "moe_offload_store", None)
            self.assertIsNotNone(store, "load_model did not auto-patch the offload dir")
            self.assertEqual(store.swapped, 2)  # first_k_dense_replace=1, 3 layers

            logits_offload = offload_model(prompt).logits
            mx.eval(logits_offload)

            diff = mx.abs(logits_resident - logits_offload).max().item()
            rel = diff / float(mx.abs(logits_resident).max())
            self.assertLess(rel, 0.02, f"offloaded output diverged: {rel:.4f} relative")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_offload_matches_resident_quantized(self):
        self._assert_offload_matches_resident(quantize=True)

    def test_patch_model_raises_on_empty_offload_dir(self):
        """A malformed offload dir (index present, no expert files) must
        fail loudly via patch_model, not silently load fully resident."""
        import json
        import os
        import shutil
        import tempfile

        import mlx.nn as nn_mod

        from mlx_vlm.moe_offload import patch_model

        tmp_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp_dir, "experts"))  # present but empty
            json.dump(
                {"layers": [], "num_experts": 4},
                open(os.path.join(tmp_dir, "offload_index.json"), "w"),
            )
            json.dump({}, open(os.path.join(tmp_dir, "config.json"), "w"))

            with self.assertRaises(ValueError):
                patch_model(nn_mod.Module(), tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_repack_raises_on_insufficient_disk_headroom(self):
        """repack() must refuse up front when free space can't cover source
        + growing offload dir coexisting, rather than fail mid-write with a
        partially-corrupted offload dir."""
        import os
        import shutil
        import tempfile
        from unittest import mock

        from mlx_vlm.moe_offload import repack

        tmp_dir = tempfile.mkdtemp()
        try:
            build = os.path.join(tmp_dir, "build")
            offload = os.path.join(tmp_dir, "offload")
            os.makedirs(build)
            with open(
                os.path.join(build, "model-00001-of-00001.safetensors"), "wb"
            ) as f:
                f.write(b"\0" * 1024)

            fake_usage = shutil.disk_usage(tmp_dir)._replace(free=0)
            with mock.patch(
                "mlx_vlm.moe_offload.shutil.disk_usage", return_value=fake_usage
            ):
                with self.assertRaises(ValueError):
                    repack(build, offload)
            self.assertFalse(
                os.path.exists(
                    os.path.join(offload, "experts", "layer_0000.safetensors")
                )
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_plan_partitions_stacked_and_shared_experts(self):
        from mlx_vlm.moe_offload import plan

        names = [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales",
            "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight",
            "language_model.model.layers.2.mlp.experts.3.gate_proj.weight",
        ]
        result = plan(names)

        self.assertIn(
            "language_model.model.layers.0.self_attn.q_proj.weight", result["resident"]
        )
        self.assertIn(
            "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight",
            result["resident"],
        )
        self.assertEqual(result["layers"], [1, 2])
        stacked_entries = result["experts"][1]
        self.assertEqual(stacked_entries[0][2], "STACK")
        perexpert_entries = result["experts"][2]
        self.assertEqual(perexpert_entries[0][0], "e3.gate_proj.weight")

    def test_expert_store_get_is_thread_safe(self):
        """get() reads only immutable post-init state (no LRU cache -- removed
        after measuring a 0% hit rate for single-request serving), so
        concurrent calls need no locking. Verify under real concurrent
        pressure: no exception, and every returned value matches a
        single-threaded reference call for the same (layer, expert)."""
        import shutil
        import tempfile
        import threading

        from mlx_vlm.moe_offload import ExpertStore

        tmp_dir = tempfile.mkdtemp()
        try:
            build, offload = self._build_and_repack(tmp_dir, quantize=True)
            store = ExpertStore(str(offload))
            reference = {
                (layer, j): store.get(layer, j) for layer in (1, 2) for j in range(4)
            }

            n_threads, n_iters = 8, 200
            errors = []
            # mx.load()'s mmap'd arrays can only be *evaluated* from the
            # thread that loaded them (a pre-existing mx.load() constraint,
            # not this store's), so workers only collect unevaluated results
            # -- get() itself is pure dict access, no eval -- and every
            # value check happens on the main thread after joining.
            results = [[] for _ in range(n_threads)]

            def worker(tid):
                for i in range(n_iters):
                    layer, j = 1 + (i % 2), (i + tid) % 4
                    try:
                        val = store.get(layer, j)
                        assert len(val) == 3
                        results[tid].append(((layer, j), val))
                    except Exception as e:
                        errors.append(e)
                        return

            threads = [
                threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            self.assertEqual(sum(len(r) for r in results), n_threads * n_iters)
            for thread_results in results:
                for key, val in thread_results:
                    want = reference[key]
                    for got, want_trip in zip(val, want):
                        for g, w in zip(got, want_trip):
                            self.assertEqual(g is None, w is None)
                            if g is not None:
                                self.assertTrue(mx.array_equal(g, w))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fused_switch_layer_offloads_correctly(self):
        """Laguna/MiniMax-M3-VL use a fused gate_up_proj instead of separate
        gate_proj/up_proj; confirm the repack-time split reconstructs it."""
        import json
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.laguna.language import LagunaPackedSwitchGLU
        from mlx_vlm.moe_offload import repack
        from mlx_vlm.utils import save_weights

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
        mx.eval(model.parameters())

        group_size, bits = 32, 4
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda p, m: "switch_mlp" in p
            and hasattr(m, "to_quantized"),
        )
        mx.eval(model.parameters())

        tmp_dir = tempfile.mkdtemp()
        try:
            build = Path(os.path.join(tmp_dir, "build"))
            offload = Path(os.path.join(tmp_dir, "offload"))
            save_weights(build, model)
            json.dump(
                {
                    "model_type": "fused_test_model",
                    "quantization": {
                        "group_size": group_size,
                        "bits": bits,
                        "mode": "affine",
                    },
                },
                open(build / "config.json", "w"),
            )
            repack(str(build), str(offload))

            x = mx.random.normal((3, 32))
            mx.eval(x)

            resident_out = model(x)
            mx.eval(resident_out)

            # Patch the already-loaded resident model directly, skipping
            # load_model's full model-registry plumbing for this test-only class.
            from mlx_vlm.moe_offload import patch_model

            store = patch_model(model, str(offload))
            self.assertEqual(store.swapped, 1)
            offload_out = model(x)
            mx.eval(offload_out)

            diff = mx.abs(resident_out - offload_out).max().item()
            rel = diff / float(mx.abs(resident_out).max())
            self.assertLess(
                rel, 0.02, f"fused switch layer offload diverged: {rel:.4f}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fused_switch_layer_loads_through_load_model(self):
        """load_model()'s malformed-offload-dir check only recognized
        PEREXPERT_RE/STACKED_RE, not STACKED_FUSED_RE -- so a fused
        gate_up_proj checkpoint (Laguna/MiniMax-M3-VL) repacked correctly
        still failed to load through the real, documented mlx_vlm.load()
        path with "malformed repack() output?", even though patch_model()
        itself handled it fine. Unlike test_fused_switch_layer_offloads_
        correctly (which calls patch_model() directly, bypassing this
        check), this goes through load_model() end-to-end."""
        import dataclasses
        import json
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from mlx_vlm.models import laguna
        from mlx_vlm.moe_offload import repack
        from mlx_vlm.utils import load_model, save_weights

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
        model = laguna.Model(config)
        mx.eval(model.parameters())

        group_size, bits = 32, 4

        def only_switch_mlp(path, module):
            return "switch_mlp" in path and hasattr(module, "to_quantized")

        nn.quantize(
            model, group_size=group_size, bits=bits, class_predicate=only_switch_mlp
        )
        mx.eval(model.parameters())

        tmp_dir = tempfile.mkdtemp()
        try:
            build = os.path.join(tmp_dir, "build")
            offload = os.path.join(tmp_dir, "offload")
            save_weights(build, model)
            cfg_dict = dataclasses.asdict(config)
            cfg_dict["quantization"] = {
                "group_size": group_size,
                "bits": bits,
                "mode": "affine",
            }
            with open(os.path.join(build, "config.json"), "w") as f:
                json.dump(cfg_dict, f)
            repack(build, offload)

            prompt = mx.array([[1, 2, 3, 4, 5, 6]])
            resident_model = load_model(Path(build))
            logits_resident = resident_model(prompt).logits
            mx.eval(logits_resident)

            offload_model = load_model(Path(offload))
            store = getattr(offload_model, "moe_offload_store", None)
            self.assertIsNotNone(
                store, "load_model did not auto-patch the fused offload dir"
            )
            self.assertEqual(store.swapped, 2)

            logits_offload = offload_model(prompt).logits
            mx.eval(logits_offload)

            diff = mx.abs(logits_resident - logits_offload).max().item()
            rel = diff / float(mx.abs(logits_resident).max())
            self.assertLess(
                rel, 0.02, f"fused offload output diverged: {rel:.4f} relative"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_patch_model_raises_on_missing_expert_layer_file(self):
        """patch_model()'s only completeness check was the global
        swapped[0]==0 count -- a single missing/corrupted
        experts/layer_*.safetensors (partial repack, interrupted transfer)
        left that one layer un-swapped with no error anywhere, silently
        running it on random-init weights, so long as at least one other
        layer swapped fine. Must now raise instead."""
        import os
        import shutil
        import tempfile

        from mlx_vlm.models import deepseek_v3
        from mlx_vlm.moe_offload import patch_model

        tmp_dir = tempfile.mkdtemp()
        try:
            build, offload = self._build_and_repack(tmp_dir, quantize=True)

            # Corrupt the repack: delete one (of two) MoE layers' expert file.
            experts_dir = os.path.join(str(offload), "experts")
            layer_files = sorted(os.listdir(experts_dir))
            self.assertGreaterEqual(len(layer_files), 2)
            os.remove(os.path.join(experts_dir, layer_files[0]))

            config = self._tiny_deepseek_v3_config()
            model = deepseek_v3.Model(config)
            mx.eval(model.parameters())

            with self.assertRaises(ValueError) as ctx:
                patch_model(model, str(offload))
            self.assertIn("missing", str(ctx.exception).lower())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_repack_sanitizes_raw_mixtral_style_expert_naming(self):
        """MiniMax's real upstream checkpoint uses Mixtral-style per-expert
        ``block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`` naming, not the
        ``experts.{e}.{gate,up,down}_proj.weight`` convention plan() expects
        directly -- repack() must fall back to the model's own sanitize()
        (already used for regular, non-offload loading) to normalize it
        first. Regression coverage for two real bugs caught building that
        fallback: (1) calling sanitize() on a later layer's tensors alone,
        with no layer-0 key present at all, silently no-ops for any
        sanitize() that gates its whole MoE-restructuring block on "is
        layer 0's raw key present" as a one-shot global probe rather than a
        true per-layer guard (minimax's does) -- both layers must still be
        offloaded, not just layer 0; (2) some models put the real
        restructuring on Model.sanitize() and delegate to LanguageModel from
        inside it (minimax doesn't even re-export LanguageModel from its
        __init__.py), so calling only one step misses the other's family."""
        import dataclasses
        import json
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from mlx_vlm.models import minimax
        from mlx_vlm.moe_offload import repack
        from mlx_vlm.utils import load_model, save_weights

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
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            rotary_dim=16,
            vocab_size=256,
            head_dim=16,
        )
        model = minimax.Model(config)
        mx.eval(model.parameters())

        tmp_dir = tempfile.mkdtemp()
        try:
            build_prefixed = os.path.join(tmp_dir, "build_prefixed")
            build_raw = os.path.join(tmp_dir, "build_raw")
            offload_raw = os.path.join(tmp_dir, "offload_raw")

            save_weights(build_prefixed, model)
            weights = {}
            for fn in sorted(os.listdir(build_prefixed)):
                if fn.endswith(".safetensors"):
                    weights.update(mx.load(os.path.join(build_prefixed, fn)))

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
                        layer_prefix = k[len("language_model.") :].split(
                            ".block_sparse_moe"
                        )[0]
                        for e in range(v.shape[0]):
                            raw_weights[
                                f"{layer_prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                            ] = v[e]
                        handled = True
                        break
                if not handled:
                    k2 = (
                        k[len("language_model.") :]
                        if k.startswith("language_model.")
                        else k
                    )
                    raw_weights[k2] = v

            os.makedirs(build_raw, exist_ok=True)
            mx.save_safetensors(
                os.path.join(build_raw, "model.safetensors"), raw_weights
            )
            with open(os.path.join(build_raw, "config.json"), "w") as f:
                json.dump(dataclasses.asdict(config), f)

            repack(build_raw, offload_raw)

            idx = json.load(open(os.path.join(offload_raw, "offload_index.json")))
            self.assertEqual(
                idx["layers"],
                [0, 1],
                "layer 1 was silently skipped -- sanitize() was only called "
                "with layer 1's own tensors, tripping a global layer-0 guard",
            )
            self.assertEqual(idx["num_experts"], 4)

            prompt = mx.array([[1, 2, 3, 4, 5]])
            logits_true = model(prompt).logits
            mx.eval(logits_true)

            offload_model = load_model(Path(offload_raw))
            store = getattr(offload_model, "moe_offload_store", None)
            self.assertIsNotNone(
                store, "load_model did not auto-patch the sanitize-fallback offload dir"
            )
            logits_offload = offload_model(prompt).logits
            mx.eval(logits_offload)

            diff = mx.abs(logits_true - logits_offload).max().item()
            rel = diff / float(mx.abs(logits_true).max())
            self.assertLess(
                rel, 0.02, f"sanitize-fallback offload output diverged: {rel:.4f}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
