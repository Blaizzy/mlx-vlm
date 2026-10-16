"""Loaded models go through one preparation call before the public Trainer."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.trainer import (
    CoreTrainingArgs,
    Trainer,
    TrainingTask,
    prepare_model_for_training,
)
from mlx_vlm.trainer.peft import DoRALinear, LoRALinear, load_adapters, save_adapter


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.embedding = nn.Embedding(16, 64)
        self.projection = nn.Linear(64, 64)

    def __call__(self, ids):
        return self.projection(self.embedding(ids).mean(axis=1))


class ModelPreparationTest(unittest.TestCase):
    def test_prepared_model_trains_directly_in_all_modes(self):
        for mode, bits in (
            ("lora", None),
            ("dora", None),
            ("full", None),
            ("lora", 4),
            ("dora", 8),
        ):
            with (
                self.subTest(mode=mode, bits=bits),
                tempfile.TemporaryDirectory() as folder,
            ):
                model = TinyModel()
                model.eval()
                result = prepare_model_for_training(
                    model,
                    train_type=mode,
                    lora_rank=2,
                    lora_alpha=4,
                    quantization_bits=bits,
                    verbose=False,
                )
                self.assertIs(result, model)
                self.assertTrue(model.training)
                trainable = dict(tree_flatten(model.trainable_parameters()))
                if mode == "full":
                    self.assertIn("embedding.weight", trainable)
                    self.assertIn("projection.weight", trainable)
                else:
                    self.assertTrue(
                        all(
                            name.endswith(("lora_a", "lora_b", ".m"))
                            for name in trainable
                        )
                    )
                    self.assertIsInstance(
                        model.projection, DoRALinear if mode == "dora" else LoRALinear
                    )
                    if bits is not None:
                        self.assertIsInstance(
                            model.projection.linear, nn.QuantizedLinear
                        )
                        self.assertEqual(model.projection.linear.bits, bits)
                        self.assertIsInstance(model.embedding, nn.QuantizedEmbedding)
                before = {name: mx.array(value) for name, value in trainable.items()}
                base_before = model.embedding.weight

                def batches(rows, train=False, seed=None):
                    while True:
                        yield from rows
                        if not train:
                            return

                task = TrainingTask(
                    loss=lambda m, ids: (((m(ids) - 1) ** 2).mean(), mx.array(1)),
                    batches=batches,
                )
                trainer = Trainer(
                    model,
                    CoreTrainingArgs(
                        iters=1,
                        learning_rate=0.01,
                        compile=False,
                        adapter_file=str(Path(folder) / "weights.safetensors"),
                    ),
                    task=task,
                    train_dataset=[mx.array([[1, 2, 3]])],
                )
                self.assertTrue(
                    mx.isfinite(mx.array(trainer.train()["train_loss"])).item()
                )
                after = dict(tree_flatten(model.trainable_parameters()))
                self.assertTrue(
                    any(
                        not mx.array_equal(value, after[name]).item()
                        for name, value in before.items()
                    )
                )
                if mode != "full":
                    self.assertTrue(
                        mx.array_equal(base_before, model.embedding.weight).item()
                    )

    def test_quantized_adapter_reload_reproduces_outputs_and_metadata(self):
        for mode in ("lora", "dora"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                model = TinyModel()
                base_weights = model.parameters()
                prepare_model_for_training(
                    model,
                    train_type=mode,
                    quantization_bits=4,
                    lora_rank=2,
                    verbose=False,
                )
                model.projection.lora_b = mx.full(model.projection.lora_b.shape, 0.1)
                ids = mx.array([[1, 2, 3]])
                expected = model(ids)
                path = Path(folder) / "weights.safetensors"
                save_adapter(model, path)
                config = json.loads((path.parent / "adapter_config.json").read_text())
                self.assertEqual(
                    set(config["base_quantization"]), {"embedding", "projection"}
                )
                for prepare in (True, False):
                    restored = TinyModel()
                    restored.update(base_weights)
                    if prepare:
                        # Saved DoRA/quantization settings override the new-adapter defaults.
                        prepare_model_for_training(
                            restored, checkpoint_path=path, verbose=False
                        )
                    else:
                        load_adapters(restored, path.parent)
                    self.assertTrue(mx.allclose(expected, restored(ids)).item())
                    self.assertEqual(restored._vlm_adapter_config, config)
                    save_adapter(restored, path.parent / "resaved" / path.name)
                    self.assertEqual(
                        json.loads(
                            (
                                path.parent / "resaved" / "adapter_config.json"
                            ).read_text()
                        ),
                        config,
                    )

    def test_full_mode_dequantizes_and_restores_full_checkpoint(self):
        model = TinyModel()
        nn.quantize(model, bits=4, group_size=64)
        model.freeze()
        ids = mx.array([[1, 2, 3]])
        expected = model(ids)
        prepare_model_for_training(model, train_type="full", verbose=False)
        self.assertIsInstance(model.embedding, nn.Embedding)
        self.assertIsInstance(model.projection, nn.Linear)
        self.assertTrue(mx.allclose(expected, model(ids), atol=1e-3).item())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            save_adapter(model, path)
            self.assertEqual(
                json.loads((path.parent / "adapter_config.json").read_text()),
                {"fine_tune_type": "full"},
            )
            restored = TinyModel()
            restored.freeze()
            prepare_model_for_training(
                restored, train_type="full", checkpoint_path=path.parent, verbose=False
            )
            self.assertTrue(mx.allclose(model(ids), restored(ids)).item())
            self.assertEqual(
                set(dict(tree_flatten(model.trainable_parameters()))),
                set(dict(tree_flatten(restored.trainable_parameters()))),
            )
            from mlx_vlm.trainer.peft.adapter_utils import (
                load_adapters as load_legacy_adapters,
            )

            for restore in (load_adapters, load_legacy_adapters):
                restored = restore(TinyModel(), path.parent)
                self.assertTrue(mx.allclose(model(ids), restored(ids)).item())
            # Older full checkpoints need no adapter sidecar.
            (path.parent / "adapter_config.json").unlink()
            prepare_model_for_training(
                TinyModel(), train_type="full", checkpoint_path=path, verbose=False
            )

    def test_explicit_targets_can_select_multimodal_towers(self):
        model = nn.Module()
        model.config = SimpleNamespace()
        model.language_model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        model.vision_model = nn.Linear(64, 64)
        prepare_model_for_training(
            model,
            target_modules=["language_model.layers.0", "vision_model"],
            verbose=False,
        )
        self.assertIsInstance(model.language_model.layers[0], LoRALinear)
        self.assertIsInstance(model.language_model.layers[1], nn.Linear)
        self.assertIsInstance(model.vision_model, LoRALinear)
        self.assertEqual(
            model.config.lora["lora_parameters"]["keys"],
            ["language_model.layers.0", "vision_model"],
        )

    def test_invalid_options_do_not_mutate_model(self):
        for options in (
            {"train_type": "unknown"},
            {"train_type": "full", "quantization_bits": 4},
            {"quantization_bits": 5},
            {"quantization_bits": 4, "lora_rank": 0},
            {"quantization_bits": 4, "target_modules": ["missing"]},
            {"checkpoint_path": "/nonexistent/weights.safetensors"},
        ):
            with self.subTest(options=options):
                model = TinyModel()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare_model_for_training(model, verbose=False, **options)
                self.assertIsInstance(model.projection, nn.Linear)
                self.assertNotIn("lora", model.config)
                self.assertIn(
                    "projection.weight",
                    dict(tree_flatten(model.trainable_parameters())),
                )

    def test_incompatible_checkpoint_mode_quantization_and_repeat_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = prepare_model_for_training(
                TinyModel(), quantization_bits=4, verbose=False
            )
            path = Path(folder) / "weights.safetensors"
            save_adapter(model, path)
            with self.assertRaisesRegex(ValueError, "already has adapters"):
                prepare_model_for_training(model, verbose=False)
            with self.assertRaisesRegex(ValueError, "adapter checkpoint"):
                prepare_model_for_training(
                    TinyModel(), train_type="full", checkpoint_path=path
                )
            with self.assertRaisesRegex(ValueError, "Quantization differs"):
                prepare_model_for_training(
                    TinyModel(), quantization_bits=8, checkpoint_path=path
                )
            base = TinyModel()
            nn.quantize(base, bits=8, group_size=64)
            with self.assertRaisesRegex(ValueError, "Quantization differs"):
                prepare_model_for_training(base, checkpoint_path=path)


if __name__ == "__main__":
    unittest.main()
