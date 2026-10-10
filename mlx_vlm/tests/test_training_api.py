"""Exercise the public API and shared engine with real MLX gradients."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx_vlm.train import _configure_model
from mlx_vlm.trainer import (
    CoreTrainingArgs,
    DPOTrainingArgs,
    ORPOTrainingArgs,
    Trainer,
    TrainingCallback,
    TrainingTask,
    VLMTrainingArgs,
)
from mlx_vlm.trainer.datasets.loading import CacheDataset


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", vocab_size=8)
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8)

    def __call__(self, input_ids, pixel_values=None, attention_mask=None, **kwargs):
        return self.projection(self.embedding(input_ids))


class Recorder(TrainingCallback):
    def __init__(self):
        self.train, self.val = [], []

    def on_train_loss_report(self, info):
        self.train.append(info)

    def on_val_loss_report(self, info):
        self.val.append(info)


def example(ids=(1, 3, 2)):
    return {
        "input_ids": mx.array(ids),
        "attention_mask": mx.ones((len(ids),), dtype=mx.int32),
        "completion_mask": mx.array([0] + [1] * (len(ids) - 1)),
    }


class EngineTest(unittest.TestCase):
    def test_custom_task_partial_update_final_save_and_evaluation_mode(self):
        for compiled in (False, True):
            with (
                self.subTest(compiled=compiled),
                tempfile.TemporaryDirectory() as folder,
            ):
                model = nn.Linear(1, 1, bias=False)
                model.weight = mx.array([[10.0]])
                callback = Recorder()

                def batches(rows, train=False, seed=None):
                    while True:
                        yield from rows
                        if not train:
                            return

                task = TrainingTask(lambda m, x: (m(x).sum(), mx.array(1)), batches)
                args = CoreTrainingArgs(
                    iters=3,
                    gradient_accumulation_steps=2,
                    steps_per_report=3,
                    steps_per_save=3,
                    steps_per_eval=3,
                    val_batches=-1,
                    compile=compiled,
                    adapter_file=f"{folder}/weights.safetensors",
                )
                trainer = Trainer(
                    model,
                    args,
                    task=task,
                    optimizer=optim.SGD(learning_rate=1.0),
                    train_dataset=[mx.ones((1, 1))],
                    eval_dataset=[mx.ones((1, 1))],
                    training_callback=callback,
                )
                result = trainer.train()
                self.assertAlmostEqual(model.weight.item(), 8.0)
                self.assertEqual(result["optimizer_step"], 2)
                self.assertEqual(callback.train[-1]["optimizer_step"], 2)
                self.assertEqual(callback.val[-1]["val_loss"], 8.0)
                for filename in ("weights.safetensors", "0000003_weights.safetensors"):
                    self.assertAlmostEqual(
                        mx.load(f"{folder}/{filename}")["weight"].item(), 8.0
                    )
                model.eval()
                self.assertEqual(trainer.evaluate()["val_loss"], 8.0)
                self.assertFalse(model.training)

    def test_builtin_sft_dpo_orpo_update_policy_and_keep_reference_frozen(self):
        for mode, cls in (
            ("sft", VLMTrainingArgs),
            ("dpo", DPOTrainingArgs),
            ("orpo", ORPOTrainingArgs),
        ):
            for compiled in (False, True):
                with (
                    self.subTest(mode=mode, compiled=compiled),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    mx.random.seed(7)
                    model, reference = TinyLM(), TinyLM()
                    before = [
                        mx.array(value) for _, value in tree_flatten(model.parameters())
                    ]
                    ref_before = [
                        mx.array(value)
                        for _, value in tree_flatten(reference.parameters())
                    ]
                    rows = (
                        [example()]
                        if mode == "sft"
                        else [{"chosen": example(), "rejected": example((1, 4, 2))}]
                    )
                    args = cls(
                        iters=2,
                        steps_per_report=2,
                        max_seq_length=8,
                        pad_to_multiple=1,
                        compile=compiled,
                        adapter_file=f"{folder}/adapter.safetensors",
                    )
                    trainer = Trainer(
                        model,
                        args,
                        task="vlm",
                        algorithm=mode,
                        train_dataset=rows,
                        eval_dataset=rows,
                        prepared=True,
                        reference_model=reference if mode == "dpo" else None,
                    )
                    self.assertTrue(
                        mx.isfinite(mx.array(trainer.train()["train_loss"])).item()
                    )
                    after = [value for _, value in tree_flatten(model.parameters())]
                    self.assertTrue(
                        any(
                            not mx.array_equal(a, b).item()
                            for a, b in zip(before, after)
                        )
                    )
                    self.assertTrue(
                        all(
                            mx.array_equal(a, b).item()
                            for a, b in zip(
                                ref_before,
                                [v for _, v in tree_flatten(reference.parameters())],
                            )
                        )
                    )
                    self.assertTrue(
                        mx.isfinite(mx.array(trainer.evaluate()["val_loss"])).item()
                    )

    def test_reference_cannot_be_policy(self):
        model = TinyLM()
        with self.assertRaisesRegex(ValueError, "separate"):
            Trainer(
                model,
                DPOTrainingArgs(),
                task="vlm",
                algorithm="dpo",
                reference_model=model,
                prepared=True,
            )

    def test_lora_targets_save_reload_metadata(self):
        model = TinyLM()
        args = SimpleNamespace(
            train_type="lora",
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["projection"],
        )
        _configure_model(model, args)
        names = [name for name, _ in tree_flatten(model.trainable_parameters())]
        self.assertTrue(names)
        self.assertTrue(all("lora_" in name for name in names))
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(
                model,
                VLMTrainingArgs(adapter_file=f"{folder}/adapter.safetensors"),
                prepared=True,
            )
            trainer.save_adapter()
            config = json.loads(Path(folder, "adapter_config.json").read_text())
            self.assertEqual(config["lora_parameters"]["keys"], ["projection"])
            restored = TinyLM()
            from mlx_vlm.trainer.peft.utils import _apply_lora_layers

            _apply_lora_layers(restored, config)
            restored.load_weights(f"{folder}/adapter.safetensors", strict=False)


class DataAndCLITest(unittest.TestCase):
    def test_raw_vlm_and_preference_preprocess_once_and_keep_cache_bounded(self):
        def template(processor, config, messages, **kwargs):
            return messages

        def prepare(**kwargs):
            messages = kwargs["prompts"][0]
            ids = [1] + [3 if m["role"] == "user" else 4 for m in messages]
            return {"input_ids": np.array([ids], dtype=np.int32)}

        helpers = ({}, template, prepare, None)
        processor = SimpleNamespace(pad_token_id=7)
        with mock.patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers", return_value=helpers
        ):
            for mode, cls in (("sft", VLMTrainingArgs), ("orpo", ORPOTrainingArgs)):
                with self.subTest(mode=mode):
                    rows = (
                        [
                            {
                                "messages": [
                                    {"role": "user", "content": "prompt"},
                                    {"role": "assistant", "content": "response"},
                                ]
                            }
                        ]
                        if mode == "sft"
                        else [{"prompt": "prompt", "good": "yes", "bad": "no"}]
                    )
                    trainer = Trainer(
                        TinyLM(),
                        cls(cache_size=1, train_on_completions=True),
                        processing_class=processor,
                        train_dataset=rows,
                        task="vlm",
                        algorithm=mode,
                        dataset_config={
                            "chosen_feature": "good",
                            "rejected_feature": "bad",
                        },
                    )
                    item = trainer.train_dataset[0]
                    self.assertIs(item, trainer.train_dataset[0])
                    self.assertEqual(len(trainer.train_dataset._processed), 1)
                    self.assertEqual(trainer.train_dataset.itemlen(0), 3)
                    self.assertEqual(trainer.args.pad_token_id, 7)
                    if mode == "orpo":
                        self.assertIn("completion_mask", item["chosen"])

    def test_bounded_cache_recomputes_evicted_features_but_keeps_lengths(self):
        data = SimpleNamespace()

        class Rows:
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return index

            def process(self, index):
                data.calls.append(index)
                return [index] * (index + 1)

        data.calls = []
        cache = CacheDataset(Rows(), max_size=1)
        self.assertEqual([cache.itemlen(i) for i in range(3)], [1, 2, 3])
        self.assertEqual(len(cache._processed), 1)
        self.assertEqual([cache.itemlen(i) for i in range(3)], [1, 2, 3])
        self.assertEqual(data.calls, [0, 1, 2])
        self.assertEqual(cache[0], [0])
        self.assertEqual(data.calls, [0, 1, 2, 0])
        self.assertEqual(cache[-1], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
