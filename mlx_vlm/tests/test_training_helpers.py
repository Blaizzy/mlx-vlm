"""Reusable training helpers preserve weights and gradient updates."""

import argparse
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_vlm.trainer.common.config import CoreTrainingArgs
from mlx_vlm.trainer.common.utils import (
    accumulate_gradients,
    apply_accumulated_gradients,
    apply_final_accumulated_gradients,
    apply_gradient_step,
    enable_gradient_checkpointing,
    evaluate_weighted_batches,
    resolve_args,
    save_trainable_weights,
)


class TrainingHelpersTest(unittest.TestCase):
    def test_save_only_trainable_parameters_and_final_refresh(self):
        model = nn.Linear(2, 1)
        model.freeze(keys=["bias"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "custom.safetensors"
            save_trainable_weights(model, path, iteration=3)
            checkpoint = path.with_name("0000003_custom.safetensors")
            initial = mx.load(str(checkpoint))
            self.assertEqual(set(initial), {"weight"})
            model.weight = model.weight + 1
            save_trainable_weights(model, path)
            self.assertTrue(
                mx.allclose(mx.load(str(path))["weight"], model.weight).item()
            )
            self.assertTrue(
                mx.allclose(
                    mx.load(str(checkpoint))["weight"], initial["weight"]
                ).item()
            )

    def test_partial_accumulation_uses_actual_microstep_count(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight = mx.array([[10.0]])
        optimizer = optim.SGD(learning_rate=1.0)
        gradient = accumulate_gradients({"weight": mx.array([[2.0]])})
        gradient = accumulate_gradients({"weight": mx.array([[4.0]])}, gradient)
        apply_accumulated_gradients(model, optimizer, gradient, steps=2)
        self.assertAlmostEqual(model.weight.item(), 7.0)

    def test_gradient_step_handles_accumulation_and_update(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight = mx.array([[10.0]])
        optimizer = optim.SGD(learning_rate=1.0)

        def loss(model, x, scale):
            return model(x).sum() * scale, mx.array(1, dtype=mx.int32)

        loss_value_and_grad = nn.value_and_grad(model, loss)
        loss_value, metrics, gradients = apply_gradient_step(
            model,
            optimizer,
            loss_value_and_grad,
            mx.ones((1, 1)),
            update=False,
            accumulation_steps=2,
            loss_args=(1.0,),
        )
        mx.eval(loss_value, metrics, gradients)
        self.assertEqual(int(metrics["weight"]), 1)
        self.assertIsNotNone(gradients)

        _, _, gradients = apply_gradient_step(
            model,
            optimizer,
            loss_value_and_grad,
            mx.ones((1, 1)),
            gradients,
            update=True,
            accumulation_steps=2,
            loss_args=(1.0,),
        )
        self.assertIsNone(gradients)
        self.assertAlmostEqual(model.weight.item(), 9.0)

    def test_final_partial_gradient_helper(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight = mx.array([[10.0]])
        optimizer = optim.SGD(learning_rate=1.0)
        partial_steps = apply_final_accumulated_gradients(
            model,
            optimizer,
            {"weight": mx.array([[2.0]])},
            total_steps=3,
            accumulation_steps=2,
        )
        self.assertEqual(partial_steps, 1)
        self.assertAlmostEqual(model.weight.item(), 8.0)

    def test_weighted_evaluation_reduces_losses(self):
        model = nn.Linear(1, 1)

        def loss(_, value):
            return mx.array(value, dtype=mx.float32), mx.array(value, dtype=mx.int32)

        result = evaluate_weighted_batches(
            model,
            [1, 3],
            loss,
            num_batches=-1,
        )
        self.assertAlmostEqual(result, 2.5)

    def test_bounded_evaluation_prepares_only_requested_batches(self):
        prepared = []

        def batches():
            for value in (1, 3, 5):
                prepared.append(value)
                yield value

        def loss(_, value):
            return mx.array(value, dtype=mx.float32), mx.array(1, dtype=mx.int32)

        result = evaluate_weighted_batches(
            nn.Linear(1, 1), batches(), loss, num_batches=2
        )
        self.assertEqual(result, 2.0)
        self.assertEqual(prepared, [1, 3])

    def test_resolve_args_merges_mapping_and_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", default=None)
        resolved = resolve_args(
            {"name": "custom"},
            parser,
            {"name": "default", "nested": {"value": 1}},
            config_name="test",
        )
        self.assertEqual(resolved.name, "custom")
        self.assertEqual(resolved.nested, {"value": 1})

    def test_checkpointing_preserves_gradients_and_is_idempotent(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mx.array([2.0])

            def __call__(self, x):
                return self.weight * x

        model = Layer()

        def loss(model, x):
            return model(x).sum()

        expected, expected_grad = nn.value_and_grad(model, loss)(model, mx.array([3.0]))
        enable_gradient_checkpointing([model])
        wrapped = Layer.__call__
        enable_gradient_checkpointing([model])
        self.assertIs(Layer.__call__, wrapped)
        actual, actual_grad = nn.value_and_grad(model, loss)(model, mx.array([3.0]))
        self.assertEqual(actual.item(), expected.item())
        self.assertEqual(actual_grad["weight"].item(), expected_grad["weight"].item())

    def test_common_argument_validation(self):
        CoreTrainingArgs(steps_per_eval=None, val_batches=-1).validate()
        for kwargs in ({"steps_per_eval": 0}, {"gradient_accumulation_steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CoreTrainingArgs(**kwargs).validate()
