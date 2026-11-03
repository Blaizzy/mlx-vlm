"""Metric forwarding without network access or optional tracking packages."""

import tempfile
import unittest
from unittest import mock

from mlx_vlm.trainer.common.callbacks import TrainingCallback, WandBCallback


class CallbacksTest(unittest.TestCase):
    def test_reports_are_sent_and_forwarded_without_mutation(self):
        wandb = mock.Mock()
        wrapped = mock.Mock(spec=TrainingCallback)
        scalar = mock.Mock()
        scalar.tolist.return_value = 0.25
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict("sys.modules", {"wandb": wandb}):
                callback = WandBCallback("test", directory, {"seed": 42}, wrapped)
            train = {"iteration": 3, "train_loss": scalar, "custom_metric": 2}
            validation = {"iteration": 3, "val_loss": 0.5}
            callback.on_train_loss_report(train)
            callback.on_val_loss_report(validation)
            callback.finish()
        self.assertIs(train["train_loss"], scalar)
        wandb.init.return_value.log.assert_has_calls(
            [
                mock.call(
                    {"iteration": 3, "train_loss": 0.25, "custom_metric": 2}, step=3
                ),
                mock.call(validation, step=3),
            ]
        )
        wrapped.on_train_loss_report.assert_called_once_with(train)
        wrapped.on_val_loss_report.assert_called_once_with(validation)
        wandb.init.return_value.finish.assert_called_once_with()

    def test_missing_optional_dependency(self):
        with mock.patch.dict("sys.modules", {"wandb": None}):
            with self.assertRaisesRegex(ImportError, "pip install wandb"):
                WandBCallback("test", "unused", {})


if __name__ == "__main__":
    unittest.main()
