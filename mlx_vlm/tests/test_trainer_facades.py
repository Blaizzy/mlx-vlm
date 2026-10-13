"""Unit tests for object facades over functional trainer recipes."""

import unittest
from unittest.mock import MagicMock, patch

from mlx_vlm.trainer.core import TrainingArgs
from mlx_vlm.trainer.vlm.orpo.trainer import ORPOTrainer, ORPOTrainingArgs
from mlx_vlm.trainer.vlm.sft.trainer import SFTTrainer


class TrainerFacadeTest(unittest.TestCase):
    """Verify facades only configure and delegate; they do not own the loop."""

    @patch("mlx_vlm.trainer.vlm.sft.trainer.train")
    def test_sft_trainer_delegates_to_functional_recipe(self, train_mock):
        model = MagicMock()
        optimizer = MagicMock()
        train_dataset = MagicMock()
        validation_dataset = MagicMock()
        args = TrainingArgs(iters=3)

        SFTTrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            args=args,
            train_on_completions=True,
            assistant_id=42,
        ).fit()

        train_mock.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            args=args,
            train_on_completions=True,
            assistant_id=42,
        )

    @patch("mlx_vlm.trainer.vlm.orpo.trainer.train_orpo")
    def test_orpo_trainer_delegates_to_functional_recipe(self, train_mock):
        model = MagicMock()
        optimizer = MagicMock()
        train_dataset = MagicMock()
        validation_dataset = MagicMock()
        args = ORPOTrainingArgs(iters=3, beta=0.2)

        ORPOTrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            args=args,
            train_on_completions=True,
            assistant_id=42,
        ).fit()

        train_mock.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            args=args,
            train_on_completions=True,
            assistant_id=42,
        )


if __name__ == "__main__":
    unittest.main()
