"""Metric callbacks shared by all modalities."""

from pathlib import Path


class TrainingCallback:
    """Receive metric dictionaries with an optional integer ``iteration``.

    Trainers should report already evaluated values on rank zero. Metric names
    are passed through, so pipelines can include their own additional metrics.
    """

    def on_train_loss_report(self, train_info: dict) -> None:
        """Receive training metrics at the trainer's reporting interval."""

    def on_val_loss_report(self, val_info: dict) -> None:
        """Receive validation metrics after evaluation."""


class WandBCallback(TrainingCallback):
    """Send metrics to an optional W&B run and forward to another callback."""

    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        wrapped_callback: TrainingCallback | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B reporting requires wandb. Install it with: pip install wandb"
            ) from error
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.wrapped_callback = wrapped_callback
        self._run = wandb.init(
            project=project_name,
            name=directory.resolve().name,
            dir=str(directory),
            config=config,
        )

    def _log(self, info: dict) -> None:
        data = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in info.items()
        }
        self._run.log(data, step=data.get("iteration"))

    def on_train_loss_report(self, train_info: dict) -> None:
        """Send training metrics and forward the original report."""
        self._log(train_info)
        if self.wrapped_callback is not None:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        """Send validation metrics and forward the original report."""
        self._log(val_info)
        if self.wrapped_callback is not None:
            self.wrapped_callback.on_val_loss_report(val_info)

    def finish(self) -> None:
        """Flush and close the run when its owning pipeline finishes."""
        self._run.finish()
