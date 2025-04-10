try:
    import wandb
except ImportError:
    wandb = None


class TrainingCallback:
    def on_train_loss_report(self, train_info: dict):
        pass

    def on_val_loss_report(self, val_info: dict):
        pass


class WandBCallback(TrainingCallback):
    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        wrapped_callback: TrainingCallback = None,
    ):
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it to use WandBCallback."
            )
        self.wrapped_callback = wrapped_callback
        wandb.init(project=project_name, dir=log_dir, config=config)

    def on_train_loss_report(self, train_info: dict):
        wandb.log(train_info)
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        wandb.log(val_info)
        if self.wrapped_callback:
            self.wrapped_callback.on_val_loss_report(val_info)
