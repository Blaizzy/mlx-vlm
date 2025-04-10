from tqdm import tqdm

def custom_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)

try:
    import wandb
except ImportError:
    wandb = None


class TrainingCallback:
    def on_train_loss_report(self, train_info: dict):
        pass

    def on_val_loss_report(self, val_info: dict):
        pass


class CustomTrainingCallback(TrainingCallback):
        def __init__(self, total_iters):
            self.progress_bar = tqdm(total=total_iters, position=0, leave=True)
            
        def on_train_loss_report(self, train_info):
            self.progress_bar.update(train_info["iteration"] - self.progress_bar.n)

            # Use dynamic keys from train_info for logging and progress bar
            postfix = {}
            log_info = {"Step": train_info["iteration"]}
            for key, value in train_info.items():
                if key == "iteration":
                    continue
                try:
                    if isinstance(value, float):
                        log_info[key] = f"{value:.4f}" if abs(value) >= 1e-3 else f"{value:.2e}"
                        postfix[key] = log_info[key]
                    elif isinstance(value, int):
                        log_info[key] = str(value)
                        postfix[key] = log_info[key]
                except Exception:
                    continue

            self.progress_bar.set_postfix(postfix)
            custom_print(log_info)
            
        def on_val_loss_report(self, val_info):
            log_info = {"Step": val_info.get("iteration", "N/A")}
            for key, value in val_info.items():
                if key == "iteration":
                    continue
                try:
                    if isinstance(value, float):
                        log_info[key] = f"{value:.4f}" if abs(value) >= 1e-3 else f"{value:.2e}"
                    elif isinstance(value, int):
                        log_info[key] = str(value)
                except Exception:
                    continue

            custom_print(log_info)


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
