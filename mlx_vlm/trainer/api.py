"""Notebook-first training API; loading models stays with their owning library."""

from collections.abc import Mapping
from dataclasses import asdict
from importlib import import_module
from types import SimpleNamespace

from mlx_vlm.trainer.common.config import CoreTrainingArgs
from mlx_vlm.trainer.common.task import TrainingTask
from mlx_vlm.trainer.vlm.config import VLMTrainingArgs
from mlx_vlm.trainer.vlm.dpo.config import DPOTrainingArgs
from mlx_vlm.trainer.vlm.orpo.config import ORPOTrainingArgs

# Explicit supported combinations, without importing optional modality libraries.
TASKS = {
    ("vlm", "sft"): ("vlm.sft.runtime", VLMTrainingArgs),
    ("vlm", "dpo"): ("vlm.dpo.trainer", DPOTrainingArgs),
    ("vlm", "orpo"): ("vlm.orpo.runtime", ORPOTrainingArgs),
}


def task_spec(task: str, algorithm: str | None = None):
    algorithm = algorithm or "sft"
    try:
        return algorithm, TASKS[task, algorithm]
    except KeyError:
        supported = ", ".join(f"{name}/{mode}" for name, mode in TASKS)
        raise ValueError(
            f"Unsupported task/algorithm {task}/{algorithm}. Supported: {supported}"
        ) from None


class Trainer:
    """Train/evaluate loaded MLX models from raw rows or prepared datasets.

    Provide ``processing_class`` for VLMs, and a separate frozen
    ``reference_model`` for DPO. Configure adapters/dtype on the model before
    construction. ``prepared=True`` skips preprocessing for the low-level API.
    A custom ``TrainingTask`` extends this API without editing the engine.
    """

    def __init__(
        self,
        model,
        args=None,
        *,
        task="vlm",
        algorithm=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        optimizer=None,
        reference_model=None,
        dataset_config=None,
        prepared=False,
        training_callback=None,
        seed=0,
    ):
        if isinstance(task, TrainingTask):
            args = args or CoreTrainingArgs()
            bound_task = task
        else:
            if algorithm is None:
                if isinstance(args, DPOTrainingArgs):
                    algorithm = "dpo"
                elif isinstance(args, ORPOTrainingArgs):
                    algorithm = "orpo"
            algorithm, (module_name, config_type) = task_spec(task, algorithm)
            args = args or config_type()
            if not isinstance(args, config_type):
                raise TypeError(f"{task}/{algorithm} expects {config_type.__name__}.")
            args.validate()
            if not prepared and processing_class is None:
                raise ValueError(f"{task} raw datasets require processing_class.")
            options = {}
            if algorithm == "dpo":
                options["reference_model"] = reference_model
            elif reference_model is not None:
                raise ValueError("reference_model is only used by DPO.")
            if task == "vlm" and processing_class is not None:
                tokenizer = getattr(processing_class, "tokenizer", processing_class)
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(tokenizer, "eos_token_id", None)
                if pad_id is not None:
                    from dataclasses import replace

                    args = replace(args, pad_token_id=int(pad_id))
            config = (
                dict(dataset_config)
                if isinstance(dataset_config, Mapping)
                else vars(dataset_config or SimpleNamespace())
            )
            config = SimpleNamespace(**(asdict(args) | config))
            module = import_module(f".{module_name}", __package__)
            bound_task = module.make_task(
                model, args, processing_class, config, **options
            )
        args.validate()
        self.model, self.args, self.task = model, args, bound_task
        self.optimizer = optimizer
        self.training_callback, self.seed, self.prepared = (
            training_callback,
            seed,
            prepared,
        )
        self.train_dataset = self._prepare(train_dataset)
        self.eval_dataset = self._prepare(eval_dataset)
        self.metrics = {}
        self.log_history = []

    def _prepare(self, dataset):
        if dataset is None:
            return None
        if isinstance(dataset, Mapping):
            raise TypeError("Pass one dataset split, such as dataset['train'].")
        if self.prepared or self.task.prepare_dataset is None:
            return dataset
        return self.task.prepare_dataset(dataset)

    def train(self) -> dict:
        """Train configured parameters and return final loss/throughput metrics."""
        import mlx.optimizers as optim

        from mlx_vlm.trainer.common.engine import train

        if self.train_dataset is None:
            raise ValueError("train() requires train_dataset.")
        if self.optimizer is None:
            cls = {
                "adam": optim.Adam,
                "adamw": optim.AdamW,
                "sgd": optim.SGD,
                "muon": optim.Muon,
            }[self.args.optimizer]
            options = (
                {"weight_decay": self.args.weight_decay}
                if self.args.optimizer in {"adamw", "muon"}
                else {}
            )
            self.optimizer = cls(learning_rate=self.args.learning_rate, **options)
        self.log_history.clear()
        self.metrics = train(
            self.model,
            self.optimizer,
            self.train_dataset,
            self.eval_dataset,
            self.args,
            self.task,
            self.training_callback,
            self.seed,
            log_history=self.log_history,
        )
        return self.metrics

    def evaluate(self, eval_dataset=None) -> dict:
        """Return weighted validation loss and restore the model's original mode."""
        from mlx_vlm.trainer.common.engine import evaluate

        dataset = (
            self.eval_dataset if eval_dataset is None else self._prepare(eval_dataset)
        )
        return evaluate(self.model, dataset, self.task, self.args, return_metrics=True)

    def save_adapter(self, path=None) -> None:
        """Save trainable weights on rank zero, including task-specific metadata."""
        import mlx.core as mx

        from mlx_vlm.trainer.common.utils import save_trainable_weights

        if mx.distributed.init().rank() == 0:
            save = self.task.save_weights or save_trainable_weights
            save(self.model, path or self.args.adapter_file)
