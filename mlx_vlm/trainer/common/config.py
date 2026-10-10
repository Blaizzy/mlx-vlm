"""Training schedule and optimizer settings shared by all modalities."""

from dataclasses import dataclass, field


@dataclass
class CoreTrainingArgs:
    """Common settings inherited by pipeline-specific training arguments."""

    batch_size: int = field(default=1, metadata={"help": "Global minibatch size."})
    iters: int = field(default=100, metadata={"help": "Number of training iterations."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of microbatches per optimizer update."}
    )
    val_batches: int = field(
        default=25,
        metadata={"help": "Validation batches; -1 uses the full validation set."},
    )
    steps_per_report: int = field(
        default=10, metadata={"help": "Training iterations between reports."}
    )
    steps_per_eval: int | None = field(
        default=200, metadata={"help": "Training iterations between validation runs."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Training iterations between checkpoints."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Output path for trainable weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Checkpoint model layers to reduce memory use."},
    )
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    compile: bool = False
    clear_cache_threshold: int = 0
    cache_size: int | None = 32

    def validate(self) -> None:
        """Validate common settings before allocating training resources."""
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "steps_per_report",
            "steps_per_save",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.iters < 0:
            raise ValueError("iters must be nonnegative.")
        if self.steps_per_eval is not None and self.steps_per_eval < 1:
            raise ValueError("steps_per_eval must be at least 1 or None.")
        if self.val_batches != -1 and self.val_batches < 1:
            raise ValueError("val_batches must be at least 1 or -1.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative."
            )
        if self.optimizer not in {"adam", "adamw", "sgd", "muon"}:
            raise ValueError("optimizer must be adam, adamw, sgd, or muon.")
        if self.clear_cache_threshold < 0:
            raise ValueError("clear_cache_threshold must be nonnegative (bytes).")
        if self.cache_size is not None and self.cache_size < 0:
            raise ValueError(
                "cache_size must be nonnegative or None for unlimited caching."
            )
