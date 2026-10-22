"""Thin shared CLI over the notebook Trainer API (python -m mlx_vlm.train)."""

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import get_args

from .trainer.api import Trainer, task_spec


def _read_config(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        if Path(path).suffix == ".json":
            values = json.load(file)
        else:
            import yaml

            values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError("Training config must contain a mapping.")
    return values


def build_parser(task="vlm", algorithm=None):
    """Generate training flags from the same dataclasses notebooks use."""
    algorithm, (_, config_type) = task_spec(task, algorithm)
    parser = argparse.ArgumentParser(description=f"MLX {task}/{algorithm} training")
    parser.add_argument("--task", choices=("vlm",), default=task)
    parser.add_argument(
        "--algorithm",
        "--train-mode",
        choices=("sft", "orpo", "dpo", "grpo"),
        default=algorithm,
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("--model", "--model-path", help="Model path or Hub identifier.")
    parser.add_argument(
        "--data", default="data", help="JSONL split folder, file, or Hub dataset."
    )
    parser.add_argument(
        "--dataset", help="Dataset loaded with Hugging Face datasets; overrides --data."
    )
    parser.add_argument(
        "--hf-dataset-config", help="Optional Hugging Face dataset configuration name."
    )
    parser.add_argument("--train-split", "--split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--epochs", type=int, help="Complete batch passes; overrides default iters."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer", choices=("adam", "adamw", "sgd", "muon"), default="adamw"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--train-type", choices=("lora", "dora", "full"), default="lora"
    )
    parser.add_argument(
        "--full-finetune", dest="train_type", action="store_const", const="full"
    )
    parser.add_argument(
        "--train-vision",
        action="store_true",
        help="Also unfreeze the vision tower/projector.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        help="Linear suffixes or qualified module names; auto-selects language transformer when omitted.",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=(4, 8),
        help="Optionally quantize the base before adapter training.",
    )
    parser.add_argument(
        "--quantization-group-size", type=int, choices=(32, 64, 128), default=64
    )
    parser.add_argument(
        "--resume-adapter-file",
        "--adapter-path",
        help="Load weights; optimizer state is not restored.",
    )
    parser.add_argument(
        "--output-path", help="Output file or folder; alias for --adapter-file."
    )
    parser.add_argument(
        "--reference-model", help="Frozen DPO/GRPO reference; defaults to --model."
    )
    if algorithm == "grpo":
        parser.add_argument(
            "--reward-functions",
            default="accuracy_reward",
            help="Comma-separated batched reward callback names.",
        )
        parser.add_argument(
            "--reward-functions-file",
            help="Python file defining custom reward callbacks.",
        )
        parser.add_argument(
            "--reward-weights", type=json.loads, help="JSON array of reward weights."
        )
    parser.add_argument("--wandb", help="Optional W&B project.")
    parser.add_argument(
        "--dataset-config",
        type=json.loads,
        default={},
        help="Preprocessing/column options as a JSON object.",
    )
    for item in fields(config_type):
        kwargs = {"default": item.default, "help": item.metadata.get("help")}
        if item.type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            types = get_args(item.type)
            if type(None) in types:
                converter = next(t for t in types if t is not type(None))
                kwargs["type"] = lambda value, converter=converter: (
                    None if value.lower() == "none" else converter(value)
                )
            else:
                kwargs["type"] = item.type
        parser.add_argument("--" + item.name.replace("_", "-"), **kwargs)
    return parser


def parse_args(argv=None, *, task="vlm"):
    # Select the dataclass before full parsing, including task/algorithm in a config.
    probe = argparse.ArgumentParser(add_help=False)
    for name in ("task", "algorithm", "config"):
        probe.add_argument("--" + name)
    probe.add_argument("-c", dest="config")
    probe.add_argument("--train-mode", dest="algorithm")
    probe.add_argument("--iters", type=int)
    early, _ = probe.parse_known_args(argv)
    config = _read_config(early.config)
    task = early.task or config.get("task", task)
    algorithm = early.algorithm or config.get("algorithm")
    try:
        parser = build_parser(task, algorithm)
    except ValueError as error:
        probe.error(str(error))
    unknown = set(config) - {action.dest for action in parser._actions}
    if unknown:
        parser.error("Unknown config options: " + ", ".join(sorted(unknown)))
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required.")
    if args.output_path:
        destination = Path(args.output_path)
        args.adapter_file = str(
            destination
            if destination.suffix == ".safetensors"
            else destination / "adapters.safetensors"
        )
    if not isinstance(args.dataset_config, dict):
        parser.error("--dataset-config must be a JSON object.")
    if args.reference_model and args.algorithm not in {"dpo", "grpo"}:
        parser.error("--reference-model is only used by DPO or GRPO.")
    if args.optimizer not in {"adam", "adamw", "sgd", "muon"}:
        parser.error("optimizer must be adam, adamw, sgd, or muon.")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error(
            "learning_rate must be positive and weight_decay nonnegative and finite."
        )
    _, (_, config_type) = task_spec(args.task, args.algorithm)
    settings = {item.name: getattr(args, item.name) for item in fields(config_type)}
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("epochs must be at least 1.")
        if early.iters is not None or config.get("iters") is not None:
            parser.error("Set either iters or epochs, not both.")
    try:
        config_type(**settings).validate()
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return args, config_type(**settings)


def _load_model(task, path):
    """Keep upstream loaders here so future mlx-omni integration changes one place."""
    from mlx_vlm import load

    return load(path)


def _configure_model(model, args):
    from .trainer.common.model import prepare_model_for_training

    model = prepare_model_for_training(
        model,
        train_type=args.train_type,
        lora_rank=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0),
        target_modules=getattr(args, "target_modules", None),
        quantization_bits=getattr(args, "quantization_bits", None),
        quantization_group_size=getattr(args, "quantization_group_size", 64),
        checkpoint_path=getattr(args, "resume_adapter_file", None),
    )
    if getattr(args, "train_vision", False):
        from .trainer.peft import unfreeze_modules

        unfreeze_modules(
            model,
            [
                "vision_model",
                "vision_tower",
                "mm_projector",
                "multi_modal_projector",
                "aligner",
                "connector",
                "vision_resampler",
            ],
        )
    return model


def _load_dataset(args):
    if args.dataset:
        from datasets import load_dataset

        return load_dataset(args.dataset, args.hf_dataset_config)
    from .trainer.datasets.loading import load

    return load(args.data)


def run(args, settings, training_callback=None):
    """Load resources and call the same Trainer used in notebooks."""
    import mlx.core as mx
    import mlx.optimizers as optim

    from mlx_vlm.trainer.common.callbacks import WandBCallback

    mx.random.seed(args.seed)
    model, processor = _load_model(args.task, args.model)
    reference = None
    if args.algorithm == "dpo" or (args.algorithm == "grpo" and settings.beta):
        reference, _ = _load_model(args.task, args.reference_model or args.model)
    raw = _load_dataset(args)
    if isinstance(raw, Mapping):
        train_rows = raw.get(args.train_split, [])
        val_rows = raw.get(args.validation_split)
        if val_rows is None and args.validation_split == "validation":
            val_rows = raw.get("valid")
    else:
        train_rows, val_rows = raw, None
    if len(train_rows) == 0:
        raise ValueError(f"Dataset has no non-empty {args.train_split!r} split.")
    model = _configure_model(model, args)
    reward_options = {}
    if args.algorithm == "grpo":
        from .trainer.vlm.grpo import rewards

        if args.reward_functions_file:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "vlm_custom_rewards", args.reward_functions_file
            )
            rewards = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rewards)
        reward_options = {
            "reward_funcs": [
                getattr(rewards, name.strip())
                for name in args.reward_functions.split(",")
            ],
            "reward_weights": args.reward_weights,
        }
    optimizer_type = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "muon": optim.Muon,
    }[args.optimizer]
    optimizer_options = (
        {"weight_decay": args.weight_decay}
        if args.optimizer in {"adamw", "muon"}
        else {}
    )
    optimizer = optimizer_type(learning_rate=args.learning_rate, **optimizer_options)
    trainer = Trainer(
        model,
        settings,
        optimizer=optimizer,
        task=args.task,
        algorithm=args.algorithm,
        train_dataset=train_rows,
        eval_dataset=val_rows if val_rows is not None and len(val_rows) else None,
        processing_class=processor,
        reference_model=reference,
        dataset_config=args.dataset_config,
        seed=args.seed,
        training_callback=training_callback,
        **reward_options,
    )
    if args.epochs is not None:
        signature = getattr(trainer.train_dataset, "media_signature", None)
        if args.algorithm == "grpo":
            # GRPO scores rows independently and accepts mixed text/image batches.
            signature = None
        counts = Counter(
            signature(i) if signature else () for i in range(len(train_rows))
        )
        batches = sum(count // settings.batch_size for count in counts.values())
        if not batches:
            raise ValueError("No complete batches fit the dataset; reduce batch_size.")
        trainer.args.iters = args.epochs * batches
    callback = None
    try:
        if args.wandb and mx.distributed.init().rank() == 0:
            callback = WandBCallback(
                args.wandb,
                str(Path(settings.adapter_file).parent),
                vars(args) | asdict(trainer.args),
                training_callback,
            )
            trainer.training_callback = callback
        return trainer.train()
    finally:
        if callback is not None:
            callback.finish()


def main(argv=None, *, task="vlm"):
    args, settings = parse_args(argv, task=task)
    run(args, settings)


if __name__ == "__main__":
    main()
