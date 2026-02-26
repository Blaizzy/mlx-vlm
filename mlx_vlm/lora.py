import argparse
import json
import logging

import mlx.optimizers as optim
from datasets import load_dataset

from .trainer import (
    Colors,
    TrainingArgs,
    VisionDataset,
    print_trainable_parameters,
    train,
)
from .trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
    not_supported_for_training,
    unfreeze_modules,
)
from .utils import load, load_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def transform_dataset_to_messages(dataset, model_type, custom_prompt_format=None):
    """
    Only transform dataset to messages format for VLMs with single-turn QA and image columns.
    If the dataset already has a 'messages' column, return as is.
    Otherwise, require 'question' and 'answer' columns.
    If present, 'image' or 'images' columns are included in the user content.
    No multi-turn or template logic. No audio support.
    """
    has_messages = (
        "messages" in dataset.column_names or "conversations" in dataset.column_names
    )
    has_qa = "question" in dataset.column_names and "answer" in dataset.column_names
    has_images = "images" in dataset.column_names or "image" in dataset.column_names

    if has_messages:
        return dataset

    if not has_qa:
        raise ValueError(
            "Dataset must have a 'messages' column or both 'question' and 'answer' columns. Optional image columns: 'image' or 'images'."
        )

    image_col = "images" if "images" in dataset.column_names else "image" if has_images else None

    def to_message(example):
        q = example["question"]
        a = example["answer"]
        img = example[image_col] if image_col else None
        if custom_prompt_format:
            try:
                template = json.loads(custom_prompt_format)

                def fill(node):
                    if isinstance(node, str):
                        try:
                            return node.format(image=img, question=q, answer=a)
                        except Exception:
                            return node
                    if isinstance(node, dict):
                        return {k: fill(v) for k, v in node.items()}
                    if isinstance(node, list):
                        return [fill(v) for v in node]
                    return node

                filled = fill(template)
                return {"messages": filled}
            except Exception as e:
                raise ValueError(f"Failed to parse or fill custom prompt format: {e}")

        # VLM-specific message formats (fallback)
        vlm_message_model_prefixes = ["gemma", "qwen", "smolvlm", "mllama", "mistral3"]
        if model_type and any(
            model_type.startswith(prefix) for prefix in vlm_message_model_prefixes
        ):
            user_content = []
            if img is not None:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": q})
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": a}]},
                ]
            }
        elif model_type == "deepseek_vl_v2" and img is not None:
            return {
                "messages": [
                    {
                        "role": "<|User|>",
                        "content": f"<image>\n<|ref|>{q}<|/ref|>.",
                        "images": [img],
                    },
                    {"role": "<|Assistant|>", "content": a},
                ]
            }
        else:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"<image>{q}"
                            if img is not None and "<image>" not in str(q)
                            else q
                        ),
                    },
                    {"role": "assistant", "content": a},
                ]
            }

    return dataset.map(
        to_message,
    )


def setup_model_for_training(model, args, adapter_path=None):
    """Setup model with LoRA or full finetuning"""

    if adapter_path:
        logger.info(
            f"{Colors.UNDERLINE}Resuming from adapter path {adapter_path}{Colors.ENDC}"
        )
        model = apply_lora_layers(model, adapter_path)
    elif args.full_finetune:
        logger.info(
            f"{Colors.UNDERLINE}Training with full weight finetuning{Colors.ENDC}"
        )
        unfreeze_modules(model, ["language_model"])
    else:
        logger.info(f"{Colors.UNDERLINE}Setting up LoRA{Colors.ENDC}")
        modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model,
            modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=False,
        )

    if args.train_vision:
        logger.info(f"{Colors.OKBLUE}Unfreezing vision stack for training{Colors.ENDC}")
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


def main(args):
    # Load model and processor
    logger.info(f"{Colors.HEADER}Loading model from {args.model_path}{Colors.ENDC}")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )

    # Validate model type
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in not_supported_for_training:
        raise ValueError(f"Model type {model_type} not supported for training")

    config = model.config.__dict__
    image_processor = load_image_processor(args.model_path)

    # Load and prepare dataset
    logger.info(f"{Colors.HEADER}Loading dataset from {args.dataset}{Colors.ENDC}")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config if args.dataset_config else None,
        split=args.split,
    )

    # Calculate training iterations
    if args.epochs is not None:
        iters = (len(dataset) // args.batch_size) * args.epochs
    else:
        iters = args.iters

    dataset = dataset.select(range(iters))

    # Transform dataset to messages format (support custom prompt template)
    dataset = transform_dataset_to_messages(
        dataset, model_type, args.custom_prompt_format
    )

    # Create training dataset
    train_dataset = VisionDataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape,
    )

    # Setup model for training
    model = setup_model_for_training(model, args, args.adapter_path)
    print_trainable_parameters(model)

    # Setup optimizer
    logger.info(f"{Colors.HEADER}Setting up optimizer{Colors.ENDC}")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    # Create training arguments
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        val_batches=args.val_batches,
        max_seq_length=args.max_seq_length,
        adapter_file=args.output_path,
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        full_finetune=args.full_finetune,
    )

    # Train model
    logger.info(f"{Colors.HEADER}Training model{Colors.ENDC}")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        train_on_completions=args.train_on_completions,
        assistant_id=args.assistant_id,
    )

    logger.info(
        f"{Colors.HEADER}Training completed! Model saved to {args.output_path}{Colors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision-Language Model")

    # Model arguments
    parser.add_argument(
        "--model-path", type=str, default="mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--train-vision", action="store_true")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument(
        "--custom-prompt-format",
        type=str,
        default=None,
        help='Custom JSON prompt template. Example: {"user": [{"type": "image","image":"{image}"}, {"type": "text","text":"{question}"}], "assistant": [{"type": "text","text":"{answer}"}]}',
    )

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--steps-per-save", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-on-completions", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--assistant-id", type=int, default=77091)

    # LoRA arguments
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    # Output arguments
    parser.add_argument("--output-path", type=str, default="adapters.safetensors")
    parser.add_argument("--adapter-path", type=str, default=None)

    args = parser.parse_args()
    main(args)
