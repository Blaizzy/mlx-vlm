import argparse
import json
import logging

import mlx.optimizers as optim
from datasets import load_dataset

from .prompt_utils import apply_chat_template
from .trainer import Dataset, TrainingArgs, Colors, train, print_trainable_parameters
from .trainer.utils import apply_lora_layers, find_all_linear_names, get_peft_model
from .utils import load, load_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    logger.info(f"{Colors.HEADER}Loading model from {args.model_path}{Colors.ENDC}")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    
    unsupported_for_training = {"lfm2-vl", "", ""}
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in unsupported_for_training:
        raise ValueError(
            f"{Colors.FAIL}Model type {model_type} not supported for training. "
            "Please choose a different model or remove it from the unsupported list.{Colors.ENDC}"
        )
    config = model.config.__dict__
    image_processor = load_image_processor(args.model_path)
    
    logger.info(f"{Colors.HEADER}Loading dataset from {args.dataset}{Colors.ENDC}")
    if args.dataset_config is not None:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    # Validate image columns
    if not ("images" in dataset.column_names or "image" in dataset.column_names):
        raise ValueError(f"{Colors.FAIL}Dataset must have either an 'images' or 'image' column{Colors.ENDC}")
    
    # Validate message columns
    if "messages" not in dataset.column_names:
        if "question" in dataset.column_names and "answer" in dataset.column_names:
            def transform_to_messages(examples):
                messages_list = []
                for q, a in zip(examples["question"], examples["answer"]):
                    messages_list.append([
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ])
                examples["messages"] = messages_list
                return examples
            dataset = dataset.map(transform_to_messages, batched=True)
        else:
            raise ValueError(f"{Colors.FAIL}Dataset must have a 'messages' column or both 'question' and 'answer' columns{Colors.ENDC}")
    
    if args.apply_chat_template:
        logger.info(f"{Colors.OKBLUE}Applying chat template to the dataset{Colors.ENDC}")
        
        def process_data(examples):
            if config["model_type"] == "pixtral":
                conversations = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
                examples["messages"] = [
                    json.dumps(item, ensure_ascii=False) for item in conversations
                ]
            else:
                examples["messages"] = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
            return examples
        
        dataset = dataset.map(process_data)
    
    # Create Dataset objects
    train_dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape,
    )
    
    # Use train dataset for validation if no validation dataset is provided
    val_dataset = train_dataset
    
    adapter_path = args.adapter_path
    if adapter_path:
        logger.info(f"{Colors.UNDERLINE}Resuming from adapter path {adapter_path}{Colors.ENDC}")
        logger.info(
            f"\033[32mLora rank, alpha, and dropout will be loaded from adapter_config.json file{Colors.ENDC}"
        )
        
        model = apply_lora_layers(model, adapter_path)
    
    elif args.full_finetune:
        logger.info(f"{Colors.UNDERLINE}Training with full weight finetuning{Colors.ENDC}")
        print_trainable_parameters(model)
    else:
        logger.info(f"{Colors.UNDERLINE}Setting up LoRA{Colors.ENDC}")
        
        list_of_modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model,
            list_of_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    
    logger.info(f"{Colors.HEADER}Setting up optimizer{Colors.ENDC}")
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Create TrainingArgs
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        val_batches=args.val_batches,
        max_seq_length=args.max_seq_length,
        adapter_file=args.output_path,
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
    )
    
    # Train the model
    logger.info(f"{Colors.HEADER}Training model{Colors.ENDC}")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        train_on_completions=args.train_on_completions,
        assistant_id=args.assistant_id,
    )
    
    logger.info(f"{Colors.HEADER}Training completed! Model saved to {args.output_path}{Colors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision-Language Model")
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train the model with full weight finetuning instead of LoRA",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for training"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset configuration name",
    )
    parser.add_argument(
        "--image-resize-shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize images to this shape",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=True,
        help="Apply chat template to the dataset",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-7,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Number of iterations to train for"
    )
    parser.add_argument(
        "--steps-per-report", type=int, default=10, help="Number of training steps between loss reporting"
    )
    parser.add_argument(
        "--steps-per-eval", type=int, default=200, help="Number of training steps between validations"
    )
    parser.add_argument(
        "--steps-per-save", type=int, default=100, help="Save the model every number of steps"
    )
    parser.add_argument(
        "--val-batches", type=int, default=25, help="Number of validation batches, -1 uses entire validation set"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--grad-checkpoint", action="store_true", help="Use gradient checkpointing to reduce memory use"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=None, help="Gradient clipping value"
    )
    parser.add_argument(
        "--train-on-completions", action="store_true", help="Train only on assistant responses"
    )
    parser.add_argument(
        "--assistant-id", type=int, default=77091, help="Token ID for assistant responses (used with train-on-completions)"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16,
        help="LoRA scaling factor",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Output arguments
    parser.add_argument(
        "--output-path",
        type=str,
        default="adapters.safetensors",
        help="Path to save the trained adapter",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Load path to resume training from a previously saved adapter",
    )
    
    args = parser.parse_args()
    main(args)