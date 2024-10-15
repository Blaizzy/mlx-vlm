import argparse
import json
import logging

import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm

from .prompt_utils import apply_chat_template
from .trainer import Dataset, Trainer, save_adapter
from .trainer.utils import find_all_linear_names, get_peft_model
from .utils import load, load_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def main(args):
    logger.info(f"\033[32mLoading model from {args.model_path}\033[0m")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    image_processor = load_image_processor(args.model_path)

    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split)

    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must have a 'messages' column")
    if "images" not in dataset.column_names:
        raise ValueError("Dataset must have an 'images' column")

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying chat template to the dataset\033[0m")

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

    dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape,
    )

    logger.info(f"\033[32mSetting up LoRA\033[0m")
    list_of_modules = find_all_linear_names(model.language_model)
    model = get_peft_model(
        model,
        list_of_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    logger.info(f"\033[32mSetting up optimizer\033[0m")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info(f"\033[32mSetting up trainer\033[0m")
    trainer = Trainer(model, optimizer)

    model.train()

    # Training loop
    logger.info(f"\033[32mTraining model\033[0m")
    for epoch in range(args.epochs):
        if args.steps == 0:
            args.steps = len(dataset) // args.batch_size

        progress_bar = tqdm(range(args.steps), position=0, leave=True)
        for i in progress_bar:
            loss = trainer.train_step(
                dataset[i * args.batch_size : (i + 1) * args.batch_size]
            )
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Epoch": epoch, "Step": i, "Loss": f"{loss.item():.4f}"}
            )

            if i % args.print_every == 0:
                # Log additional information
                custom_print(
                    {
                        "Epoch": epoch,
                        "Step": i,
                        "Loss": f"{loss.item():.4f}",
                    }
                )

    # Save the adapter
    save_adapter(model, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoLLaVA model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for training"
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
        action="store_false",
        help="Apply chat template to the dataset",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--print-every", type=int, default=10, help="Print loss every n steps"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=0.1, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora-rank", type=int, default=10, help="LoRA rank")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--output-path",
        type=str,
        default="adapters",
        help="Path to save the trained adapter",
    )

    args = parser.parse_args()
    main(args)
