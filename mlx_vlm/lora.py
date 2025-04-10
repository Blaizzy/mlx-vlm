import argparse
import logging

import mlx.optimizers as optim
from tqdm import tqdm

from .trainers import TrainingArgs, TrainingCallback, save_adapter, save_full_model, train
from .trainers.dataset import load_and_prepare_dataset
from .trainers.utils import get_peft_model, print_trainable_parameters
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

    dataset = load_and_prepare_dataset(
        config=config,
        args=args,
        processor=processor,
        image_processor=image_processor,
    )

    if args.full_weight_training:
        logger.info(f"\033[32mUsing full weight training (all parameters will be trained)\033[0m")
    else:
        logger.info(f"\033[32mSetting up LoRA\033[0m")
        model = get_peft_model(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )

    print_trainable_parameters(model)

    logger.info(f"\033[32mSetting up optimizer\033[0m")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info(f"\033[32mSetting up training arguments\033[0m")
    
    # Convert existing args to TrainingArgs
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.steps,
        steps_per_report=args.print_every,
        adapter_file=args.output_path
    )
    
    # Create a custom TrainingCallback for tqdm progress bar
    class CustomTrainingCallback(TrainingCallback):
        def __init__(self, total_iters):
            self.progress_bar = tqdm(total=total_iters, position=0, leave=True)
            
        def on_train_loss_report(self, train_info):
            self.progress_bar.update(train_info["iteration"] - self.progress_bar.n)
            self.progress_bar.set_postfix({
                "Loss": f"{train_info['train_loss']:.4f}",
                "LR": f"{train_info['learning_rate']:.2e}",
                "Tokens/sec": f"{train_info['tokens_per_second']:.0f}"
            })
            
            # Log additional information
            custom_print({
                "Step": train_info["iteration"],
                "Loss": f"{train_info['train_loss']:.4f}",
                "Learning Rate": f"{train_info['learning_rate']:.2e}",
                "Tokens/sec": f"{train_info['tokens_per_second']:.0f}"
            })
            
        def on_val_loss_report(self, val_info):
            # Log validation results
            custom_print({
                "Step": val_info["iteration"],
                "Val Loss": f"{val_info['val_loss']:.4f}",
                "Val Time": f"{val_info['val_time']:.2f}s"
            })
    
    model.train()
    
    logger.info(f"\033[32mTraining model\033[0m")
    callback = CustomTrainingCallback(training_args.iters)
    
    # Use the functional train approach instead of manual loop
    train(
        model=model,
        tokenizer=processor,
        optimizer=optimizer,
        dataset=dataset,
        args=training_args,
        training_callback=callback,
        train_on_completions=getattr(args, 'train_on_completions', False),
        assistant_id=getattr(args, 'assistant_id', 77091),
        clip_gradients=getattr(args, 'clip_gradients', None)
    )
    
    # Save model weights
    if args.full_weight_training:
        save_full_model(model, args.output_path)
    else:
        save_adapter(model, args.output_path)
    
    logger.info(f"\033[32mTraining complete. Model saved to {args.output_path}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoLLaVA model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--full-weight-training",
        action="store_true",
        help="Enable full weight training instead of LoRA. When this flag is set, all LoRA settings (lora-alpha, lora-rank, lora-dropout) will be ignored."
    )
    parser.add_argument(
        "--training-type",
        action="store_true",
        help="Enable full weight training instead of LoRA. When this flag is set, all LoRA settings (lora-alpha, lora-rank, lora-dropout) will be ignored."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for training"
    )
    parser.add_argument(
        "--dataset-config", type=str, default=None, help="Use a individual configuration from the dataset"
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
        "--steps", type=int, default=1000, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--print-every", type=int, default=10, help="Print loss every n steps"
    )
    parser.add_argument(
        "--wandb_project", type=int, default=0.1, help="LoRA alpha parameter"
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
