import argparse
import logging
import warnings
import json
import mlx.core as mx
import mlx.optimizers as optim
from datasets import load_dataset

from .trainer import TrainingArgs, Colors, train, print_trainable_parameters
from .trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
    unfreeze_modules,
    supported_for_training
)
from .utils import load, load_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif "tokenizer" in processor.__dict__.keys():
        prompt = processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    return prompt


class VisionDataset():
    """Simplified dataset class for Vision LLMs"""
    def __init__(self, hf_dataset, config, processor, image_processor=None,
                 image_resize_shape=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape
        
    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")
    
    def __getitem__(self, idx):
        return self.process(self.dataset[idx])
    
    def process(self, item):
        """Process a single item from the dataset"""
        from mlx_vlm.utils import prepare_inputs
        
        # Handle images
        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []
        
        # Handle audio
        audio = item.get("audio", item.get("audios", []))
        if not isinstance(audio, list):
            audio = [audio] if audio else []
        
        # Get conversations
        conversations = item.get("messages", item.get("conversations"))

        model_type = self.config.get("model_type")
        
        prompts = []
        # Format prompt based on model type
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if model_type == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )

                prompt = get_prompt(
                    model_type, self.processor, conversation
                )
                prompts.append(prompt)

        else:
            if model_type == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompt = get_prompt(
                model_type, self.processor, conversations
            )
            prompts.append(prompt)
        
        # Prepare inputs
        image_token_index = (self.config.get("image_token_index") or
                           self.config.get("image_token_id"))
        if not image_token_index:
            raise ValueError("Config must contain 'image_token_index' or 'image_token_id'")
        
        inputs = prepare_inputs(
            processor=self.processor,
            images=None if images else None,
            audio=audio if audio else None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        
        return {
            "pixel_values": inputs.get("pixel_values"),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", mx.ones_like(inputs["input_ids"])),
            **{k: v for k, v in inputs.items()
               if k not in ["input_ids", "pixel_values", "attention_mask"]}
        }


def transform_dataset_to_messages(dataset, model_type, custom_prompt_format=None):
    """
    Only transform dataset to messages format for VLMs with single-turn QA and image columns.
    If the dataset already has a 'messages' column, return as is.
    Otherwise, require 'question', 'answer', and 'image' or 'images' columns.
    No multi-turn or template logic. No audio support.
    """
    has_messages = "messages" in dataset.column_names or "conversations" in dataset.column_names
    has_qa = "question" in dataset.column_names and "answer" in dataset.column_names
    has_images = "images" in dataset.column_names or "image" in dataset.column_names

    if has_messages:
        return dataset

    if not (has_qa and has_images):
        raise ValueError("Dataset must have 'messages' column or both 'question' and 'answer' columns and an 'image' or 'images' column.")

    image_col = "images" if "images" in dataset.column_names else "image"

    def to_message(example):
        q = example["question"]
        a = example["answer"]
        img = example[image_col] if has_images else None
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
        if model_type == "gemma3" or model_type.startswith("qwen") or model_type == "smolvlm":
            return {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": q}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": a}
                    ]}
                ]
            }
        elif model_type == "deepseek_vl_v2":
            return {
                "messages": [
                    {"role": "<|User|>", "content": f"<image>\n<|ref|>{q}<|/ref|>.", "images": [img]},
                    {"role": "<|Assistant|>", "content": a}
                ]
            }
        else:
            return {
                "messages": [
                    {"role": "user", "content": f"<image>{q}" if "<image>" not in str(q) else q},
                    {"role": "assistant", "content": a}
                ]
            }

    return dataset.map(to_message, )


def setup_model_for_training(model, args, adapter_path=None):
    """Setup model with LoRA or full finetuning"""
    
    if adapter_path:
        logger.info(f"{Colors.UNDERLINE}Resuming from adapter path {adapter_path}{Colors.ENDC}")
        model = apply_lora_layers(model, adapter_path)
    elif args.full_finetune:
        logger.info(f"{Colors.UNDERLINE}Training with full weight finetuning{Colors.ENDC}")
        unfreeze_modules(model, ["language_model"])
    else:
        logger.info(f"{Colors.UNDERLINE}Setting up LoRA{Colors.ENDC}")
        modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model, modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=False
        )
    
    if args.train_vision:
        logger.info(f"{Colors.OKBLUE}Unfreezing vision stack for training{Colors.ENDC}")
        unfreeze_modules(model, [
            "vision_model", "vision_tower", "mm_projector",
            "multi_modal_projector", "aligner", "connector", "vision_resampler"
        ])
    
    return model


def main(args):
    # Load model and processor
    logger.info(f"{Colors.HEADER}Loading model from {args.model_path}{Colors.ENDC}")
    model, processor = load(args.model_path, processor_config={"trust_remote_code": True})
    
    # Validate model type
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in supported_for_training:
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
    dataset = transform_dataset_to_messages(dataset, model_type, args.custom_prompt_format)
    
    # Create training dataset
    train_dataset = VisionDataset(
        dataset, config, processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape
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
    
    logger.info(f"{Colors.HEADER}Training completed! Model saved to {args.output_path}{Colors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision-Language Model")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2-VL-2B-Instruct-bf16")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument("--custom-prompt-format", type=str, default=None,
                        help='Custom JSON prompt template. Example: {"user": [{"type": "image","image":"{image}"}, {"type": "text","text":"{question}"}], "assistant": [{"type": "text","text":"{answer}"}]}')
    
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