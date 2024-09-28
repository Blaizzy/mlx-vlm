import argparse

import mlx.optimizers as optim

from mlx_vlm.trainer import Dataset, Trainer
from mlx_vlm.trainer.lora import *
from mlx_vlm.trainer.utils import *
from mlx_vlm.utils import load, load_image_processor


def add_image_token(items, image_token="<image>"):
    conversations = []
    for item in items["conversations"]:
        if item["role"] == "user":
            if item["content"].startswith(image_token):
                conversations.append({"role": "user", "content": item["content"]})
            else:
                conversations.append(
                    {"role": "user", "content": image_token + "\n" + item["content"]}
                )
        else:
            conversations.append({"role": "assistant", "content": item["content"]})
    return {"conversations": conversations}


def main(args):
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    image_processor = load_image_processor(args.model_path)

    dataset = Dataset(
        args.dataset,
        model.config.__dict__,
        processor,
        image_processor=image_processor,
        take=None,
        split=None,
    )
    dataset = dataset.map(add_image_token)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    trainer = Trainer(model, optimizer)

    list_of_modules = find_all_linear_names(model.language_model.model)
    model = get_peft_model(model, list_of_modules)

    model.vision_tower.freeze()
    model.train()

    for epoch in range(args.epochs):
        for i in range(args.steps):
            loss = trainer.train_step(
                dataset[i * args.batch_size : (i + 1) * args.batch_size]
            )
            if i % args.print_every == 0:
                print(f"Epoch {epoch} Step {i} Loss {loss}")

    save_adapter(model, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoLLaVA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mlx-community/nanoLLaVA-1.5-bf16",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--print_every", type=int, default=10, help="Print loss every n steps"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="nanollava_lora_adapter.safetensors",
        help="Path to save the trained adapter",
    )

    args = parser.parse_args()
    main(args)
