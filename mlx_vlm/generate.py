import argparse
import codecs

import mlx.core as mx

from .prompt_utils import get_message_json
from .utils import generate, get_model_path, load, load_config, load_image_processor

MODEL_TYPE = ""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qnguyen3/nanoLLaVA",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are these?",
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp", type=float, default=0.3, help="Temperature for sampling."
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Detailed output.",
        default=True,
    )
    return parser.parse_args()


def get_model_and_processors(model_path):
    model_path = get_model_path(model_path)
    config = load_config(model_path)
    model, processor = load(model_path, {"trust_remote_code": True})
    image_processor = load_image_processor(model_path)
    return model, processor, image_processor, config


def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def main():
    args = parse_arguments()
    model, processor, image_processor, config = get_model_and_processors(args.model)

    prompt = codecs.decode(args.prompt, "unicode_escape")

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            [get_message_json(config["model_type"], prompt)],
            tokenize=False,
            add_generation_prompt=True,
        )

    elif "tokenizer" in processor.__dict__.keys():
        if model.config.model_type != "paligemma":
            prompt = processor.tokenizer.apply_chat_template(
                [get_message_json(config["model_type"], prompt)],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )

    output = generate(
        model,
        processor,
        args.image,
        prompt,
        image_processor,
        args.temp,
        args.max_tokens,
        args.verbose,
    )
    if not args.verbose:
        print(output)


if __name__ == "__main__":
    main()
