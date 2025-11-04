import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from mlx_vlm.generate import generate

from ..prompt_utils import apply_chat_template
from ..utils import load, load_config, prepare_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on MathVista benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the MLX VLM model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="AI4Math/MathVista",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testmini",
        choices=["testmini", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mathvista",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for debugging",
    )
    return parser.parse_args()


def process_question(sample: dict) -> str:
    """Format the question with choices if it's multiple choice."""
    question = sample["question"]

    if sample["question_type"] == "multi_choice" and sample["choices"]:
        choices_text = "\n".join(
            [f"({chr(65+i)}) {choice}" for i, choice in enumerate(sample["choices"])]
        )
        question = f"{question}\n{choices_text}"

    return question


def normalize_answer(response: str, problem: dict) -> Optional[str]:
    """Normalize the model's response to extract the answer."""
    response = response.strip()

    if not response:
        return None

    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem.get("choices", [])

    # For multiple choice, try to extract the letter
    if question_type == "multi_choice":
        # Look for patterns like "(A)", "A)", "A.", "A", etc.
        import re

        match = re.search(r"\(?([A-Z])\)?\.?", response.upper())
        if match:
            letter = match.group(1)
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx]

        # If the response is exactly one of the choices
        if response in choices:
            return response

        # Try to find the most similar choice
        def distance(s1, s2):
            if len(s1) < len(s2):
                return distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        if choices:
            distances = [
                distance(response.lower(), choice.lower()) for choice in choices
            ]
            return choices[distances.index(min(distances))]

    # For integer answers
    elif answer_type == "integer":
        import re

        numbers = re.findall(r"-?\d+", response)
        if numbers:
            try:
                return str(int(numbers[0]))
            except:
                pass

    # For float answers
    elif answer_type == "float":
        import re

        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            try:
                precision = int(problem.get("precision", 2))
                return str(round(float(numbers[0]), precision))
            except:
                pass

    return response


def evaluate_answer(prediction: Optional[str], ground_truth: str) -> bool:
    """Check if the prediction matches the ground truth."""
    if prediction is None:
        return False

    try:
        return str(prediction).strip() == str(ground_truth).strip()
    except:
        return False


def inference(
    model,
    processor,
    question,
    image,
    max_tokens=3000,
    temperature=0.7,
    resize_shape=None,
    verbose=False,
):
    # Check if image is a list or a single image
    if image is None:
        num_images = 0
    elif isinstance(image, list):
        num_images = len(image)
    else:
        num_images = 1

    prompt = apply_chat_template(
        processor, model.config, question, num_images=num_images
    )

    response = generate(
        model,
        processor,
        prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=temperature,
        resize_shape=resize_shape,
        verbose=verbose,
    )
    return response


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Loading model from {args.model}")
    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )

    # Load dataset
    logging.info(f"Loading dataset {args.dataset_name}, split {args.split}")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logging.info(f"Evaluating on {len(dataset)} samples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    correct = 0
    total = 0

    # Evaluate each sample
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        pid = sample["pid"]

        try:
            # Load and process image
            if "decoded_image" in sample and sample["decoded_image"]:
                if isinstance(sample["decoded_image"], str):
                    image_path = sample["decoded_image"]
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert("RGB")
                    else:
                        logging.warning(
                            f"Image not found: {image_path}, skipping sample {pid}"
                        )
                        continue
                else:
                    # Image is already loaded
                    image = sample["decoded_image"].convert("RGB")
            else:
                logging.warning(f"No image for sample {pid}, skipping")
                continue

            # Create prompt
            prompt = process_question(sample)

            # Generate response
            output = inference(
                model,
                processor,
                prompt,
                image=image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ).text

            response = output.strip()

            # Normalize answer
            prediction = normalize_answer(response, sample)

            # Evaluate
            is_correct = evaluate_answer(prediction, sample["answer"])

            if is_correct:
                correct += 1
            total += 1

            # Store results
            results[pid] = {
                "pid": pid,
                "question": sample["question"],
                "question_type": sample["question_type"],
                "answer_type": sample["answer_type"],
                "choices": sample.get("choices", []),
                "unit": sample.get("unit", ""),
                "precision": sample.get("precision", 0),
                "ground_truth": sample["answer"],
                "response": response,
                "prediction": prediction,
                "correct": is_correct,
                "metadata": sample.get("metadata", {}),
            }

            if args.verbose:
                logging.info(f"\nSample {pid}:")
                logging.info(f"Question: {sample['question']}")
                logging.info(f"Response: {response}")
                logging.info(f"Prediction: {prediction}")
                logging.info(f"Ground Truth: {sample['answer']}")
                logging.info(f"Correct: {is_correct}")

        except Exception as e:
            logging.error(f"Error processing sample {pid}: {e}")
            continue

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    # Save results
    results_file = output_dir / f"results_{args.split}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary = {
        "model": args.model,
        "dataset": args.dataset_name,
        "split": args.split,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    summary_file = output_dir / f"summary_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"MathVista Evaluation Results")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*50}")
    print(f"\nResults saved to {results_file} and {summary_file}")


if __name__ == "__main__":
    main()
