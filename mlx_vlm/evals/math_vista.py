import argparse
import csv
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.evals.utils import inference


def process_question(sample: dict) -> str:
    """Format the question with choices if it's multiple choice."""
    question = sample["query"]

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
        # First, try to find boxed answers
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Check if it's a choice letter
            letter_match = re.match(
                r"^\(?([A-Z])\)?\.?$", boxed_content.strip().upper()
            )
            if letter_match:
                letter = letter_match.group(1)
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]
            # Check if it's directly one of the choices
            if boxed_content.strip() in choices:
                return boxed_content.strip()

        # Try to find Chinese answer pattern "故选：X" or "故选X"
        chinese_match = re.search(r"故选[：:]\s*([A-Z])", response.upper())
        if not chinese_match:
            chinese_match = re.search(r"故选\s*([A-Z])", response.upper())
        if chinese_match:
            letter = chinese_match.group(1)
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx]

        # Try to find "the answer is X" or "answer: X" patterns near the end
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+\(?([A-Z])\)?",
            r"answer:\s*\(?([A-Z])\)?",
            r"choose\s+\(?([A-Z])\)?",
            r"option\s+\(?([A-Z])\)?",
        ]

        # Search from the end of the response (last 500 chars)
        end_section = response[-500:] if len(response) > 500 else response
        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                # Take the last match
                letter = matches[-1].group(1).upper()
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]

        # Look for patterns like "(A)", "A)", "A.", "A" - prioritize from the end
        matches = list(re.finditer(r"\(?([A-Z])\)?\.?", response.upper()))
        if matches:
            # Try the last few matches first
            for match in reversed(matches[-5:]):
                letter = match.group(1)
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]

        # If the response is exactly one of the choices
        if response in choices:
            return response

        # Try to find the most similar choice using edit distance
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
        # First try to find boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Remove commas from numbers
            boxed_content = boxed_content.replace(",", "")
            # Try scientific notation first
            sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", boxed_content)
            if sci_numbers:
                try:
                    return str(int(float(sci_numbers[0])))
                except:
                    pass
            # Then regular numbers
            numbers = re.findall(r"-?\d+", boxed_content)
            if numbers:
                try:
                    return str(int(numbers[0]))
                except:
                    pass

        # Try common answer patterns near the end
        end_section = response[-500:] if len(response) > 500 else response
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
            r"answer:\s*(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
            r"(?:total|result|left|remaining)(?:\s+is|\s+are|:)\s*(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
        ]

        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                try:
                    # Remove commas before converting
                    num_str = matches[-1].group(1).replace(",", "")
                    return str(int(float(num_str)))
                except:
                    pass

        # Look for scientific notation anywhere in response
        sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", response)
        if sci_numbers:
            try:
                return str(int(float(sci_numbers[-1])))
            except:
                pass

        # Fall back to finding all numbers (including comma-formatted) and taking the last one
        # Match numbers with optional commas: 7,518 or 7518
        numbers = re.findall(r"-?[\d,]+", response)
        if numbers:
            try:
                # Remove commas and try the last number first
                return str(int(numbers[-1].replace(",", "")))
            except:
                pass

    # For float answers
    elif answer_type == "float":
        precision = int(problem.get("precision", 2))

        # First try to find boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Try scientific notation first
            sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", boxed_content)
            if sci_numbers:
                try:
                    return str(round(float(sci_numbers[0]), precision))
                except:
                    pass
            # Then regular numbers
            numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
            if numbers:
                try:
                    return str(round(float(numbers[0]), precision))
                except:
                    pass

        # Try common answer patterns near the end
        end_section = response[-500:] if len(response) > 500 else response
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",
            r"answer:\s*(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",
            r"d\s*=\s*(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",  # For physics problems with d=
        ]

        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                try:
                    return str(round(float(matches[-1].group(1)), precision))
                except:
                    pass

        # Look for scientific notation anywhere in response
        sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", response)
        if sci_numbers:
            try:
                return str(round(float(sci_numbers[-1]), precision))
            except:
                pass

        # Fall back to finding all numbers and taking the last one
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            try:
                # Try the last number first
                return str(round(float(numbers[-1]), precision))
            except:
                pass

    return response


def evaluate_answer(prediction: Optional[str], ground_truth: str) -> bool:
    """Check if the prediction matches the ground truth."""
    if prediction is None:
        return False
    try:
        # First check exact match
        if str(prediction).strip() == str(ground_truth).strip():
            return True

        # Handle numeric word representations
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        pred_normalized = str(prediction).strip().lower()
        gt_normalized = str(ground_truth).strip().lower()

        # Convert words to numbers
        if pred_normalized in word_to_num:
            pred_normalized = word_to_num[pred_normalized]
        if gt_normalized in word_to_num:
            gt_normalized = word_to_num[gt_normalized]

        return pred_normalized == gt_normalized
    except:
        return False


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
        "--dataset",
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
        "--streaming",
        action="store_true",
        help="Use streaming dataset loading",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)

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
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    category_scores = {}
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
            )

            response = output.strip()

            # Normalize answer
            prediction = normalize_answer(response, sample)

            # Evaluate
            ground_truth = sample.get("answer", "")
            if args.split == "testmini" and ground_truth:
                is_correct = evaluate_answer(prediction, ground_truth)
                if is_correct:
                    correct += 1
            else:
                is_correct = None

            total += 1

            # Store results
            results[pid] = {
                "pid": pid,
                "question": sample["question"],
                "query": sample["query"],
                "question_type": sample["question_type"],
                "answer_type": sample["answer_type"],
                "choices": sample.get("choices", []),
                "unit": sample.get("unit", ""),
                "precision": sample.get("precision", 0),
                "ground_truth": ground_truth,
                "response": response,
                "prediction": prediction,
                "correct": is_correct,
                "metadata": sample.get("metadata", {}),
            }
            # Track category-wise performance
            category = sample.get("metadata", {}).get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = {"correct": 0, "total": 0}

            category_scores[category]["total"] += 1
            if is_correct:
                category_scores[category]["correct"] += 1

            if args.verbose:
                logging.info(f"\nSample {pid}:")
                logging.info(f"Question: {sample['question']}")
                logging.info(f"Response: {response}")
                logging.info(f"Prediction: {prediction}")
                logging.info(f"Ground Truth: {ground_truth}")
                logging.info(f"Correct: {is_correct}")

        except Exception as e:
            logging.error(f"Error processing sample {pid}: {e}")
            continue

    # Calculate accuracy if applicable
    if args.split == "testmini":
        accuracy = correct / total if total > 0 else 0
    else:
        accuracy = None
        correct = None

    # Save results
    model_name = args.model.split("/")[-1]
    results_file = output_dir / f"{model_name}_MathVista_{args.split}.csv"

    # Convert results to list of dictionaries for CSV writing
    fieldnames = [
        "pid",
        "question",
        "query",
        "question_type",
        "answer_type",
        "choices",
        "unit",
        "precision",
        "ground_truth",
        "response",
        "prediction",
        "correct",
        "metadata",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results.values():
            # Convert list and dict fields to strings for CSV
            row = result.copy()
            if isinstance(row.get("choices"), list):
                row["choices"] = "; ".join(row["choices"])
            if isinstance(row.get("metadata"), dict):
                row["metadata"] = json.dumps(row["metadata"])
            writer.writerow(row)

    # Save summary
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": total,
        "category_scores": category_scores,
    }

    if accuracy is not None:
        summary["correct"] = correct
        summary["accuracy"] = accuracy

    summary_file = output_dir / f"{model_name}_MathVista_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("MathVista Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Total Samples: {total}")
    if accuracy is not None:
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")
    else:
        print("Accuracy not computed for this split (no ground truth labels)")

    print("\n" + "-" * 80)
    print(f"Subcategory Scores:")
    print(f"{'-'*80}")
    for category, scores in category_scores.items():
        cat_total = scores["total"]
        cat_correct = scores["correct"]
        cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
        print(f"  {category}: {cat_correct}/{cat_total} ({cat_accuracy*100:.2f}%)")
    print(f"{'='*80}")
    print(f"\nResults saved to {results_file} and {summary_file}")


if __name__ == "__main__":
    main()
