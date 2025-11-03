import argparse
import csv
import os
import random
import re
from json import dump
from logging import getLogger

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

logger = getLogger(__name__)

# All 30 MMMU subjects (confirmed from dataset)
MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]


def normalize_number(s):
    """Normalize numeric strings for comparison."""
    try:
        return float(str(s).strip().replace(",", ""))
    except:
        return str(s).strip()


def MMMU_eval(data: list, eval_file: str):
    """
    Evaluate MMMU results by subject.
    Handles both multiple choice (A-F) and open-ended questions.
    """

    # Track by subject
    subject_scores = {}
    subject_counters = {}

    total_correct = 0
    total_questions = 0

    for line in data:
        predict = str(line["prediction"])
        answer = str(line["answer"])
        subject = str(line.get("subject", "Unknown"))

        # Initialize subject tracking if needed
        if subject not in subject_scores:
            subject_scores[subject] = 0
            subject_counters[subject] = 0

        # Count this question
        subject_counters[subject] += 1
        total_questions += 1

        # Normalize for comparison
        predict_lower = predict.lower().strip()
        answer_lower = answer.lower().strip()

        is_correct = False

        # Check if this is a multiple choice question (answer is A-F or I)
        if answer in ["A", "B", "C", "D", "E", "F", "I"]:
            # Multiple choice extraction with prioritized patterns
            patterns = [
                (r"option\s+([a-f])\b", 10),  # High priority
                (r"answer\s+is:?\s+([a-f])\b", 10),
                (r"choice\s+is:?\s+([a-f])\b", 10),
                (r"correct\s+answer\s+is:?\s+([a-f])\b", 10),
                (r"correct\s+option\s+is:?\s+\(?([a-f])\)?", 10),
                (r"\(([a-f])\)", 8),  # Medium priority
                (r"^([a-f])[.:\)]\s", 8),
                (r"\b([a-f])\b", 5),  # Low priority - isolated letters
            ]

            best_match = None
            best_priority = -1

            # Try each pattern, keeping the highest priority match
            for pattern, priority in patterns:
                matches = re.findall(pattern, predict_lower, re.IGNORECASE)
                if matches and priority > best_priority:
                    best_match = matches[0].lower()
                    best_priority = priority
                    # Stop early if we found a high-confidence pattern
                    if priority >= 10:
                        break

            # Check if match is correct
            if best_match and best_match == answer_lower:
                is_correct = True
            # Fallback: check first character
            elif (
                not best_match
                and len(predict_lower) > 0
                and predict_lower[0] in "abcdef"
            ):
                if predict_lower[0] == answer_lower:
                    is_correct = True

        else:
            # Open-ended question - check if answer appears in prediction
            # Exact substring match (case-insensitive)
            if answer_lower in predict_lower:
                is_correct = True
            # For numeric answers, try numeric comparison
            elif answer.replace(".", "").replace("-", "").replace(",", "").isdigit():
                numbers = re.findall(r"-?\d+\.?\d*", predict)
                answer_num = normalize_number(answer)
                for num_str in numbers:
                    try:
                        if abs(normalize_number(num_str) - answer_num) < 0.01:
                            is_correct = True
                            break
                    except:
                        pass
            # Word-level match for text answers
            else:
                answer_words = set(answer_lower.split())
                predict_words = set(predict_lower.split())
                if answer_words and answer_words.issubset(predict_words):
                    is_correct = True

        if is_correct:
            total_correct += 1
            subject_scores[subject] += 1
            line["score"] = 1
        else:
            line["score"] = 0

    # Calculate final scores
    results = {}
    results["overall_accuracy"] = (
        float(total_correct) / float(total_questions) if total_questions > 0 else 0.0
    )
    results["total_correct"] = total_correct
    results["total_questions"] = total_questions

    # Calculate subject scores
    for subject in sorted(subject_scores.keys()):
        if subject_counters[subject] > 0:
            results[f"subject_{subject}_accuracy"] = float(
                subject_scores[subject]
            ) / float(subject_counters[subject])
            results[f"subject_{subject}_correct"] = subject_scores[subject]
            results[f"subject_{subject}_total"] = subject_counters[subject]

    # Print scores
    print("\nMMMU Evaluation Results:")
    print("=" * 80)
    print(
        f"Overall Accuracy: {results['overall_accuracy']:.4f} ({total_correct}/{total_questions})"
    )
    print("=" * 80)
    print("Subject Breakdown:")
    for subject in sorted(subject_scores.keys()):
        acc = results.get(f"subject_{subject}_accuracy", 0.0)
        correct = results.get(f"subject_{subject}_correct", 0)
        total = results.get(f"subject_{subject}_total", 0)
        print(f"  {subject}: {acc:.4f} ({correct}/{total})")
    print("=" * 80)

    # Save results
    score_pth = eval_file.replace(".csv", "_score.json")
    with open(score_pth, "w") as f:
        dump(results, f, indent=2)

    with open(eval_file, "w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    logger.info(
        f"MMMU_eval successfully finished evaluating {eval_file}, results saved in {score_pth}"
    )


def process_question(example):
    """
    Process MMMU question to format it properly.
    MMMU questions may have options and images.
    """
    question = example.get("question", "")
    # Remove <image n> tags from the question
    question = re.sub(r"<image \d+>", "", question).strip()

    # Add options if they exist
    options = example.get("options", None)
    options = re.sub(r'[\[\]"\']', "", options).split(", ") if options else None

    if options and isinstance(options, list):
        question += "\n\nOptions:"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, ...
            question += f"\n{letter}. {option}"

    explanation = example.get("explanation", None)
    if explanation:
        question += f"\n\nExplanation: {explanation}"

    return question


def get_images(example):
    """
    Extract images from MMMU example.
    MMMU can have multiple images per question.
    """
    images = []

    # MMMU dataset may have image_1, image_2, etc.
    for i in range(1, 8):  # Check up to 7 images
        img_key = f"image_{i}"
        if img_key in example and example[img_key] is not None:
            try:
                img = example[img_key].convert("RGB")
                images.append(img)
            except:
                pass

    return images


def inference(
    model,
    processor,
    question,
    image,
    max_tokens=3000,
    temperature=0.0,
    resize_shape=None,
    verbose=False,
):
    """Run inference on a single question."""
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
    return response.text


def list_subjects():
    """Print all available MMMU subjects."""
    print("\n" + "=" * 80)
    print("MMMU Available Subjects (30 total)")
    print("=" * 80)
    for i, subject in enumerate(MMMU_SUBJECTS, 1):
        print(f"{i:2d}. {subject}")
    print("=" * 80 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MMMU Evaluation - Massive Multi-discipline Multimodal Understanding",
        epilog="Use --subset to evaluate a specific subject, or omit to evaluate all 30 subjects.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Model path",
    )
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")
    parser.add_argument("--dataset", type=str, default="MMMU/MMMU", help="Dataset path")
    parser.add_argument(
        "--split", type=str, default="validation", help="Split to use for evaluation"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=f"Subset to use - one of 30 subjects: {', '.join(MMMU_SUBJECTS[:5])}... (see SUBJECTS.md for full list)",
    )
    parser.add_argument(
        "--streaming", action="store_false", help="Use streaming dataset loading"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0.0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty parameter",
    )
    parser.add_argument(
        "--resize_shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize shape for the image",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all 30 available subjects and exit",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help="Path to the prediction file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_arguments()

    random.seed(args.seed)

    if args.prediction_file:
        # Load predictions from file
        logger.info(f"\033[32mLoading predictions from {args.prediction_file}\033[0m")
        results = []
        with open(args.prediction_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        # Evaluate loaded predictions
        MMMU_eval(results, args.prediction_file)
        logger.info(f"\033[32mEvaluation complete\033[0m")
        return

    # Handle --list-subjects flag
    if args.list_subjects:
        list_subjects()
        return

    logger.info("\033[32mStarting MMMU Evaluation\033[0m")

    # Validate subset if provided
    if args.subset and args.subset not in MMMU_SUBJECTS:
        logger.error(f"\033[31mError: Invalid subset '{args.subset}'\033[0m")
        logger.error(f"\033[31mValid subjects are: {', '.join(MMMU_SUBJECTS)}\033[0m")
        logger.error(f"\033[31mSee SUBJECTS.md for more details\033[0m")
        return

    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")

    # Load dataset
    if args.subset:
        logger.info(f"\033[32mUsing subset: {args.subset}\033[0m")
        datasets = {
            args.subset: load_dataset(
                args.dataset, args.subset, split=args.split, streaming=args.streaming
            )
        }
        subset_name = args.subset
    else:
        logger.info(f"\033[32mEvaluating all 30 subjects\033[0m")
        datasets = {}
        for subject in MMMU_SUBJECTS:
            datasets[subject] = load_dataset(
                args.dataset, name=subject, split=args.split, streaming=args.streaming
            )
        subset_name = "all"

    # Limit samples if specified
    if args.max_samples:
        datasets = {
            k: v.select(range(min(args.max_samples, len(v))))
            for k, v in datasets.items()
        }
        logger.info(f"\033[33mLimited to {len(datasets)} samples for testing\033[0m")

    logger.info(f"\033[32mDataset subset size: {len(datasets.keys())}\033[0m")
    logger.info(f"\033[32mLoading model from {args.model}\033[0m")

    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )
    config = model.config
    logger.info(f"\033[32mConfig: {config}\033[0m")

    # Create results directory
    model_name = args.model.split("/")[-1]
    result_file = (
        f"./results/{model_name}_MMMU_{subset_name}_{args.split}_predictions.csv"
    )
    os.makedirs("./results/", exist_ok=True)

    results = []
    for subject, dataset in tqdm(datasets.items(), desc="Processing subjects"):
        for idx, example in enumerate(tqdm(dataset, desc=f"Processing {subject}")):
            question = process_question(example)
            images = get_images(example)

            # Get prediction
            prediction = inference(
                model,
                processor,
                question,
                images,
                args.max_tokens,
                args.temperature,
                args.resize_shape,
                args.verbose,
            )

            # Store result
            result = {
                "id": example.get("id", idx),
                "question": question,
                "answer": example.get("answer", ""),
                "subfield": example.get("subfield", "Unknown"),
                "topic_difficulty": example.get("topic_difficulty", "Unknown"),
                "question_type": example.get("question_type", "Unknown"),
                "prediction": prediction,
                "subject": subject,
            }
            results.append(result)

            # Show progress
            if (idx + 1) % 10 == 0 or idx < 5:
                logger.info(
                    f"Sample {idx + 1}: Answer={result['answer']}, Prediction={prediction[:50]}..."
                )

    # Print first few results
    print("\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(
            f"{i+1}. Question: {result['question'][:50]}... | Answer: {result['answer']} | Prediction: {result['prediction'][:50]}..."
        )

    # Save results to CSV
    with open(result_file, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"\033[32mSaved results to {result_file}\033[0m")

    # Evaluate results
    MMMU_eval(results, result_file)

    logger.info(f"\033[32mEvaluation complete\033[0m")


if __name__ == "__main__":
    main()
