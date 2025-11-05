import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.evals.utils import inference


def process_question(sample: dict) -> str:
    """Format the question."""
    return sample["question"]


def normalize_answer(response: str, problem: dict) -> Optional[str]:
    """Normalize the model's response to extract the answer."""
    if not response:
        return None
    return response.strip()


def evaluate_answer(prediction: Optional[str], ground_truth: list) -> bool:
    """Check if any ground truth answer is contained in the prediction."""
    if prediction is None:
        return False
    pred = prediction.strip().lower()
    return any(str(a).strip().lower() in pred for a in ground_truth)


def OCRBench_val(results_list, args, model_name, dataset="OCRBench"):
    correct = 0
    total = len(results_list)
    category_scores = {}
    for row in results_list:
        ground_truth = row["ground_truth"]
        if isinstance(ground_truth, str):

            ground_truth = [a.strip() for a in ground_truth.split(";")]
        prediction = row["prediction"]

        is_correct = evaluate_answer(prediction, ground_truth)
        row["correct"] = is_correct
        if is_correct:
            correct += 1
        category = row["type"]
        if category not in category_scores:
            category_scores[category] = {"correct": 0, "total": 0}
        category_scores[category]["total"] += 1
        if is_correct:
            category_scores[category]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{model_name}_{dataset}_{args.split}.csv"

    fieldnames = [
        "id",
        "question",
        "dataset",
        "type",
        "ground_truth",
        "response",
        "prediction",
        "correct",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_list:
            out_row = row.copy()
            if isinstance(out_row["ground_truth"], list):
                out_row["ground_truth"] = "; ".join(map(str, out_row["ground_truth"]))
            writer.writerow(out_row)

    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "category_scores": category_scores,
    }

    summary_file = output_dir / f"{model_name}_{dataset}_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"{dataset} Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {summary['model']}")
    print(f"Split: {args.split}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    if len(category_scores.items()) > 1:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on OCRBench benchmark"
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
        default="echo840/OCRBench",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
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
        "--predictions-file",
        type=str,
        default=None,
        help="File with predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ocrbench",
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

    if args.predictions_file:
        logging.info(
            f"\033[32mLoading predictions from {args.predictions_file} for evaluation\033[0m"
        )
        with open(args.predictions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            loaded_results = list(reader)
        model_name = Path(args.predictions_file).stem.split("_OCRBench")[0]
        dataset = (
            "OCRBench-v2" if "OCRBench-v2" in args.predictions_file else "OCRBench"
        )
        OCRBench_val(loaded_results, args, model_name, dataset)
        logging.info(f"\033[32mEvaluation complete\033[0m")
        return

    logging.info(f"Loading model from {args.model}")
    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    results = {}
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        pid = sample.get("id", str(idx))

        try:
            # Load and process image
            if "image" in sample and sample["image"]:
                image = sample["image"].convert("RGB")
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

            # Store results (evaluation happens later)
            results[pid] = {
                "id": pid,
                "question": sample["question"],
                "dataset": sample.get("dataset", ""),
                "type": sample.get("type", ""),
                "ground_truth": (
                    sample.get("answers", [])
                    if hasattr(sample, "answers")
                    else sample.get("answer", [])
                ),
                "response": response,
                "prediction": prediction,
                "correct": False,
            }

            if args.verbose:
                logging.info(f"\nSample {pid}:")
                logging.info(f"Question: {sample['question']}")
                logging.info(f"Response: {response}")
                logging.info(f"Prediction: {prediction}")
                logging.info(f"Ground Truth: {sample.get('answers', [])}")

        except Exception as e:
            logging.error(f"Error processing sample {pid}: {e}")
            continue

    results_list = list(results.values())
    model_name = args.model.split("/")[-1]
    dataset = args.dataset.split("/")[-1]
    OCRBench_val(results_list, args, model_name, dataset)


if __name__ == "__main__":
    main()
