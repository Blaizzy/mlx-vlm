import argparse
import csv
import json
import logging
import random
import traceback
from pathlib import Path
from typing import Optional

import mlx.core as mx
from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.evals.utils import inference
from mlx_vlm.generate import batch_generate
from mlx_vlm.sample_utils import top_p_sampling


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (1 = sequential, >1 = batch generation)",
    )
    return parser.parse_args()


def create_sampler(temperature: float, top_p: float = 1.0):
    """Create a sampler function for batch generation.

    For accuracy consistency across batch sizes, we use deterministic sampling
    (temperature=0) by default. This ensures the same outputs regardless of batch size.
    """

    def sampler(logits: mx.array) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                return top_p_sampling(logits, top_p, temperature)
            else:
                return mx.random.categorical(logits * (1 / temperature))

    return sampler


def process_batch(
    model,
    processor,
    batch_samples,
    args,
):
    """Process a batch of samples using batch_generate.

    batch_generate now handles image size sorting internally to minimize
    padding effects and maintain accuracy.
    """
    prompts = []
    images = []
    sample_metadata = []

    for sample in batch_samples:
        pid = sample.get("id", str(sample.get("_idx", 0)))

        # Load and process image
        if "image" in sample and sample["image"]:
            image = sample["image"].convert("RGB")
        else:
            logging.warning(f"No image for sample {pid}, skipping")
            continue

        images.append(image)

        # Create prompt
        prompt = process_question(sample)
        prompts.append(prompt)

        # Store metadata for results
        sample_metadata.append(
            {
                "id": pid,
                "question": sample["question"],
                "dataset": sample.get("dataset", ""),
                "type": sample.get("type", ""),
                "ground_truth": (
                    sample.get("answers", [])
                    if hasattr(sample, "answers")
                    else sample.get("answer", [])
                ),
            }
        )

    if not prompts:
        return []

    # Create sampler for deterministic output (temperature=0 by default)
    sampler = create_sampler(args.temperature)

    # Use batch_generate for processing
    # batch_generate now handles image size sorting internally to avoid padding issues
    batch_response = batch_generate(
        model,
        processor,
        images=images,
        prompts=prompts,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=args.verbose,
    )

    # Process results
    results = []
    for text, metadata in zip(batch_response.texts, sample_metadata):
        response = text.strip()
        prediction = normalize_answer(response, {"question": metadata["question"]})

        result = {
            **metadata,
            "response": response,
            "prediction": prediction,
            "correct": False,
        }
        results.append(result)

        if args.verbose:
            logging.info(f"\nSample {metadata['id']}:")
            logging.info(f"Question: {metadata['question']}")
            logging.info(f"Response: {response}")
            logging.info(f"Prediction: {prediction}")
            logging.info(f"Ground Truth: {metadata['ground_truth']}")

    return results


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

    # Convert to list for batching if streaming
    if args.streaming:
        dataset = list(dataset)

    results = {}
    batch_size = args.batch_size

    if batch_size > 1:
        # Batch generation mode
        logging.info(f"Using batch generation with batch_size={batch_size}")

        # Collect samples into batches
        batch = []
        all_samples = list(dataset) if hasattr(dataset, "__iter__") else dataset

        # Add index to samples for tracking
        for idx, sample in enumerate(all_samples):
            sample["_idx"] = idx

        for idx, sample in enumerate(
            tqdm(all_samples, desc=f"Evaluating (batch_size={batch_size})")
        ):
            batch.append(sample)

            # Process batch when full or at the end
            if len(batch) >= batch_size or idx == len(all_samples) - 1:
                try:
                    batch_results = process_batch(model, processor, batch, args)
                    for result in batch_results:
                        results[result["id"]] = result
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    traceback.print_exc()

                batch = []

                # Clear memory after each batch
                mx.clear_cache()

    else:
        # Sequential generation mode (original behavior)
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
                traceback.print_exc()
                logging.error(f"Error processing sample {pid}: {e}")
                continue

    results_list = list(results.values())
    model_name = args.model.split("/")[-1]
    dataset = args.dataset.split("/")[-1]
    OCRBench_val(results_list, args, model_name, dataset)


if __name__ == "__main__":
    main()
