import argparse
import csv
import logging
import os
import random
import re
from copy import deepcopy
from json import dump

from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.evals.utils import inference


def extract_answer(predict, answer):
    """
    Extracts the answer from the model's predictions.
    predict: Model prediction text
    answer: Ground truth answer (A, B, C, or D)
    Returns: bool: True if answer matches, False otherwise
    """
    text = predict.lower().replace("\n", " ").strip()
    answer_lower = answer.lower()

    general_templates = [
        r"^{0}\b",
        r"^\({0}",
        r"^option {0}\b",
        r"\b{0}\s*[:\.\)]",
        r"(?:^|\.|\s)\s*{0}\.",
        r"\({0}\)",
        r"option\s+{0}\b",
        r"choice\s+{0}\b",
    ]

    concluding_templates = [
        r"^the answer is {0}\b",
        r"answer:\s*{0}\b",
        r"answer\s+is\s+{0}\b",
        r"correct\s+(?:answer|option|choice)\s+is:?\s+{0}\b",
        r"the\s+answer\s+is\s+{0}\b",
        r"is\s+{0}\s*:",
        r"(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?{0}\b",
        r"(?:select|choose)\s+{0}\b",
        r"it\s+is\s+{0}\b",
        r"would\s+be\s+{0}\b",
        r"\*\*(?:revised\s+)?answer\*\*:\s*{0}\b",
        r"(?:correct\s+)?category\s+(?:for\s+this\s+image\s+)?is\s+\*\*{0}[:\s]",
    ]

    possible_answers = ["a", "b", "c", "d", "e"]
    matches = []

    for ans in possible_answers:
        for pri, template_list in [(2, concluding_templates), (1, general_templates)]:
            for template in template_list:
                pattern = template.format(ans)
                for match in re.finditer(pattern, text):
                    matches.append((match.end(), ans, pri))

    if not matches:
        return False

    # Sort ascending by (-priority, -end_position) to prefer higher priority first, then latest position
    matches.sort(key=lambda m: (-m[2], -m[0]))
    latest_ans = matches[0][1]

    return latest_ans == answer_lower


def MMStar_eval(data: list, eval_file: str):
    MMStar_score_l2 = {
        "coarse perception": {
            "image scene and topic": 0,
            "image style & quality": 0,
            "image emotion": 0,
        },
        "fine-grained perception": {
            "object counting": 0,
            "recognition": 0,
            "localization": 0,
        },
        "instance reasoning": {
            "single-instance reasoning": 0,
            "cross-instance attribute reasoning": 0,
            "cross-instance relation reasoning": 0,
        },
        "logical reasoning": {
            "code & sequence reasoning": 0,
            "diagram reasoning": 0,
            "common reasoning": 0,
        },
        "science & technology": {
            "biology & chemistry & physics": 0,
            "electronics & energy & mechanical eng.": 0,
            "geography & earth science & agriculture": 0,
        },
        "math": {
            "geometry": 0,
            "numeric commonsense and calculation": 0,
            "statistical reasoning": 0,
        },
    }
    MMStar_counter = deepcopy(MMStar_score_l2)

    for line in tqdm(data, desc="Evaluating"):
        predict = str(line["prediction"])
        answers = str(line["answer"])
        category = str(line["category"])
        l2_category = str(line["l2_category"])

        MMStar_counter[category][l2_category] += 1

        # Use comprehensive extraction
        if extract_answer(predict, answers):
            MMStar_score_l2[category][l2_category] += 1

            line["score"] = 1
        else:
            line["score"] = 0

    # Calculate scores
    MMStar_score = {}
    MMStar_score["final score"] = 0
    total_correct = 0

    for k, v in MMStar_score_l2.items():
        cat_total = sum(MMStar_counter[k].values())
        cat_correct = 0
        for l2_k, l2_v in v.items():
            count = MMStar_counter[k][l2_k]
            if count > 0:
                MMStar_score[f"{k}({l2_k})"] = float(l2_v) / float(count)
            else:
                MMStar_score[f"{k}({l2_k})"] = 0.0
            cat_correct += l2_v
            total_correct += l2_v
        MMStar_score[k] = float(cat_correct) / cat_total if cat_total > 0 else 0.0
        MMStar_score["final score"] += cat_correct

    if len(data) > 0:
        MMStar_score["final score"] = float(MMStar_score["final score"]) / float(
            len(data)
        )

    # Print results
    print("\n" + "=" * 80)
    print("MMStar Evaluation Results")
    print("=" * 80)
    print(
        f"\nFinal Score: {total_correct}/{len(data)} = {MMStar_score['final score']*100:.2f}%\n"
    )

    print("-" * 80)
    print("Category Scores:")
    print("-" * 80)
    for category in [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "science & technology",
        "math",
    ]:
        if category in MMStar_score:
            cat_total = sum(MMStar_counter[category].values())
            cat_correct = sum(MMStar_score_l2[category].values())
            print(
                f"{category:30s}: {cat_correct:4d}/{cat_total:4d} = {MMStar_score[category]*100:6.2f}%"
            )

    print("\n" + "-" * 80)
    print("Subcategory Scores:")
    print("-" * 80)
    for category in [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "science & technology",
        "math",
    ]:
        print(f"\n{category.upper()}:")
        for l2_cat, score in MMStar_score_l2[category].items():
            count = MMStar_counter[category][l2_cat]
            pct = (score / count * 100) if count > 0 else 0
            print(f"  {l2_cat:55s}: {score:4d}/{count:4d} = {pct:6.2f}%")

    print("\n" + "=" * 80)

    # Save scores
    score_pth = eval_file.replace(".csv", "_score.json")
    with open(score_pth, "w") as f:
        dump(MMStar_score, f, indent=2)

    with open(eval_file, "w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def parse_arguments():
    parser = argparse.ArgumentParser(description="MMStar Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Model path",
    )
    parser.add_argument("--adapter-path", type=str, default=None, help="Adapter path")
    parser.add_argument(
        "--dataset", type=str, default="Lin-Chen/MMStar", help="Dataset path"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Split to use for evaluation"
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming dataset loading"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize shape for the image",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--prediction-file", type=str, default=None, help="Path to the prediction file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mmstar",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_arguments()

    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("\033[32mStarting MMStar Evaluation\033[0m")
    if args.prediction_file:
        logging.info(
            f"\033[32mLoading predictions from {args.prediction_file} for evaluation\033[0m"
        )
        results = []
        with open(args.prediction_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]
        MMStar_eval(results, args.prediction_file)
        logging.info(f"\033[32mEvaluation complete\033[0m")
        return
    logging.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    logging.info(f"\033[32mLoading model from {args.model}\033[0m")
    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )
    config = model.config
    logging.info(f"\033[32mConfig: {config}\033[0m")

    result_file = f'{args.output_dir}/{args.model.split("/")[-1]}_{args.dataset.split("/")[-1]}_{args.split}_predictions.csv'
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for example in tqdm(dataset, desc="Running inference"):
        question = example["question"]
        image = example["image"].convert("RGB")
        prediction = inference(
            model,
            processor,
            question,
            image,
            args.max_tokens,
            args.temperature,
            args.resize_shape,
            args.verbose,
        )

        results.append(
            {
                "question": question,
                "answer": example["answer"],
                "category": example["category"],
                "l2_category": example["l2_category"],
                "meta_info": example["meta_info"],
                "prediction": prediction,
            }
        )

    print("\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(
            f"{i+1}. Question: {result['question'][:50]}... | Answer: {result['answer']} | Prediction: {result['prediction'][:50]}..."
        )

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    MMStar_eval(results, result_file)

    logging.info(f"\033[32mSaving results to {result_file}\033[0m")
    logging.info(f"\033[32mEvaluation complete\033[0m")


if __name__ == "__main__":
    main()
