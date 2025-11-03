import argparse
import csv
import os
import re
from copy import deepcopy
from json import dump
from logging import getLogger

from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

logger = getLogger(__name__)


def extract_answer(predict, answer):
    """
    Extracts the answer from the model's predictions.
        predict: Model prediction text
        answer: Ground truth answer (A, B, C, or D)

    Returns:
        bool: True if answer matches, False otherwise
    """
    answer_upper = answer.upper()
    answer_lower = answer.lower()
    pred_lower = predict.lower().strip().replace("\n", " ")

    # STRATEGY 1: Strict positional matching (from original evaluation)
    try:
        # Check if first character matches
        if len(pred_lower) > 0 and answer_lower == pred_lower[0]:
            return True
        # Check if starts with "(" and second character matches
        if (
            len(pred_lower) > 1
            and pred_lower[0] == "("
            and answer_lower == pred_lower[1]
        ):
            return True
        # Check if starts with "option " and 8th character matches
        if (
            len(pred_lower) >= 8
            and pred_lower[0:7] == "option "
            and answer_lower == pred_lower[7]
        ):
            return True
        # Check if starts with "the answer is " and 15th character matches
        if (
            len(pred_lower) >= 15
            and pred_lower[0:14] == "the answer is "
            and answer_lower == pred_lower[14]
        ):
            return True
    except:
        pass

    # STRATEGY 2: Flexible pattern matching for common answer formats
    patterns = [
        # Direct answer markers
        rf"\b{answer_upper}\s*:",  # "A:"
        rf"(?:^|\.|\n)\s*{answer_upper}\.",  # "A." at sentence start
        rf"\({answer_upper}\)",  # "(A)"
        # Explicit answer statements
        rf"answer\s+is\s+{answer_upper}\b",  # "answer is A"
        rf"correct\s+(?:answer|option|choice)\s+is\s+{answer_upper}\b",
        rf"the\s+answer\s+is\s+{answer_upper}\b",
        # Option/choice references
        rf"option\s+{answer_upper}\b",  # "option A"
        rf"choice\s+{answer_upper}\b",  # "choice A"
        # Conclusion markers
        rf"(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?{answer_upper}\b",
        # Answer with separators
        rf"\b{answer_upper}\s*[:\.\)]",  # "A:" or "A." or "A)"
        # Answer after action verbs
        rf"(?:select|choose)\s+{answer_upper}\b",
        rf"it\s+is\s+{answer_upper}\b",
        rf"would\s+be\s+{answer_upper}\b",
    ]

    for pattern in patterns:
        if re.search(pattern, predict, re.IGNORECASE | re.MULTILINE):
            return True

    return False


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

    # Calculate scores
    MMStar_score = {}
    MMStar_score["final score"] = 0
    total_correct = 0

    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            count = MMStar_counter[k][l2_k]
            if count > 0:
                MMStar_score[f"{k}({l2_k})"] = float(l2_v) / float(count)
            else:
                MMStar_score[f"{k}({l2_k})"] = 0.0
            MMStar_score[k] += l2_v
            total_correct += l2_v
        MMStar_score["final score"] += MMStar_score[k]
        if MMStar_score[k] > 0:
            MMStar_score[k] = float(MMStar_score[k]) / 250.0

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

    logger.info(
        f"MMStar_eval successfully finished evaluating {eval_file}, results saved in {score_pth}"
    )


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
    return response.text


def parse_arguments():
    parser = argparse.ArgumentParser(description="MMStar Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Model path",
    )
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")
    parser.add_argument(
        "--dataset", type=str, default="Lin-Chen/MMStar", help="Dataset path"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Split to use for evaluation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--resize_shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize shape for the image",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger.info("\033[32mStarting MMStar Evaluation\033[0m")
    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split)
    logger.info(f"\033[32mLoading model from {args.model}\033[0m")
    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )
    config = model.config
    logger.info(f"\033[32mConfig: {config}\033[0m")

    result_file = f'./results/{args.model.split("/")[-1]}_{args.dataset.split("/")[-1]}_{args.split}_predictions.csv'
    os.makedirs("./results/", exist_ok=True)

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

    logger.info(f"\033[32mSaving results to {result_file}\033[0m")
    logger.info(f"\033[32mEvaluation complete\033[0m")


if __name__ == "__main__":
    main()
