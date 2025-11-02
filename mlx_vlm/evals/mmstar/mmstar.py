import argparse
import csv
import os
from copy import deepcopy
from json import dump
from logging import getLogger

from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

logger = getLogger(__name__)


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

    for line in tqdm(data):
        predict = str(line["prediction"])
        answers = str(line["answer"])
        category = str(line["category"])
        l2_category = str(line["l2_category"])
        MMStar_counter[category][l2_category] += 1

        answer = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        try:
            if answer == predict[0]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0] == "(" and answer == predict[1]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:7] == "option " and answer == predict[7]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:14] == "the answer is " and answer == predict[14]:
                MMStar_score_l2[category][l2_category] += 1
        except Exception as e:
            pass

    MMStar_score = {}
    MMStar_score["final score"] = 0
    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            MMStar_score[f"{k}({l2_k})"] = float(l2_v) / float(MMStar_counter[k][l2_k])
            MMStar_score[k] += l2_v
        MMStar_score["final score"] += MMStar_score[k]
        MMStar_score[k] = float(MMStar_score[k]) / 250.0
    MMStar_score["final score"] = float(MMStar_score["final score"]) / 1500.0

    score_pth = eval_file.replace(".csv", "_score.json")
    with open(score_pth, "w") as f:
        dump(MMStar_score, f, indent=2)

    logger.info(
        f"MMStar_eval successfully finished evaluating {eval_file}, results saved in {score_pth}"
    )
    logger.info("Score: ")
    for key, value in MMStar_score.items():
        logger.info("{}:{}".format(key, value))


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

    result_file = f'./results/{args.model.split("/")[-1]}_{args.dataset.split("/")[-1] }_{args.split}_predictions.csv'
    os.makedirs("./results/", exist_ok=True)

    results = []
    for example in tqdm(dataset):
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

    print("First 5 results:")
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
