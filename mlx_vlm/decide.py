"""Run typed decision models from the command line."""

import argparse
import json
from pathlib import Path

from .decision import _validate_question_structure, predict
from .utils import load


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict typed decisions with MLX-VLM")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face ID")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--state", help="Text to evaluate")
    state.add_argument("--state-file", type=Path, help="UTF-8 text file to evaluate")
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        metavar="PATH_OR_URL",
        help="Image to evaluate; repeat for several (state is optional with images)",
    )
    parser.add_argument(
        "--audio",
        metavar="PATH_OR_URL",
        help="Audio clip to evaluate (state is optional with audio)",
    )
    questions = parser.add_mutually_exclusive_group(required=True)
    questions.add_argument(
        "--questions", type=json.loads, help="Named questions as JSON"
    )
    questions.add_argument(
        "--questions-file", type=Path, help="JSON file of named questions"
    )
    args = parser.parse_args(argv)
    media = {"images": args.images, "audio": args.audio}
    media = {name: value for name, value in media.items() if value}
    if args.state is None and args.state_file is None and not media:
        parser.error(
            "one of the arguments --state --state-file --image --audio is required"
        )
    try:
        state = (
            args.state_file.read_text(encoding="utf-8")
            if args.state_file is not None
            else args.state
        )
        questions = (
            json.loads(args.questions_file.read_text(encoding="utf-8"))
            if args.questions_file is not None
            else args.questions
        )
        _validate_question_structure(questions)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    model, processor = load(args.model)
    try:
        result = predict(model, processor, state, questions, **media)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
