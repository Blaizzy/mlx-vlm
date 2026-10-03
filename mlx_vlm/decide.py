"""Run typed decision models from the command line."""

import argparse
import json
from pathlib import Path

from .decision import _validate_question_structure, predict
from .utils import load


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict typed decisions with MLX-VLM")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face ID")
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--state", help="Text to evaluate")
    state.add_argument("--state-file", type=Path, help="UTF-8 text file to evaluate")
    questions = parser.add_mutually_exclusive_group(required=True)
    questions.add_argument(
        "--questions", type=json.loads, help="Named questions as JSON"
    )
    questions.add_argument(
        "--questions-file", type=Path, help="JSON file of named questions"
    )
    args = parser.parse_args(argv)
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
        result = predict(model, processor, state, questions)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
