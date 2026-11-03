"""Run typed decision models from the command line."""

import argparse
import json
from pathlib import Path

from .decision import _validate_question_structure, predict
from .models.d1_lfm3_vl.api import SCORE_READOUT, SCORE_READOUTS
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
        action="append",
        metavar="PATH_OR_URL",
        help="Audio clip to evaluate (state is optional with audio)",
    )
    parser.add_argument(
        "--video", action="append", dest="videos", help="Video to evaluate"
    )
    parser.add_argument("--set-size", type=int, help="d1 questions per set")
    parser.add_argument(
        "--score-readout", choices=SCORE_READOUTS, default=SCORE_READOUT
    )
    questions = parser.add_mutually_exclusive_group(required=True)
    questions.add_argument(
        "--questions", type=json.loads, help="Named questions as JSON"
    )
    questions.add_argument(
        "--questions-file", type=Path, help="JSON file of named questions"
    )
    args = parser.parse_args(argv)
    media = {
        "images": args.images,
        "audio": args.audio[0] if args.audio and len(args.audio) == 1 else args.audio,
        "videos": args.videos,
    }
    media = {name: value for name, value in media.items() if value}
    if args.state is None and args.state_file is None and not media:
        parser.error(
            "one of the arguments --state --state-file --image --audio is required (or --video)"
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
        if callable(getattr(model, "system_one", None)):
            if args.state_file is not None and args.state_file.suffix == ".json":
                state = json.loads(state)
            inputs = (
                [{"type": "image", "path": p} for p in args.images or []]
                + [{"type": "audio", "path": p} for p in args.audio or []]
                + [{"type": "video", "path": p} for p in args.videos or []]
            )
            result = model.system_one(
                state,
                questions,
                inputs or None,
                score_readout=args.score_readout,
                set_size=args.set_size,
            )
        else:
            if args.set_size is not None or args.score_readout != SCORE_READOUT:
                raise ValueError("--set-size and --score-readout require a d1 model")
            result = predict(model, processor, state, questions, **media)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
