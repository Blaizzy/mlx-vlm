"""Native Clef decisions with the TypeSafe System One wire format."""

import argparse
import json
import math
import sys
from typing import Annotated, Dict, List, Literal, Union

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models.clef.config import ModelConfig
from .models.clef.processing_clef import encode_record

Content = Union[str, Dict, List]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instructions: Union[Content, None] = None


class NoulQuestion(Question):
    type: Literal["noul"]
    criteria: Union[Dict[Literal["true", "false"], Union[Content, None]], None] = None


class ChoiceQuestion(Question):
    type: Literal["choice"]
    criteria: Dict[str, Union[Content, None]] = Field(min_length=1, max_length=255)


class ScoreQuestion(Question):
    type: Literal["score"]
    criteria: List[Content] = Field(min_length=1, max_length=10)


DecisionQuestion = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1)
    state: Content
    questions: Dict[str, DecisionQuestion] = Field(min_length=1, max_length=64)
    images: List[str] = Field(default_factory=list, max_length=10)
    videos: List[List[str]] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_media(self):
        if any(not frames or len(frames) > 64 for frames in self.videos):
            raise ValueError("Each video must contain 1 to 64 frames")
        # JSON cannot represent NaN/Infinity, including inside structured criteria.
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: Dict[str, Content]
    probabilities: Dict[str, float]


class DecisionUsage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[
        str,
        Annotated[
            Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")
        ],
    ]
    usage: DecisionUsage


def systemone_answer(question, probabilities):
    """TypeSafe adapter confidence statistics, computed before output rounding."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = list(question["criteria"])
        choice = max(options, key=probabilities.__getitem__)
        n = len(options)
        confidence = 1.0 if n == 1 else (probabilities[choice] - 1 / n) / (1 - 1 / n)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "probabilities": {key: round(probabilities[key], 4) for key in options},
        }
    levels = [str(i) for i in range(len(question["criteria"]))]
    probs = [probabilities[key] for key in levels]
    mode = max(range(len(probs)), key=probs.__getitem__)
    center = (len(probs) - 1) / 2
    uniform_deviation = sum(abs(i - center) for i in range(len(probs))) / len(probs)
    deviation = sum(p * abs(i - mode) for i, p in enumerate(probs))
    confidence = (
        max(0.0, 1 - deviation / uniform_deviation) if uniform_deviation else 1.0
    )
    return {
        "type": "score",
        "score": round(sum(i * p for i, p in enumerate(probs)), 4),
        "confidence": round(confidence, 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {key: round(probabilities[key], 4) for key in levels},
    }


def _decode_media(request, check_cancelled=lambda: None):
    import numpy as np

    from .utils import load_image

    def image(value):
        check_cancelled()
        if not value.startswith(("https://", "http://", "data:image/")):
            raise ValueError("Media must be an HTTP(S) URL or an image data URL")
        return load_image(value).convert("RGB")

    record = dict(request)
    record["images"] = [image(value) for value in request.get("images", [])]
    record["videos"] = [
        np.stack([np.asarray(image(frame)) for frame in frames])
        for frames in request.get("videos", [])
    ]
    return record


def prepare_request(processor, request, max_length=65536, check_cancelled=lambda: None):
    request = SystemOneRequest.model_validate(request).model_dump(exclude_none=True)
    check_cancelled()
    encoded = encode_record(
        processor.tokenizer,
        _decode_media(request, check_cancelled),
        processor=processor,
        max_length=max_length,
    )
    check_cancelled()
    return request, encoded


def decision_probabilities(logits):
    probabilities = [
        mx.softmax(values.astype(mx.float32), precise=True) for values in logits
    ]
    mx.eval(probabilities)
    result = [values.tolist() for values in probabilities]
    if not all(math.isfinite(p) for values in result for p in values):
        raise RuntimeError("Clef produced non-finite probabilities")
    return result


def format_response(request, encoded, probabilities):
    answers = {}
    for question, probs in zip(encoded.questions, probabilities):
        answers[question.question_id] = systemone_answer(
            request["questions"][question.question_id],
            dict(zip(question.option_ids, probs)),
        )
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }


def systemone(model, processor, request, max_length=None, *, engine=None):
    from .models.clef.inference import context_limit

    if not callable(getattr(model, "decide", None)):
        raise ValueError("This model does not support native decisions")
    limit = (
        context_limit(model.config, max_length)
        if isinstance(getattr(model, "config", None), ModelConfig)
        else (max_length or 65536)
    )
    request, encoded = prepare_request(processor, request, limit)
    logits = (engine or model).decide(encoded)
    return format_response(request, encoded, decision_probabilities(logits))


def comparison_report(
    reference, candidate, *, max_probability_drift=0.01, max_choice_flips=0
):
    """Compare unrounded distributions; the report is a regression gate, not a calibration claim."""
    if len(reference) != len(candidate):
        raise ValueError("Comparison requires the same number of records")
    deltas, flips, count = [], 0, 0
    for expected, actual in zip(reference, candidate):
        if len(expected) != len(actual):
            raise ValueError("Comparison question counts differ")
        for p, q in zip(expected, actual):
            if len(p) != len(q):
                raise ValueError("Comparison option counts differ")
            deltas.extend(abs(a - b) for a, b in zip(p, q))
            flips += max(range(len(p)), key=p.__getitem__) != max(
                range(len(q)), key=q.__getitem__
            )
            count += 1
    maximum = max(deltas, default=0.0)
    return {
        "records": len(reference),
        "questions": count,
        "max_probability_drift": maximum,
        "mean_probability_drift": sum(deltas) / len(deltas) if deltas else 0.0,
        "argmax_flips": flips,
        "passed": maximum <= max_probability_drift and flips <= max_choice_flips,
        "thresholds": {
            "max_probability_drift": max_probability_drift,
            "max_choice_flips": max_choice_flips,
        },
    }


def _json_argument(value):
    if value.startswith("@"):
        with open(value[1:]) as stream:
            return json.load(stream)
    return json.loads(value)


def _read_requests(args, parser):
    if args.request is not None:
        if (
            args.state is not None
            or args.questions is not None
            or args.images
            or args.video
        ):
            parser.error("Use --request or inline state/questions/media")
        stream = sys.stdin if args.request == "-" else open(args.request)
        try:
            if args.jsonl:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid JSON on line {line_number}: {exc}"
                            ) from exc
            else:
                yield json.load(stream)
        finally:
            if stream is not sys.stdin:
                stream.close()
    else:
        if args.state is None or args.questions is None or not args.model:
            parser.error("Supply --request FILE|- or --model, --state and --questions")
        if args.jsonl:
            parser.error("--jsonl requires --request FILE|-")
        state = _json_argument(args.state) if args.state.startswith("@") else args.state
        yield {
            "model": args.model,
            "state": state,
            "questions": _json_argument(args.questions),
            "images": args.images,
            "videos": [json.loads(v) for v in args.video],
        }


def main(argv=None):
    import contextlib
    from collections import deque
    from pathlib import Path

    from .decision_scheduler import DecisionScheduler
    from .utils import get_model_path, load_model, load_processor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="JSON/JSONL request file; - reads stdin")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Read and emit one JSON object per line, in input order",
    )
    parser.add_argument(
        "--model", help="Model ID/path; overrides the model in request files"
    )
    parser.add_argument(
        "--state", help="Inline state text or @FILE containing JSON state"
    )
    parser.add_argument("--questions", help="Inline questions JSON or @FILE")
    parser.add_argument(
        "--images", nargs="*", default=[], help="Image URLs or data URLs"
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="JSON array of frame URLs; repeat for multiple videos",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Complete input limit (default: min(65536, backbone limit))",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument(
        "--prefix-cache-mb",
        type=int,
        default=256,
        help="Prefix cache budget; 0 disables reuse",
    )
    parser.add_argument(
        "--compare-model",
        help="Reference checkpoint for probability drift and argmax comparison",
    )
    parser.add_argument(
        "--report", help="Write comparison report to this JSON file (otherwise stderr)"
    )
    parser.add_argument("--max-probability-drift", type=float, default=0.01)
    parser.add_argument("--max-choice-flips", type=int, default=0)
    args = parser.parse_args(argv)
    output_stream = sys.stdout
    if args.batch_size < 1 or args.prefill_step_size < 1 or args.prefix_cache_mb < 0:
        parser.error(
            "Batch/prefill sizes must be positive and cache budget nonnegative"
        )
    if not 0 <= args.max_probability_drift <= 1 or args.max_choice_flips < 0:
        parser.error(
            "Comparison thresholds must be nonnegative; probability drift must be <= 1"
        )
    if args.report and not args.compare_model:
        parser.error("--report requires --compare-model")

    def load(name):
        # Keep stdout parseable for shell pipelines.
        with contextlib.redirect_stdout(sys.stderr):
            path = get_model_path(name)
            model = load_model(path)
            return model, load_processor(path, add_detokenizer=False), model.config

    def scheduler():
        return DecisionScheduler(
            load,
            max_length=args.max_length,
            # Isolate checkpoint drift from batch-size-dependent BF16 rounding.
            batch_size=1 if args.compare_model else args.batch_size,
            prefill_step_size=args.prefill_step_size,
            cache_bytes=args.prefix_cache_mb * 1024**2,
            max_pending=max(64, args.batch_size * 2),
        )

    candidates, inputs = [], []
    queue = scheduler()
    pending = deque()

    def emit(job):
        result = job.future.result()
        print(
            json.dumps(result["response"], indent=None if args.jsonl else 2),
            flush=True,
            file=output_stream,
        )
        if args.compare_model:
            candidates.append(result["probabilities"])

    try:
        for raw in _read_requests(args, parser):
            if not isinstance(raw, dict):
                raise ValueError("Each request must be a JSON object")
            if args.model:
                raw["model"] = args.model
            body = SystemOneRequest.model_validate(raw).model_dump(exclude_none=True)
            if args.compare_model:
                inputs.append(body)
            pending.append(queue.submit(body))
            if len(pending) >= args.batch_size * 2:
                emit(pending.popleft())
        while pending:
            emit(pending.popleft())
    except (ValueError, OSError, KeyboardInterrupt) as exc:
        parser.exit(130 if isinstance(exc, KeyboardInterrupt) else 2, f"{exc}\n")
    finally:
        queue.stop_and_join()
    if args.compare_model:
        if not inputs:
            parser.error("Comparison requires at least one evaluation request")
        # Load the reference after releasing candidate weights to avoid keeping
        # two large checkpoints resident just to measure quantization drift.
        queue = scheduler()
        reference = []
        try:
            for body in inputs:
                job = queue.submit({**body, "model": args.compare_model})
                reference.append(job.future.result()["probabilities"])
        finally:
            queue.stop_and_join()
        report = comparison_report(
            reference,
            candidates,
            max_probability_drift=args.max_probability_drift,
            max_choice_flips=args.max_choice_flips,
        )
        report["reference_model"] = args.compare_model
        report["candidate_models"] = sorted({body["model"] for body in inputs})
        rendered = json.dumps(report, indent=2) + "\n"
        if args.report:
            Path(args.report).write_text(rendered)
        else:
            sys.stderr.write(rendered)
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
