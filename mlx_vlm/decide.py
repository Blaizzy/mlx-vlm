"""Run typed decisions, JSONL batches, and checkpoint comparisons."""

import argparse
import contextlib
import json
import sys
from collections import deque
from pathlib import Path

from .decision import _validate_question_structure, predict
from .decision_scheduler import DecisionScheduler
from .systemone import SystemOneRequest, format_response, native_request
from .utils import load


def comparison_report(
    reference, candidate, *, max_probability_drift=0.01, max_choice_flips=0
):
    """Compare unrounded distributions; the report is a regression gate, not a calibration claim."""
    if len(reference) != len(candidate):
        raise ValueError("Comparison requires the same number of records")
    if any(record is None for record in [*reference, *candidate]):
        raise ValueError("Model does not expose unrounded decision probabilities")
    deltas, flips, count = [], 0, 0
    for expected, actual in zip(reference, candidate):
        if isinstance(expected, dict) and isinstance(actual, dict):
            if expected.keys() != actual.keys():
                raise ValueError("Comparison question names differ")
            labels = [(expected[name], actual[name]) for name in expected]
            if any(p.keys() != q.keys() for p, q in labels):
                raise ValueError("Comparison option labels differ")
            expected = [list(p.values()) for p, q in labels]
            actual = [[q[label] for label in p] for p, q in labels]
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
        try:
            return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(str(error)) from error
    return json.loads(value)


def _requests(args, parser):
    if args.request is not None:
        if (
            any(
                value is not None
                for value in (
                    args.state,
                    args.state_file,
                    args.questions,
                    args.questions_file,
                )
            )
            or args.images
            or args.audio
            or args.video
        ):
            parser.error("Use --request or inline state/questions/media")
        stream = (
            sys.stdin if args.request == "-" else open(args.request, encoding="utf-8")
        )
        try:
            if args.jsonl:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except ValueError as error:
                            raise ValueError(
                                f"Invalid JSON on line {line_number}: {error}"
                            ) from error
            else:
                yield json.load(stream)
        finally:
            if stream is not sys.stdin:
                stream.close()
        return
    if args.jsonl:
        parser.error("--jsonl requires --request FILE|-")
    if not args.model:
        parser.error("--model is required")
    if args.questions is None and args.questions_file is None:
        parser.error("--questions or --questions-file is required")
    if (
        args.state is None
        and args.state_file is None
        and not (args.images or args.audio or args.video)
    ):
        parser.error(
            "one of the arguments --state --state-file --image --audio is required (or --video)"
        )
    state = (
        args.state_file.read_text(encoding="utf-8") if args.state_file else args.state
    )
    if args.format == "systemone" and isinstance(state, str) and state.startswith("@"):
        state = _json_argument(state)
    questions = (
        json.loads(args.questions_file.read_text(encoding="utf-8"))
        if args.questions_file
        else args.questions
    )
    body = {"model": args.model, "state": state, "questions": questions}
    if args.images:
        body["images"] = args.images
    if args.audio:
        body["audio"] = args.audio
    if args.video:
        body["videos"] = [json.loads(video) for video in args.video]
    yield body


def _validated_body(raw, args):
    if not isinstance(raw, dict):
        raise ValueError("Each request must be a JSON object")
    body = dict(raw)
    if args.model:
        body["model"] = args.model
    if args.format == "systemone":
        return SystemOneRequest.model_validate(body).model_dump(exclude_none=True)
    _validate_question_structure(body.get("questions"))
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise ValueError("Specify a model for each request or with --model")
    if body.get("state") is None and not any(
        body.get(name) for name in ("images", "audio", "videos")
    ):
        raise ValueError("state is required unless media are given")
    return body


def main(argv=None, *, default_format="native"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Model path or Hugging Face ID; overrides request files"
    )
    state = parser.add_mutually_exclusive_group()
    state.add_argument(
        "--state", help="Text to evaluate (legacy format also accepts @JSONFILE)"
    )
    state.add_argument("--state-file", type=Path)
    questions = parser.add_mutually_exclusive_group()
    questions.add_argument("--questions", type=_json_argument)
    questions.add_argument("--questions-file", type=Path)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--images", nargs="*", action="extend", dest="images")
    parser.add_argument("--audio")
    parser.add_argument(
        "--video", action="append", default=[], help="JSON array of frame URLs"
    )
    parser.add_argument("--request", help="JSON/JSONL file; - reads stdin")
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument(
        "--format", choices=["native", "systemone"], default=default_format
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--prefix-cache-mb", type=int, default=256)
    parser.add_argument("--compare-model")
    parser.add_argument("--report")
    parser.add_argument("--max-probability-drift", type=float, default=0.01)
    parser.add_argument("--max-choice-flips", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.prefill_step_size < 1
        or args.prefix_cache_mb < 0
        or (args.max_length is not None and args.max_length < 1)
    ):
        parser.error(
            "Batch/prefill/input sizes must be positive and cache budget nonnegative"
        )
    if not 0 <= args.max_probability_drift <= 1 or args.max_choice_flips < 0:
        parser.error(
            "Comparison thresholds must be nonnegative; probability drift must be <= 1"
        )
    if args.report and not args.compare_model:
        parser.error("--report requires --compare-model")
    output_stream = sys.stdout
    requests = (_validated_body(raw, args) for raw in _requests(args, parser))
    # Retain the simple native CLI's standard load()/predict() contract.
    simple = (
        args.format == "native"
        and args.request is None
        and not args.compare_model
        and args.max_length is None
        and args.batch_size == 4
        and args.prefill_step_size == 512
        and args.prefix_cache_mb == 256
    )
    if simple:
        try:
            body = next(requests)
            with contextlib.redirect_stdout(sys.stderr):
                model, processor = load(body["model"])
                result = predict(
                    model,
                    processor,
                    body.get("state"),
                    body["questions"],
                    **{
                        k: v
                        for k, v in body.items()
                        if k not in ("model", "state", "questions")
                    },
                )
            print(json.dumps(result, ensure_ascii=False, indent=2), file=output_stream)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    def loader(name):
        with contextlib.redirect_stdout(sys.stderr):
            model, processor = load(name)
        return model, processor, model.config

    def scheduler():
        return DecisionScheduler(
            loader,
            max_length=args.max_length,
            batch_size=1 if args.compare_model else args.batch_size,
            prefill_step_size=args.prefill_step_size,
            cache_bytes=args.prefix_cache_mb * 1024**2,
            max_pending=max(64, args.batch_size * 2),
        )

    def submit(queue, body):
        native = native_request(body) if args.format == "systemone" else body
        return queue.submit(native, allow_single_criterion=args.format == "systemone")

    candidates, inputs, pending = [], [], deque()
    queue = scheduler()

    def emit(body, job):
        output = job.future.result()
        result = (
            format_response(body, output)
            if args.format == "systemone"
            else output["response"]
        )
        print(
            json.dumps(result, indent=None if args.jsonl else 2, ensure_ascii=False),
            file=output_stream,
            flush=True,
        )
        if args.compare_model:
            candidates.append(output["probabilities"])

    try:
        for body in requests:
            if args.compare_model:
                inputs.append(body)
            pending.append((body, submit(queue, body)))
            if len(pending) >= args.batch_size * 2:
                emit(*pending.popleft())
        while pending:
            emit(*pending.popleft())
    except (OSError, ValueError, KeyboardInterrupt) as error:
        parser.exit(130 if isinstance(error, KeyboardInterrupt) else 2, f"{error}\n")
    finally:
        queue.stop_and_join()
    if args.compare_model:
        if not inputs:
            parser.error("Comparison requires at least one evaluation request")
        queue, reference = scheduler(), []
        try:
            for body in inputs:
                job = submit(queue, {**body, "model": args.compare_model})
                reference.append(job.future.result()["probabilities"])
            report = comparison_report(
                reference,
                candidates,
                max_probability_drift=args.max_probability_drift,
                max_choice_flips=args.max_choice_flips,
            )
        except ValueError as error:
            parser.error(str(error))
        finally:
            queue.stop_and_join()
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
