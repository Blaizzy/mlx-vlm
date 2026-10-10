from collections.abc import Mapping

_MEDIA = {"images": "image", "videos": "video", "audio": "audio"}


def _validate_question_structure(questions):
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("At least one named question is required")
    if any(
        not isinstance(name, str) or not isinstance(question, Mapping)
        for name, question in questions.items()
    ):
        raise ValueError("Questions must map names to question dictionaries")


def predict(model, processor, state, questions, **kwargs):
    """Predict named decisions using a model's native scoring and calibration.

    Questions contain a type, instructions, and criteria. Types are ``choice``,
    ``score``, ``bool``, and ``multi_label``; support is model-specific.
    Probabilities and independent scores remain distinct, and model-specific
    metrics stay in ``metadata``. ``state`` may be ``None`` when media such as
    ``images=[...]`` carry the whole state; a model reads the media named in
    its ``decision_media``.
    """
    normalized, kwargs = normalize_questions(model, questions, **kwargs)
    return model.predict(processor, state, normalized, **kwargs)


def normalize_questions(model, questions, *, allow_single_criterion=False, **kwargs):
    """Validate capabilities independently of eager or scheduled execution."""
    supported = getattr(model, "decision_types", ())
    if not supported:
        raise ValueError("This model does not support decision prediction")
    kwargs = {k: v for k, v in kwargs.items() if k not in _MEDIA or v is not None}
    media = getattr(model, "decision_media", ())
    for name, kind in _MEDIA.items():
        if name in kwargs and name not in media:
            raise ValueError(f"This model does not support {kind} input")
    _validate_question_structure(questions)
    normalized = {}
    for name, question in questions.items():
        question = dict(question)
        kind = question.get("type", "choice")
        if kind not in supported:
            raise ValueError(f"This model does not support {kind!r} decisions")
        criteria = question.get("criteria")
        if kind in ("choice", "multi_label", "score"):
            minimum = 1 if kind == "multi_label" or allow_single_criterion else 2
            if (
                not isinstance(criteria, (list, tuple, Mapping))
                or len(criteria) < minimum
            ):
                raise ValueError(f"{kind} needs at least {minimum} criteria")
            if kind != "score" and any(
                not isinstance(label, str) or not label for label in criteria
            ):
                raise ValueError("Decision labels must be nonempty strings")
            if kind != "score" and len(set(criteria)) != len(criteria):
                raise ValueError("Decision labels must be unique")
        if kind == "score" and isinstance(criteria, Mapping):
            raise ValueError("score criteria must be an ordered list")
        if kind == "multi_label":
            threshold = question.get("threshold", 0.5)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0 <= threshold <= 1
            ):
                raise ValueError("threshold must be a number between zero and one")
        if (
            kind in ("bool", "noul")
            and criteria is not None
            and not isinstance(criteria, Mapping)
        ):
            raise ValueError("bool criteria must map false and true to descriptions")
        if isinstance(criteria, Mapping):
            question["criteria"] = dict(criteria)
        elif isinstance(criteria, tuple):
            question["criteria"] = list(criteria)
        question["type"] = kind
        normalized[name] = question
    return normalized, kwargs


class DecisionCancelled(Exception):
    pass


def check_cancelled(event):
    if event.is_set():
        raise DecisionCancelled("Decision cancelled")


def context_limit(config, requested=None):
    text = getattr(config, "text_config", config)
    native = getattr(text, "max_position_embeddings", 65536)
    limit = min(65536, native) if requested is None else requested
    if not 0 < limit <= native:
        raise ValueError(f"Decision context limit must be between 1 and {native}")
    return limit


def main(argv=None):
    """Compatibility entry point for the original System One CLI."""
    from .decide import main as decide

    return decide(argv, default_format="systemone")


def systemone(model, processor, request, max_length=None, *, engine=None):
    """Compatibility helper using the same model capabilities as native serving."""
    from .decision_scheduler import make_decision_engine
    from .systemone import SystemOneRequest, format_response, native_request

    request = SystemOneRequest.model_validate(request).model_dump(exclude_none=True)
    body = native_request(request)
    questions, media = normalize_questions(
        model,
        body["questions"],
        allow_single_criterion=True,
        **{
            key: value
            for key, value in body.items()
            if key not in ("model", "state", "questions")
        },
    )
    body = {**body, "questions": questions, **media}
    backend = engine or make_decision_engine(
        model,
        processor,
        max_length=max_length,
        prefill_step_size=context_limit(getattr(model, "config", None), max_length),
        cache_bytes=0,
    )
    state = backend.prepare(body)
    while not state.done:
        backend.step([state])
    return format_response(request, backend.finish(state))


def comparison_report(*args, **kwargs):
    from .decide import comparison_report as compare

    return compare(*args, **kwargs)


if __name__ == "__main__":
    main()
