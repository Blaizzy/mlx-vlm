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


def predict(model, processor, input, questions, **kwargs):
    """Return OpenAI Decisions answers in question order.

    ``input`` is text or an ordered conversation. Questions use ``predicate``,
    ``choice`` with typed choices, or ``score`` with ordered levels. Supported
    Python image/audio/video objects may be supplied through keyword arguments.
    Model calibration is applied before formatting; confidence uses the local
    decision statistics, not an OpenAI calibration guarantee.
    """
    from .decision_protocol import DecisionRequest, format_response, native_request

    if "format" in kwargs:
        raise TypeError("predict() does not accept 'format'; decisions use one schema")
    metadata = {
        key: kwargs[key]
        for key in ("prompt_cache_key", "prompt_cache_options", "safety_identifier")
        if key in kwargs
    }
    validation = {"input": input, "questions": questions, **metadata}
    request = DecisionRequest.model_validate(
        validation,
        context={"has_media": any(kwargs.get(key) is not None for key in _MEDIA)},
    ).model_dump(exclude_none=True)
    request.update(kwargs)
    body = native_request(request)
    body.update(
        {
            key: value
            for key, value in kwargs.items()
            if key
            not in (
                *_MEDIA,
                "prompt_cache_key",
                "prompt_cache_options",
                "safety_identifier",
            )
        }
    )
    options = {
        key: value for key, value in body.items() if key not in ("state", "questions")
    }
    normalized, options = normalize_questions(model, body["questions"], **options)
    body = {"state": body["state"], "questions": normalized, **options}
    return format_response(request, _predict_output(model, processor, body))


def normalize_questions(model, questions, **kwargs):
    """Validate capabilities independently of eager or scheduled execution."""
    from .systemone import validate_questions

    supported = getattr(model, "decision_types", ())
    if not supported:
        raise ValueError("This model does not support decision prediction")
    kwargs = {k: v for k, v in kwargs.items() if k not in _MEDIA or v is not None}
    media = getattr(model, "decision_media", ())
    for name, kind in _MEDIA.items():
        if name in kwargs and name not in media:
            raise ValueError(f"This model does not support {kind} input")
    _validate_question_structure(questions)
    validate_questions(questions)
    normalized = {}
    for name, question in questions.items():
        question = dict(question)
        kind = question.get("type", "choice")
        if kind == "noul" and "bool" in supported:
            kind = "bool"
        if kind not in supported:
            raise ValueError(f"This model does not support {kind!r} decisions")
        criteria = question.get("criteria")
        if kind in ("choice", "score"):
            if not isinstance(criteria, (list, tuple, Mapping)) or not criteria:
                raise ValueError(f"{kind} needs at least one criterion")
            if kind != "score" and any(
                not isinstance(label, str) or not label for label in criteria
            ):
                raise ValueError("Decision labels must be nonempty strings")
            if kind != "score" and len(set(criteria)) != len(criteria):
                raise ValueError("Decision labels must be unique")
        if kind == "score" and isinstance(criteria, Mapping):
            raise ValueError("score criteria must be an ordered list")
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


def _predict_output(model, processor, body, *, engine=None):
    """Retain raw probabilities from incremental backends before formatting."""
    from .decision_scheduler import make_decision_engine

    kwargs = {
        key: value
        for key, value in body.items()
        if key not in ("model", "state", "questions")
    }
    if engine is None and getattr(model, "make_decision_engine", None) is None:
        return {
            "response": model.predict(
                processor, body.get("state"), body["questions"], **kwargs
            )
        }
    max_length = kwargs.get("max_length")
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
    return backend.finish(state)


def systemone(model, processor, request, max_length=None, *, engine=None):
    """Accept a strict System One request using the shared prediction formatter."""
    from .systemone import SystemOneRequest, format_response, native_request

    request = SystemOneRequest.model_validate(request).model_dump(exclude_none=True)
    body = native_request(request)
    questions, media = normalize_questions(
        model,
        body["questions"],
        **{
            key: value
            for key, value in body.items()
            if key not in ("model", "state", "questions")
        },
    )
    body = {**body, "questions": questions, **media}
    if max_length is not None:
        body["max_length"] = max_length
    return format_response(
        request, _predict_output(model, processor, body, engine=engine)
    )


def comparison_report(*args, **kwargs):
    from .decide import comparison_report as compare

    return compare(*args, **kwargs)
