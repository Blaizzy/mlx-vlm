from collections.abc import Mapping


def predict(model, processor, state, questions, **kwargs):
    """Predict named decisions using a model's native scoring and calibration.

    Questions contain a type, instructions, and criteria. Types are ``choice``,
    ``score``, ``bool``, and ``multi_label``; support is model-specific.
    Probabilities and independent scores remain distinct, and model-specific
    metrics stay in ``metadata``.
    """
    supported = getattr(model, "decision_types", ())
    if not supported:
        raise ValueError("This model does not support decision prediction")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("At least one named question is required")
    normalized = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not isinstance(question, Mapping):
            raise ValueError("Questions must map names to question dictionaries")
        question = dict(question)
        kind = question.get("type", "choice")
        if kind not in supported:
            raise ValueError(f"This model does not support {kind!r} decisions")
        criteria = question.get("criteria")
        if kind in ("choice", "multi_label", "score"):
            minimum = 1 if kind == "multi_label" else 2
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
        if kind == "multi_label" and not 0 <= question.get("threshold", 0.5) <= 1:
            raise ValueError("threshold must be between zero and one")
        if (
            kind == "bool"
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
    return model.predict(processor, state, normalized, **kwargs)
