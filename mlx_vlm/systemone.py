"""TypeSafe System One compatibility over native typed decision results."""

import json
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        for value in [
            *self.images,
            *(frame for video in self.videos for frame in video),
        ]:
            if not value.startswith(("https://", "http://", "data:image/")):
                raise ValueError("Media must be an HTTP(S) URL or an image data URL")
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


def native_request(request):
    body = SystemOneRequest.model_validate(request).model_dump(exclude_none=True)
    body["questions"] = {
        name: {
            **question,
            "type": "bool" if question["type"] == "noul" else question["type"],
        }
        for name, question in body["questions"].items()
    }
    return {
        key: value
        for key, value in body.items()
        if key not in ("images", "videos") or value
    }


def format_response(request, output):
    answers = {}
    raw = output.get("probabilities")
    native = output["response"]
    for name, question in request["questions"].items():
        answer = native["answers"][name]
        if raw is not None:
            probabilities = raw[name]
        elif question["type"] == "noul":
            p = answer["probability"]
            probabilities = {"true": p, "false": 1 - p}
        else:
            probabilities = answer["probabilities"]
        answers[name] = systemone_answer(question, probabilities)
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {
            "input_tokens": native.get("usage", {}).get("input_tokens", 0),
            "output_tokens": 0,
        },
    }
