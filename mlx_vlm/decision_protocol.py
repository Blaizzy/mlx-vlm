"""OpenAI Decisions schemas and adapters to model-owned prediction interfaces."""

import json
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

from .systemone import systemone_answer


class DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InputText(DecisionModel):
    type: Literal["input_text"]
    text: str


class InputImage(DecisionModel):
    type: Literal["input_image"]
    image_url: str
    detail: Union[Literal["auto"], None] = "auto"

    @model_validator(mode="after")
    def validate_url(self):
        if not self.image_url.startswith(("https://", "http://", "data:image/")):
            raise ValueError("input_image needs an HTTP(S) URL or image data URL")
        return self


class OutputText(DecisionModel):
    type: Literal["output_text"]
    text: str


class RefusalContent(DecisionModel):
    type: Literal["refusal"]
    refusal: str


InputPart = Annotated[Union[InputText, InputImage], Field(discriminator="type")]
MessagePart = Annotated[
    Union[InputText, InputImage, OutputText, RefusalContent],
    Field(discriminator="type"),
]


class InputMessage(DecisionModel):
    type: Literal["message"] = "message"
    role: Literal["user", "developer", "system", "assistant"]
    content: Union[str, list[MessagePart]]
    phase: Union[Literal["commentary", "final_answer"], None] = None

    @model_validator(mode="after")
    def validate_role(self):
        allowed = {
            "user": (InputText, InputImage),
            "developer": (InputText,),
            "system": (InputText,),
            "assistant": (OutputText, RefusalContent),
        }
        if isinstance(self.content, list) and any(
            not isinstance(part, allowed[self.role]) for part in self.content
        ):
            raise ValueError(f"Unsupported content for {self.role} message")
        if self.phase is not None and self.role != "assistant":
            raise ValueError("phase is only supported for assistant messages")
        return self


class ToolItem(DecisionModel):
    call_id: str = Field(min_length=1, max_length=64)
    id: Union[str, None] = None
    status: Union[Literal["in_progress", "completed", "incomplete"], None] = None


class FunctionCall(ToolItem):
    type: Literal["function_call"]
    name: str = Field(min_length=1, max_length=128)
    arguments: str
    namespace: Union[str, None] = Field(default=None, min_length=1, max_length=64)


class CustomToolCall(ToolItem):
    type: Literal["custom_tool_call"]
    name: str = Field(min_length=1, max_length=128)
    input: str
    namespace: Union[str, None] = Field(default=None, min_length=1, max_length=64)


class ToolOutput(ToolItem):
    type: Literal["function_call_output", "custom_tool_call_output"]
    output: Union[str, list[InputPart]]


InputItem = Union[InputMessage, FunctionCall, CustomToolCall, ToolOutput]


class Question(DecisionModel):
    instructions: str
    name: Union[str, None] = None

    @model_validator(mode="after")
    def validate_name(self):
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name must be a string when supplied")
        return self


class PredicateQuestion(Question):
    type: Literal["predicate"]


class ChoiceOption(DecisionModel):
    value: Union[str, bool]
    description: str = ""


class ChoiceQuestion(Question):
    type: Literal["choice"]
    choices: list[ChoiceOption] = Field(min_length=2, max_length=255)

    @model_validator(mode="after")
    def validate_choices(self):
        values = [(type(option.value), option.value) for option in self.choices]
        if len(set(values)) != len(values):
            raise ValueError("Choice values must be distinct")
        return self


class ScoreLevel(DecisionModel):
    label: str
    description: str = ""


class ScoreQuestion(Question):
    type: Literal["score"]
    levels: list[ScoreLevel] = Field(min_length=2, max_length=10)


DecisionQuestion = Annotated[
    Union[PredicateQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]
_questions = TypeAdapter(
    Annotated[list[DecisionQuestion], Field(min_length=1, max_length=200)]
)
_input = TypeAdapter(Union[str, list[InputItem]])


class PromptCacheOptions(DecisionModel):
    # Explicit breakpoints and retention guarantees need backend support first.
    mode: Union[Literal["implicit"], None] = None


class DecisionRequest(DecisionModel):
    # Optional model and media-only input preserve the local server conveniences.
    model: Union[str, None] = Field(default=None, min_length=1)
    input: Union[str, list[InputItem], None] = None
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=200)
    prompt_cache_key: Union[str, None] = Field(default=None, max_length=64)
    prompt_cache_options: Union[PromptCacheOptions, None] = None
    safety_identifier: Union[str, None] = Field(default=None, max_length=128)
    images: Union[list[str], None] = None
    videos: Union[list[list[str]], None] = None
    audio: Union[str, None] = None

    @model_validator(mode="after")
    def validate_input(self, info: ValidationInfo):
        has_media = (
            self.images
            or self.videos
            or self.audio
            or (info.context or {}).get("has_media")
        )
        if self.input is None and not has_media:
            raise ValueError(
                "input is required unless images, videos, or audio are given"
            )
        _native_input(self.input)
        return self


class Answer(DecisionModel):
    name: Union[str, None]


class PredicateAnswer(Answer):
    type: Literal["predicate"] = "predicate"
    probability: float


class ChoiceProbability(DecisionModel):
    value: Union[str, bool]
    probability: float


class ChoiceAnswer(Answer):
    type: Literal["choice"] = "choice"
    choice: Union[str, bool]
    probabilities: list[ChoiceProbability]
    confidence: float


class ScoreProbability(DecisionModel):
    value: int
    label: str
    probability: float


class ScoreAnswer(Answer):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[ScoreProbability]
    confidence: float


class RefusalAnswer(Answer):
    type: Literal["refusal"] = "refusal"


class InputTokensDetails(DecisionModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class OutputTokensDetails(DecisionModel):
    reasoning_tokens: int = 0


class DecisionUsage(DecisionModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class DecisionResponse(DecisionModel):
    model: str
    answers: list[
        Annotated[
            Union[PredicateAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer],
            Field(discriminator="type"),
        ]
    ]
    usage: DecisionUsage


def _native_input(value):
    """Keep conversation roles/order as evidence; never execute supplied tool calls."""
    if value is None or isinstance(value, str):
        return value, []
    items = _input.validate_python(value)
    images, history, calls = [], [], {}
    for item in items:
        if isinstance(item, (FunctionCall, CustomToolCall)):
            if item.call_id in calls:
                raise ValueError("Tool call IDs must be unique")
            calls[item.call_id] = item.type
        elif isinstance(item, ToolOutput):
            if calls.get(item.call_id) != item.type.removesuffix("_output"):
                raise ValueError(
                    "Tool output must follow a matching call_id and call type"
                )
        record = item.model_dump(exclude_none=True)
        key = "content" if isinstance(item, InputMessage) else "output"
        parts = record.get(key)
        if isinstance(parts, list):
            for index, part in enumerate(parts):
                if part["type"] == "input_image":
                    images.append(part["image_url"])
                    parts[index] = {
                        "type": "input_image",
                        "image_index": len(images) - 1,
                    }
        history.append(record)
    if len(images) > 128:
        raise ValueError("input supports at most 128 images")
    return json.dumps(history, ensure_ascii=False), images


def _native_questions(questions):
    normalized = {}
    for index, item in enumerate(_questions.validate_python(questions)):
        question = {"instructions": item.instructions}
        if isinstance(item, PredicateQuestion):
            question["type"] = "bool"
        elif isinstance(item, ChoiceQuestion):
            # Quote string labels when needed to distinguish true from "true",
            # and to represent an empty string in backends requiring nonempty labels.
            encode = any(
                isinstance(c.value, bool) or c.value == "" for c in item.choices
            )
            question.update(
                type="choice",
                criteria={
                    (
                        json.dumps(c.value, ensure_ascii=False) if encode else c.value
                    ): c.description
                    for c in item.choices
                },
            )
        else:
            question.update(
                type="score",
                criteria=[
                    (
                        f"{level.label}: {level.description}"
                        if level.description
                        else level.label
                    )
                    for level in item.levels
                ],
            )
        normalized[str(index)] = question
    return normalized


def native_request(request):
    """Adapt a validated public request; Python media objects remain untouched."""
    state, images = _native_input(request.get("input"))
    body = {"state": state, "questions": _native_questions(request["questions"])}
    if request.get("model") is not None:
        body["model"] = request["model"]
    additional = request.get("images")
    if images:
        if additional is not None:
            images.extend(
                additional if isinstance(additional, (list, tuple)) else [additional]
            )
        body["images"] = images
    elif additional is not None:
        body["images"] = additional
    for key in ("videos", "audio", "max_length"):
        if request.get(key) is not None:
            body[key] = request[key]
    return body


def cache_namespace(request, tenant=None):
    key = request.get("prompt_cache_key")
    return (tenant, key) if key is not None else tenant


def format_response(request, output):
    questions = _native_questions(request["questions"])
    native, raw = output["response"], output.get("probabilities")
    answers = []
    for index, question in enumerate(request["questions"]):
        key = str(index)
        result = native["answers"][key]
        answer = {"type": question["type"], "name": question.get("name")}
        if result.get("type") == "refusal":
            answers.append({**answer, "type": "refusal"})
            continue
        if raw is not None:
            probabilities = raw[key]
        elif question["type"] == "predicate":
            p = result["probability"]
            probabilities = {"true": p, "false": 1 - p}
        else:
            probabilities = result["probabilities"]
        formatted = systemone_answer(questions[key], probabilities)
        if question["type"] == "predicate":
            answer["probability"] = formatted["noul"]
        elif question["type"] == "choice":
            labels = list(questions[key]["criteria"])
            answer.update(
                choice=question["choices"][labels.index(formatted["choice"])]["value"],
                confidence=formatted["confidence"],
                probabilities=[
                    {
                        "value": option["value"],
                        "probability": formatted["probabilities"][label],
                    }
                    for option, label in zip(question["choices"], labels)
                ],
            )
        else:
            answer.update(
                score=formatted["score"],
                confidence=formatted["confidence"],
                probabilities=[
                    {
                        "value": i,
                        "label": level["label"],
                        "probability": formatted["probabilities"][str(i)],
                    }
                    for i, level in enumerate(question["levels"])
                ],
            )
        answers.append(answer)
    tokens = native.get("usage", {}).get("input_tokens", 0)
    return {
        "model": request.get("model", native.get("model", "")),
        "answers": answers,
        "usage": {
            "input_tokens": tokens,
            "input_tokens_details": {
                "cached_tokens": output.get("cached_tokens", 0),
                "cache_write_tokens": output.get("cache_write_tokens", 0),
            },
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": tokens,
        },
    }
