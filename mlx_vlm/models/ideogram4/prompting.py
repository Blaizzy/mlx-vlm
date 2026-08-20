from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd
from typing import Any

_HEX_COLOR_RE = re.compile(r"^#[0-9A-F]{6}$")

_COLOR_PALETTE_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "pattern": r"^#[0-9A-F]{6}$"},
}
_BBOX_SCHEMA = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}
_OBJECT_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["obj"]},
        "bbox": _BBOX_SCHEMA,
        "desc": {"type": "string", "minLength": 1},
        "color_palette": {**_COLOR_PALETTE_SCHEMA, "maxItems": 5},
    },
    "required": ["type", "desc"],
    "additionalProperties": False,
}
_TEXT_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["text"]},
        "bbox": _BBOX_SCHEMA,
        "text": {"type": "string"},
        "desc": {"type": "string", "minLength": 1},
        "color_palette": {**_COLOR_PALETTE_SCHEMA, "maxItems": 5},
    },
    "required": ["type", "text", "desc"],
    "additionalProperties": False,
}
_PHOTO_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "aesthetics": {"type": "string", "minLength": 1},
        "lighting": {"type": "string", "minLength": 1},
        "photo": {"type": "string", "minLength": 1},
        "medium": {"type": "string", "minLength": 1},
        "color_palette": {**_COLOR_PALETTE_SCHEMA, "maxItems": 16},
    },
    "required": ["aesthetics", "lighting", "photo", "medium"],
    "additionalProperties": False,
}
_ART_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "aesthetics": {"type": "string", "minLength": 1},
        "lighting": {"type": "string", "minLength": 1},
        "medium": {"type": "string", "minLength": 1},
        "art_style": {"type": "string", "minLength": 1},
        "color_palette": {**_COLOR_PALETTE_SCHEMA, "maxItems": 16},
    },
    "required": ["aesthetics", "lighting", "medium", "art_style"],
    "additionalProperties": False,
}
IDEOGRAM4_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "high_level_description": {"type": "string", "minLength": 1},
        "style_description": {"anyOf": [_PHOTO_STYLE_SCHEMA, _ART_STYLE_SCHEMA]},
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string", "minLength": 1},
                "elements": {
                    "type": "array",
                    "items": {"anyOf": [_OBJECT_ELEMENT_SCHEMA, _TEXT_ELEMENT_SCHEMA]},
                },
            },
            "required": ["background", "elements"],
            "additionalProperties": False,
        },
    },
    "required": ["compositional_deconstruction"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class NormalizedPrompt:
    text: str
    is_json_caption: bool
    is_structured_caption: bool
    was_wrapped: bool
    warnings: tuple[str, ...] = ()
    prompt_expansion_model: str | None = None
    prompt_expansion_used: bool = False
    prompt_expansion_error: str | None = None


@dataclass(frozen=True, slots=True)
class PromptExpansionResult:
    text: str
    raw_text: str
    model: str


class PromptExpansionCaptionError(ValueError):
    pass


def format_caption(caption: Mapping[str, Any]) -> str:
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def is_structured_caption(prompt: str) -> bool:
    caption = _parse_json_caption(prompt)
    if caption is None:
        return False
    return not _caption_warnings(caption)


def normalize_prompt(
    prompt: str,
    *,
    auto_json_caption: bool = True,
    warn: bool = True,
) -> NormalizedPrompt:
    stripped = prompt.strip()
    if _looks_like_json_object(stripped):
        caption = _loads_json_caption(stripped)
        issues = tuple(_caption_warnings(caption))
        if warn:
            for issue in issues:
                warnings.warn(issue, stacklevel=2)
        return NormalizedPrompt(
            text=prompt,
            is_json_caption=True,
            is_structured_caption=not issues,
            was_wrapped=False,
            warnings=issues,
        )
    if not auto_json_caption:
        return NormalizedPrompt(
            text=prompt,
            is_json_caption=False,
            is_structured_caption=False,
            was_wrapped=False,
        )
    return NormalizedPrompt(
        text=format_caption(_minimal_caption(stripped)),
        is_json_caption=True,
        is_structured_caption=True,
        was_wrapped=True,
    )


def prepare_prompt(
    prompt: str,
    *,
    auto_json_caption: bool = True,
    prompt_expansion_model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    warn: bool = True,
) -> NormalizedPrompt:
    stripped = prompt.strip()
    if _looks_like_json_object(stripped) or prompt_expansion_model is None:
        return normalize_prompt(
            prompt,
            auto_json_caption=auto_json_caption,
            warn=warn,
        )

    try:
        expansion = generate_prompt_expansion_caption(
            stripped,
            model=prompt_expansion_model,
            aspect_ratio=_aspect_ratio_from_size(width, height),
        )
        prepared = normalize_prompt(
            expansion.text,
            auto_json_caption=False,
            warn=warn,
        )
        return NormalizedPrompt(
            text=prepared.text,
            is_json_caption=prepared.is_json_caption,
            is_structured_caption=prepared.is_structured_caption,
            was_wrapped=False,
            warnings=prepared.warnings,
            prompt_expansion_model=expansion.model,
            prompt_expansion_used=True,
        )
    except PromptExpansionCaptionError as exc:
        if not auto_json_caption:
            raise ValueError("Prompt expansion failed") from exc
        if warn:
            warnings.warn(
                "Prompt expansion failed; falling back to the minimal "
                f"Ideogram 4 JSON caption wrapper. {exc}",
                stacklevel=2,
            )
        fallback = normalize_prompt(
            prompt,
            auto_json_caption=True,
            warn=warn,
        )
        return NormalizedPrompt(
            text=fallback.text,
            is_json_caption=fallback.is_json_caption,
            is_structured_caption=fallback.is_structured_caption,
            was_wrapped=fallback.was_wrapped,
            warnings=fallback.warnings,
            prompt_expansion_model=str(prompt_expansion_model),
            prompt_expansion_used=False,
            prompt_expansion_error=str(exc),
        )


def generate_prompt_expansion_caption(
    prompt: str,
    *,
    model: str,
    aspect_ratio: str | None = None,
) -> PromptExpansionResult:
    import gc

    import mlx.core as mx

    from ...generate.dispatch import generate
    from ...prompt_utils import apply_chat_template
    from ...structured import build_json_schema_logits_processor
    from ...utils import load

    model_obj = None
    processor = None
    try:
        model_obj, processor = load(model)
        messages = [
            {"role": "system", "content": PROMPT_EXPANSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _prompt_expansion_user_prompt(prompt, aspect_ratio),
            },
        ]
        formatted_prompt = apply_chat_template(
            processor,
            model_obj.config,
            messages,
        )
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        logits_processor = build_json_schema_logits_processor(
            tokenizer,
            IDEOGRAM4_CAPTION_SCHEMA,
        )
        result = generate(
            model_obj,
            processor,
            formatted_prompt,
            logits_processors=[logits_processor],
            verbose=False,
            skip_special_tokens=True,
        )
        raw_text = result.text.strip()
        return PromptExpansionResult(
            text=format_caption(_load_prompt_expansion_caption(raw_text)),
            raw_text=raw_text,
            model=str(model),
        )
    finally:
        del model_obj, processor
        gc.collect()
        mx.clear_cache()


PROMPT_EXPANSION_SYSTEM_PROMPT = """\
You prepare structured JSON captions for Ideogram 4 image generation. Return
only JSON matching the provided schema. Preserve the user's intent, requested
wording, and constraints while making the visual description more specific and
useful to the image model.

Always include a concrete high_level_description and
compositional_deconstruction. Write descriptions as observations of the desired
image, never as commands or as a copy of the user's request. The background must
describe the actual scene, not a generic placeholder.

Use one obj element for each explicitly named visual subject. Use one text
element for every quoted string or other visible wording the user requests.
Copy each text field verbatim, including capitalization, punctuation, line
breaks, and non-ASCII characters. Do not hide requested lettering inside an obj
description.

Bounding boxes are optional. Include them only when useful for layout, using
integer normalized [0, 1000] coordinates as [y_min, x_min, y_max, x_max] with
y_min < y_max and x_min < x_max. If style_description is included, use exactly
one of photo or art_style. Use only uppercase #RRGGBB values in color palettes.
"""


def _prompt_expansion_user_prompt(prompt: str, aspect_ratio: str | None) -> str:
    aspect = (
        f"\nTarget aspect ratio: {aspect_ratio}. Use it only to plan the "
        "composition; do not add an aspect_ratio field."
        if aspect_ratio
        else ""
    )
    return f"Convert this prompt into an Ideogram 4 JSON caption:{aspect}\n{prompt}"


def _aspect_ratio_from_size(width: int | None, height: int | None) -> str | None:
    if not width or not height:
        return None
    divisor = gcd(int(width), int(height))
    return f"{int(width) // divisor}:{int(height) // divisor}"


def _looks_like_json_object(prompt: str) -> bool:
    return prompt.startswith("{")


def _parse_json_caption(prompt: str) -> dict[str, Any] | None:
    stripped = prompt.strip()
    if not _looks_like_json_object(stripped):
        return None
    try:
        return _loads_json_caption(stripped)
    except ValueError:
        return None


def _loads_json_caption(prompt: str) -> dict[str, Any]:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid Ideogram 4 JSON caption") from exc
    if not isinstance(value, dict):
        raise ValueError("Ideogram 4 JSON caption must be an object")
    return value


def _load_prompt_expansion_caption(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PromptExpansionCaptionError(
            "Prompt expansion model did not return valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PromptExpansionCaptionError(
            "Prompt expansion model did not return a JSON object"
        )
    issues = _caption_warnings(value)
    if issues:
        raise PromptExpansionCaptionError(
            f"Prompt expansion model returned an invalid caption: {issues[0]}"
        )
    return value


def _minimal_caption(prompt: str) -> dict[str, Any]:
    return {
        "high_level_description": prompt,
        "compositional_deconstruction": {
            "background": (
                "The setting, environment, and surrounding context implied by "
                "the prompt."
            ),
            "elements": [{"type": "obj", "desc": prompt}],
        },
    }


def _caption_warnings(caption: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    compositional = caption.get("compositional_deconstruction")
    if not isinstance(compositional, Mapping):
        issues.append(
            "Ideogram 4 JSON caption should include a "
            "'compositional_deconstruction' object."
        )
    else:
        if not _is_non_empty_string(compositional.get("background")):
            issues.append(
                "Ideogram 4 JSON caption should include "
                "'compositional_deconstruction.background' as a non-empty string."
            )
        elements = compositional.get("elements")
        if not isinstance(elements, list):
            issues.append(
                "Ideogram 4 JSON caption should include "
                "'compositional_deconstruction.elements' as a list."
            )
        else:
            for idx, element in enumerate(elements):
                issues.extend(_element_warnings(element, idx))

    style = caption.get("style_description")
    if isinstance(style, Mapping):
        has_photo = "photo" in style
        has_art_style = "art_style" in style
        if has_photo == has_art_style:
            issues.append(
                "Ideogram 4 JSON caption 'style_description' should include "
                "exactly one of 'photo' or 'art_style'."
            )
        required = ("aesthetics", "lighting", "medium")
        for key in required:
            if not _is_non_empty_string(style.get(key)):
                issues.append(
                    "Ideogram 4 JSON caption 'style_description' should include "
                    f"'{key}' as a non-empty string."
                )
        if has_photo and not _is_non_empty_string(style.get("photo")):
            issues.append(
                "Ideogram 4 JSON caption 'style_description.photo' should be "
                "a non-empty string."
            )
        if has_art_style and not _is_non_empty_string(style.get("art_style")):
            issues.append(
                "Ideogram 4 JSON caption 'style_description.art_style' should be "
                "a non-empty string."
            )
    elif style is not None:
        issues.append(
            "Ideogram 4 JSON caption 'style_description' should be an object."
        )

    issues.extend(_color_palette_warnings(caption))
    return issues


def _element_warnings(value: Any, idx: int) -> list[str]:
    path = f"compositional_deconstruction.elements[{idx}]"
    if not isinstance(value, Mapping):
        return [f"Ideogram 4 JSON caption '{path}' should be an object."]

    issues: list[str] = []
    element_type = value.get("type")
    if element_type not in {"obj", "text"}:
        issues.append(
            f"Ideogram 4 JSON caption '{path}.type' should be 'obj' or 'text'."
        )
    if not _is_non_empty_string(value.get("desc")):
        issues.append(
            f"Ideogram 4 JSON caption '{path}.desc' should be a non-empty string."
        )
    if element_type == "text" and not isinstance(value.get("text"), str):
        issues.append(f"Ideogram 4 JSON caption '{path}.text' should be a string.")
    if "bbox" in value:
        issues.extend(_bbox_warnings(value["bbox"], f"{path}.bbox"))
    return issues


def _bbox_warnings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 4:
        return [
            f"Ideogram 4 JSON caption '{path}' should contain four integer "
            "coordinates."
        ]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in value
    ):
        return [
            f"Ideogram 4 JSON caption '{path}' coordinates should be integers "
            "between 0 and 1000."
        ]
    y_min, x_min, y_max, x_max = value
    if y_min >= y_max or x_min >= x_max:
        return [
            f"Ideogram 4 JSON caption '{path}' should satisfy y_min < y_max and "
            "x_min < x_max."
        ]
    return []


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _color_palette_warnings(value: Any, path: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if key == "color_palette":
                issues.extend(_validate_color_palette(nested, nested_path))
            else:
                issues.extend(_color_palette_warnings(nested, nested_path))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            issues.extend(_color_palette_warnings(nested, f"{path}[{idx}]"))
    return issues


def _validate_color_palette(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        return [f"Ideogram 4 JSON caption '{path}' should be a list of hex colors."]
    issues = []
    for idx, color in enumerate(value):
        if not isinstance(color, str) or _HEX_COLOR_RE.fullmatch(color) is None:
            issues.append(
                "Ideogram 4 JSON caption "
                f"'{path}[{idx}]' should be an uppercase #RRGGBB hex color."
            )
    return issues
