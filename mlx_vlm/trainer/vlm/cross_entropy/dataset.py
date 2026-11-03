"""Model-aware dataset preparation for vision-language SFT."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlx.core as mx
import numpy as np


def _config_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    raise TypeError("model config must be a mapping or expose its attributes")


def _load_vlm_helpers():
    """Import mlx-vlm utilities only when a dataset is actually prepared."""
    try:
        from mlx_vlm.prompt_utils import MODEL_CONFIG, apply_chat_template
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback
    except ImportError as error:
        raise ImportError(
            "VLM dataset preparation requires `mlx-vlm`. Install mlx-vlm and its "
            "model dependencies before creating a VisionDataset."
        ) from error
    return (
        MODEL_CONFIG,
        apply_chat_template,
        prepare_inputs,
        process_inputs_with_fallback,
    )


def _as_media_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_media(value: Any, base_path: Path | None, *, image: bool = False) -> Any:
    """Resolve local media references in JSONL/Hugging Face rows."""
    if isinstance(value, Mapping):
        if image and value.get("bytes") is not None:
            from PIL import Image

            with Image.open(BytesIO(value["bytes"])) as source:
                return source.convert("RGB")
        if not image and value.get("array") is not None:
            return value["array"]
        path = value.get("path")
        if path is not None:
            return _resolve_media(path, base_path, image=image)
        if value.get("bytes") is not None:
            return BytesIO(value["bytes"])
        return value

    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if value.startswith(("http://", "https://", "data:")):
            return value
        path = (
            Path(unquote(urlparse(value).path))
            if value.startswith("file://")
            else Path(value)
        )
    else:
        return value

    if not path.is_absolute() and base_path is not None:
        path = base_path / path
    if not path.exists():
        if base_path is not None:
            raise FileNotFoundError(f"Media file does not exist: {path}")
        return str(value)
    return str(path)


def _media_value(
    row: Mapping[str, Any],
    configured_key: str | None,
    alternatives: tuple[str, ...],
) -> Any:
    keys = ((configured_key,) if configured_key else ()) + alternatives
    return next((row[key] for key in keys if key in row and row[key] is not None), None)


def _to_mlx(inputs: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in inputs.items():
        if value is None or isinstance(value, mx.array):
            result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = mx.array(value)
        elif isinstance(value, (list, tuple)):
            try:
                result[key] = mx.array(np.asarray(value))
            except (TypeError, ValueError):
                result[key] = value
        else:
            result[key] = value
    return result


def _image_token_index(config: Mapping[str, Any]) -> int | None:
    token_id = config.get("image_token_index")
    if token_id is None:
        token_id = config.get("image_token_id")
    return None if token_id is None else int(token_id)


def _sequence(value: Any, field: str) -> mx.array:
    array = value if isinstance(value, mx.array) else mx.array(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(
            f"Each VLM dataset row must produce one {field} sequence, got "
            f"shape {array.shape}. Split batched conversations into separate rows."
        )
    return array


class VisionDataset:
    """Prepare chat rows using the model's mlx-vlm processor.

    Rows use ``messages`` or ``conversations`` plus optional ``image``/``images``,
    ``audio``/``audios`` and ``video``/``videos`` fields. For JSONL datasets,
    relative media paths are resolved against the split's ``base_path``.
    """

    def __init__(
        self,
        data,
        model_config: Any,
        processor: Any,
        config: Any | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        if isinstance(data, Mapping):
            raise TypeError(
                "VisionDataset expects one split of rows, not a split mapping. "
                "Pass e.g. load(source)['train']."
            )
        if not hasattr(data, "__len__") or not hasattr(data, "__getitem__"):
            raise TypeError("VisionDataset requires a finite, indexable dataset.")

        self.dataset = data
        self.config = _config_mapping(model_config)
        self.processor = processor
        self.image_resize_shape = _get_option(config, "image_resize_shape")
        self.train_on_completions = bool(
            _get_option(config, "train_on_completions", False)
        )
        configured_base = (
            base_path if base_path is not None else getattr(data, "base_path", None)
        )
        self.base_path = Path(configured_base) if configured_base is not None else None
        self.message_feature = _get_option(config, "message_feature")
        self.prompt_feature = _get_option(config, "prompt_feature") or "prompt"
        self.completion_feature = (
            _get_option(config, "completion_feature") or "completion"
        )
        self.text_feature = _get_option(config, "text_feature") or "text"
        self.tools_feature = _get_option(config, "tools_feature") or "tools"
        self.chat_template_kwargs = dict(
            _get_option(config, "chat_template_kwargs") or {}
        )
        self.image_feature = _get_option(config, "image_feature")
        self.audio_feature = _get_option(config, "audio_feature")
        self.video_feature = _get_option(config, "video_feature")
        self._media_fields = (
            ("image", self.image_feature, ("images", "image")),
            ("audio", self.audio_feature, ("audio", "audios")),
            ("video", self.video_feature, ("video", "videos")),
        )

        (
            self._model_configurations,
            self._apply_chat_template,
            self._prepare_inputs,
            self._process_inputs,
        ) = _load_vlm_helpers()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.process(self.dataset[index])

    def media_signature(self, index: int) -> tuple[str, ...]:
        """Return present modalities without decoding media for batch grouping."""
        row = self.dataset[index]
        return tuple(
            name
            for name, key, alternatives in self._media_fields
            if _as_media_list(_media_value(row, key, alternatives))
        )

    def _conversation(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        keys = ((self.message_feature,) if self.message_feature else ()) + (
            "messages",
            "conversations",
            "conversation",
        )
        conversation = next((row[key] for key in keys if key in row), None)
        if conversation is None:
            conversation = self._pair_conversation(row)
        conversation = _maybe_json(conversation, "VLM conversation strings")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError(
                "Each VLM row must contain a non-empty `messages` or "
                "`conversations` list, or a prompt/completion pair."
            )
        if isinstance(conversation[0], list):
            raise ValueError(
                "A dataset row contains multiple conversations. Split them into "
                "separate rows so each conversation retains its media alignment."
            )
        try:
            conversation = [
                json.loads(message) if isinstance(message, str) else message
                for message in conversation
            ]
        except json.JSONDecodeError as error:
            raise ValueError("String messages must be valid JSON.") from error
        return [_normalize_message(message) for message in conversation]

    def _pair_conversation(self, row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        """Build a two-turn chat from prompt/completion style columns."""
        pairs = (
            (self.prompt_feature, self.completion_feature),
            ("question", "answer"),
            ("instruction", "output"),
            ("input", "output"),
        )
        for prompt_key, completion_key in pairs:
            if prompt_key in row and completion_key in row:
                prompt = row[prompt_key]
                if prompt_key == "instruction" and row.get("input"):
                    prompt = f"{prompt}\n\n{row['input']}"
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": row[completion_key]},
                ]
                system = row.get("system")
                return ([{"role": "system", "content": system}] if system else []) + (
                    conversation
                )
        return None

    def _template_kwargs(self, row: Mapping[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.chat_template_kwargs)
        tools = _maybe_json(row.get(self.tools_feature), "Tool definitions")
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _render(self, conversation, add_generation_prompt, images, audio, kwargs):
        render = partial(
            self._apply_chat_template,
            self.processor,
            self.config,
            conversation,
            add_generation_prompt=add_generation_prompt,
            num_images=len(images),
            num_audios=len(audio),
            **kwargs,
        )
        has_extra_fields = any(set(row) - {"role", "content"} for row in conversation)
        if (
            not has_extra_fields
            or self.config.get("model_type") in _LAST_MESSAGE_MODEL_TYPES
        ):
            return render()
        return _render_with_row_fields(
            self.processor,
            conversation,
            render(return_messages=True),
            add_generation_prompt,
            kwargs,
        )

    def _prepare(
        self,
        prompt: str,
        images: list[Any],
        audio: list[Any],
        videos: list[Any],
    ) -> dict[str, Any]:
        model_type = self.config.get("model_type")
        kwargs = dict(
            processor=self.processor,
            prompts=[prompt],
            images=images or None,
            audio=audio or None,
            videos=videos or None,
            add_special_tokens=False,
        )
        if model_type in self._model_configurations:
            try:
                inputs = self._process_inputs(**kwargs)
            except Exception as native_error:
                try:
                    inputs = self._prepare_inputs(
                        **kwargs,
                        image_token_index=_image_token_index(self.config),
                        resize_shape=self.image_resize_shape,
                    )
                except Exception as fallback_error:
                    raise ValueError(
                        "Both mlx-vlm native preprocessing and its generic "
                        "preprocessing fallback failed for this VLM row."
                    ) from fallback_error
                if not isinstance(inputs, Mapping):
                    raise ValueError(
                        "mlx-vlm preprocessing returned a non-mapping result."
                    ) from native_error
        else:
            inputs = self._prepare_inputs(
                **kwargs,
                image_token_index=_image_token_index(self.config),
                resize_shape=self.image_resize_shape,
            )

        if "images" in inputs and "pixel_values" not in inputs:
            inputs = dict(inputs)
            inputs["pixel_values"] = inputs.pop("images")
        inputs = _to_mlx(inputs)
        if "input_ids" not in inputs:
            raise ValueError("The mlx-vlm processor did not return `input_ids`.")
        if images and not any(
            inputs.get(key) is not None
            for key in ("pixel_values", "image_embeds", "images")
        ):
            raise ValueError(
                "The mlx-vlm processor received images but returned no image "
                "features. Check the model processor and image field format."
            )
        if audio and not any(
            inputs.get(key) is not None
            for key in (
                "audio_features",
                "input_features",
                "input_values",
                "features",
                "audio",
                "audios",
            )
        ):
            raise ValueError(
                "The mlx-vlm processor received audio but returned no audio "
                "features. Check that this model and processor support audio."
            )
        if videos and not any(
            inputs.get(key) is not None
            for key in ("videos", "video_values", "pixel_values", "video_grid_thw")
        ):
            raise ValueError(
                "The mlx-vlm processor received video but returned no video "
                "features. Check that this model and processor support video."
            )
        input_ids = _sequence(inputs["input_ids"], "input_ids")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        else:
            attention_mask = _sequence(attention_mask, "attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Processor input_ids and attention_mask shapes differ.")
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        return inputs

    def process(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError("VLM dataset rows must be mappings.")
        conversation = None
        if self.text_feature not in row or any(
            key in row
            for key in (
                self.message_feature,
                "messages",
                "conversations",
                "conversation",
            )
            if key
        ):
            conversation = self._conversation(row)
        images, audio, videos = (
            [
                _resolve_media(item, self.base_path, image=name == "image")
                for item in _as_media_list(_media_value(row, key, alternatives))
            ]
            for name, key, alternatives in self._media_fields
        )
        if conversation is None:
            # Pre-rendered text is used as-is, bypassing the chat template.
            if self.train_on_completions:
                raise ValueError(
                    "train_on_completions needs chat or prompt/completion rows, "
                    f"not pre-rendered `{self.text_feature}` rows."
                )
            return self._prepare(row[self.text_feature], images, audio, videos)
        template_kwargs = self._template_kwargs(row)
        prompt = self._render(conversation, False, images, audio, template_kwargs)
        inputs = self._prepare(prompt, images, audio, videos)

        if self.train_on_completions:
            if (
                not isinstance(conversation[-1], Mapping)
                or conversation[-1].get("role") != "assistant"
            ):
                raise ValueError(
                    "train_on_completions requires the last message in every "
                    "conversation to have role='assistant'."
                )
            prefix = self._render(
                conversation[:-1], True, images, audio, template_kwargs
            )
            prefix_inputs = self._prepare(prefix, images, audio, videos)
            prefix_length = int(prefix_inputs["attention_mask"].sum().item())
            full_attention = np.asarray(inputs["attention_mask"])
            active_positions = np.flatnonzero(full_attention)
            valid_start = int(active_positions[0]) if active_positions.size else 0
            if prefix_length > inputs["input_ids"].shape[0]:
                raise ValueError(
                    "The tokenized assistant prefix is longer than the full "
                    "conversation; processor chat templates are not prefix-stable."
                )
            completion_mask = mx.arange(inputs["input_ids"].shape[0]) >= (
                valid_start + prefix_length
            )
            completion_mask = mx.logical_and(
                completion_mask,
                inputs["attention_mask"].astype(mx.bool_),
            )
            if int(completion_mask.sum().item()) == 0:
                raise ValueError(
                    "The final assistant message produced no completion tokens."
                )
            inputs["completion_mask"] = completion_mask.astype(mx.int32)

        return inputs


# mlx-vlm renders only the last message for these models.
_LAST_MESSAGE_MODEL_TYPES = {"paligemma", "florence2", "falcon_ocr"}
_ROLE_KEYS = ("role", "from", "speaker", "author")
_CONTENT_KEYS = ("content", "value", "text", "message")
_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}


def _normalize_message(message: Any) -> dict[str, Any]:
    """Add canonical role/content while keeping every original field.

    Custom chat templates may read ShareGPT keys (``from``/``value``) or extra
    fields such as ``reasoning_content``; those stay untouched.
    """
    if not isinstance(message, Mapping):
        raise ValueError("Conversation messages must be mappings.")
    message = dict(message)
    role = next((message[k] for k in _ROLE_KEYS if message.get(k) is not None), None)
    content = next(
        (message[k] for k in _CONTENT_KEYS if message.get(k) is not None), None
    )
    if content is None and (
        message.get("tool_calls") or message.get("function_call") or role == "tool"
    ):
        content = ""
    if role is None or content is None:
        raise ValueError(
            "Each conversation message needs a role (one of "
            f"{', '.join(_ROLE_KEYS)}) and content (one of "
            f"{', '.join(_CONTENT_KEYS)}); got fields {sorted(message)}."
        )
    message["role"] = _ROLE_ALIASES.get(str(role).lower(), role)
    message["content"] = content
    return message


def _render_with_row_fields(processor, rows, messages, add_generation_prompt, kwargs):
    """Render mlx-vlm formatted messages with each row's extra fields restored.

    mlx-vlm rebuilds VLM messages from role/content, so fields that custom chat
    templates read (``from``, ``reasoning_content``, ...) are merged back first.
    """
    from mlx_vlm.prompt_utils import get_chat_template

    messages = [
        {**row, **message} if isinstance(message, Mapping) else message
        for row, message in zip(rows, messages)
    ]
    return get_chat_template(processor, messages, add_generation_prompt, **kwargs)


def _maybe_json(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"{name} must be valid JSON.") from None


def _get_option(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


__all__ = ["VisionDataset"]
