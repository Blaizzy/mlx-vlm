from enum import Enum
from functools import partial
from typing import Any, Dict, List, Union

from pydantic import BaseModel


class MessageFormat(Enum):
    """Enum for different message format types."""

    LIST_WITH_IMAGE = "list_with_image"
    LIST_WITH_IMAGE_FIRST = "list_with_image_first"
    LIST_WITH_IMAGE_URL_FIRST = "list_with_image_url_first"
    LIST_WITH_IMAGE_TYPE = "list_with_image_type"
    LIST_WITH_IMAGE_TYPE_TEXT = "list_with_image_type_text"
    LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST = "list_with_image_type_text_image_last"
    IMAGE_TOKEN = "image_token"
    IMAGE_TOKEN_PIPE = "image_token_pipe"
    START_IMAGE_TOKEN = "start_image_token"
    IMAGE_TOKEN_NEWLINE = "image_token_newline"
    NUMBERED_IMAGE_TOKENS = "numbered_image_tokens"
    PROMPT_ONLY = "prompt_only"
    PROMPT_WITH_IMAGE_TOKEN = "prompt_with_image_token"
    PROMPT_WITH_START_IMAGE_TOKEN = "prompt_with_start_image_token"
    VIDEO_WITH_TEXT = "video_with_text"


# Model configuration mapping
MODEL_CONFIG = {
    # List with image format models
    "jina_vlm": MessageFormat.IMAGE_TOKEN_PIPE,
    "jvlm": MessageFormat.IMAGE_TOKEN_PIPE,
    "idefics2": MessageFormat.LIST_WITH_IMAGE,
    "idefics3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "lfm2-vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "lfm2_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "aya_vision": MessageFormat.LIST_WITH_IMAGE,
    "cohere2_vision": MessageFormat.LIST_WITH_IMAGE,
    "paddleocr_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen2_vl": MessageFormat.LIST_WITH_IMAGE,
    "qwen2_5_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_vl_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_5": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "qwen3_5_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "mistral3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm4v": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm4v_moe": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "glm_ocr": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "dots_ocr": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "ernie4_5_moe_vl": MessageFormat.LIST_WITH_IMAGE_URL_FIRST,
    "internvl_chat": MessageFormat.LIST_WITH_IMAGE_TYPE,
    "kimi_vl": MessageFormat.LIST_WITH_IMAGE,
    "gemma3": MessageFormat.START_IMAGE_TOKEN,
    "gemma3n": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT,
    "llama4": MessageFormat.LIST_WITH_IMAGE,
    "smolvlm": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "llava": MessageFormat.LIST_WITH_IMAGE,
    "llava_next": MessageFormat.LIST_WITH_IMAGE,
    "mllama": MessageFormat.LIST_WITH_IMAGE,
    "pixtral": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT,
    "molmo2": MessageFormat.LIST_WITH_IMAGE_FIRST,
    # Token-based models
    "llava-qwen2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "llava_qwen2": MessageFormat.IMAGE_TOKEN_NEWLINE,  # fastvlm
    "bunny-llama": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "phi3_v": MessageFormat.NUMBERED_IMAGE_TOKENS,
    "multi_modality": MessageFormat.IMAGE_TOKEN,
    "deepseek_vl_v2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "deepseekocr_2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "deepseekocr": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "hunyuan_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    # Prompt-only models
    "florence2": MessageFormat.PROMPT_ONLY,
    "molmo": MessageFormat.PROMPT_ONLY,
    "paligemma": MessageFormat.PROMPT_WITH_IMAGE_TOKEN,
}

# Models that don't support multi-image
SINGLE_IMAGE_ONLY_MODELS = {
    "llava_next",
    "llava-qwen2",
    "bunny-llama",
    "paligemma",
    "multi_modality",
    "mllama",
}


def extract_text_from_content(content: Any) -> str:
    """
    Extract text from multimodal content.

    When using OpenAI-compatible multimodal API, content can be a list like:
    [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]

    This function extracts only the text parts, preventing base64 image data
    from being tokenized as text (which would cause token explosion).

    Args:
        content: Either a string or a list of content items

    Returns:
        A string containing only the text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                # Extract text from text-type items
                if item_type in ("text", "input_text"):
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        text_parts.append(text)
                # Skip image_url, input_image, input_audio - these are handled separately
        return " ".join(text_parts).strip() if text_parts else ""

    # Fallback: convert to string (shouldn't happen in normal usage)
    return str(content) if content else ""


class MessageBuilder:
    """Builder for creating messages in various formats."""

    @staticmethod
    def text_message(text: str) -> Dict[str, str]:
        """Create a simple text message."""
        return {"type": "text", "text": text, "content": text}

    @staticmethod
    def content_message(content: str) -> Dict[str, str]:
        """Create a content-type text message."""
        return {"type": "text", "text": content, "content": content}

    @staticmethod
    def image_message() -> Dict[str, str]:
        """Create an image message."""
        return {"type": "image"}

    @staticmethod
    def image_url_message() -> Dict[str, str]:
        """Create an image_url message (for models like ERNIE that expect this format)."""
        return {"type": "image_url"}

    @staticmethod
    def audio_message() -> Dict[str, str]:
        """Create an audio message."""
        return {"type": "audio"}

    @staticmethod
    def video_message(
        video_path: str, max_pixels: int = 224 * 224, fps: int = 1
    ) -> Dict[str, Any]:
        """Create a video message."""
        return {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
            "fps": fps,
        }


class MessageFormatter:
    """Handles formatting messages for different model types."""

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.format_type = MODEL_CONFIG.get(self.model_name)
        if not self.format_type:
            raise ValueError(f"Unsupported model: {model_name}")

    def format_message(
        self,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 1,
        num_audios: int = 1,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """Format a message based on the model type."""

        # Check multi-image support
        if num_images > 1 and self.model_name in SINGLE_IMAGE_ONLY_MODELS:
            raise ValueError(
                f"Model {self.model_name} does not support multi-image chat. "
                f"Please only use 1 image."
            )

        # Handle video format for specific models
        if self.model_name in [
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ] and kwargs.get("video"):
            return self._format_video_message(prompt, kwargs)

        # Route to appropriate formatter
        formatter_map = {
            MessageFormat.LIST_WITH_IMAGE: self._format_list_with_image,
            MessageFormat.LIST_WITH_IMAGE_FIRST: partial(
                self._format_list_with_image, image_first=True
            ),
            MessageFormat.LIST_WITH_IMAGE_URL_FIRST: partial(
                self._format_list_with_image, image_first=True, use_image_url=True
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE: self._format_list_with_image_type,
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT: partial(
                self._format_list_with_image_type, message_type="text"
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST: partial(
                self._format_list_with_image_type,
                message_type="text",
                image_first=False,
            ),
            MessageFormat.IMAGE_TOKEN: partial(
                self._format_with_token, token="<image>"
            ),
            MessageFormat.IMAGE_TOKEN_PIPE: partial(
                self._format_with_token, token="<|image|>"
            ),
            MessageFormat.START_IMAGE_TOKEN: partial(
                self._format_with_token, token="<start_of_image>", image_first=False
            ),
            MessageFormat.IMAGE_TOKEN_NEWLINE: partial(
                self._format_with_token, token="<image>\n"
            ),
            MessageFormat.NUMBERED_IMAGE_TOKENS: self._format_numbered_tokens,
            MessageFormat.PROMPT_ONLY: lambda *args, **kw: prompt,
            MessageFormat.PROMPT_WITH_IMAGE_TOKEN: lambda *args, **kw: "<image>"
            * num_images
            + prompt,
            MessageFormat.PROMPT_WITH_START_IMAGE_TOKEN: lambda *args, **kw: prompt
            + "<start_of_image>" * num_images,
            MessageFormat.VIDEO_WITH_TEXT: self._format_video_message,
        }

        formatter = formatter_map.get(self.format_type)
        return formatter(
            prompt,
            role,
            skip_image_token,
            skip_audio_token,
            num_images,
            num_audios,
            **kwargs,
        )

    def _format_list_with_image(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        image_first: bool = False,
        use_image_url: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format as a list with image tokens."""
        content = [MessageBuilder.text_message(prompt)]

        if role == "user" and not skip_image_token and num_images > 0:
            image_builder = (
                MessageBuilder.image_url_message
                if use_image_url
                else MessageBuilder.image_message
            )
            image_tokens = [image_builder()] * num_images
            content = image_tokens + content if image_first else content + image_tokens

        return {"role": role, "content": content}

    def _format_list_with_image_type(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        message_type: str = "content",
        image_first: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format as a list with typed messages."""
        msg_func = (
            MessageBuilder.content_message
            if message_type == "content"
            else MessageBuilder.text_message
        )
        message = {"role": role, "content": [msg_func(prompt)]}

        if role == "user":
            if not skip_image_token and num_images > 0:
                message["content"] = (
                    [MessageBuilder.image_message()] * num_images + message["content"]
                    if image_first
                    else message["content"]
                    + [MessageBuilder.image_message()] * num_images
                )
            if not skip_audio_token and num_audios > 0:
                message["content"] = (
                    message["content"] + [MessageBuilder.audio_message()] * num_audios
                )

        if role == "assistant":
            message["content"] = message["content"][0].get(
                "content", message["content"][0].get("text")
            )

        return message

    def _format_with_token(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        token: str,
        image_first: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format with image tokens in the text."""
        content = prompt

        if role == "user" and not skip_image_token and num_images > 0:
            prefix = token * num_images
            content = f"{prefix}{content}" if image_first else f"{content}{prefix}"

        return {"role": role, "content": content}

    def _format_numbered_tokens(
        self,
        prompt: str,
        role: str,
        skip_image_token: bool,
        skip_audio_token: bool,
        num_images: int,
        num_audios: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format with numbered image tokens."""
        content = prompt

        if role == "user" and not skip_image_token and num_images > 0:
            # phi3_v uses single token regardless of num_images
            prefix = (
                "<|image_1|>"
                if self.model_name == "phi3_v"
                else " ".join([f"<|image_{i+1}|>" for i in range(num_images)])
            )
            content = f"{prefix}{content}"

        return {"role": role, "content": content}

    def _format_video_message(
        self,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format a video message with text."""
        return {
            "role": role,
            "content": [
                MessageBuilder.video_message(
                    kwargs["video"],
                    kwargs.get("max_pixels", 224 * 224),
                    kwargs.get("fps", 1),
                ),
                MessageBuilder.text_message(prompt),
            ],
        }


def get_message_json(
    model_name: str,
    prompt: str,
    role: str = "user",
    skip_image_token: bool = False,
    skip_audio_token: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs,
) -> Union[str, Dict[str, Any]]:
    """
    Get the appropriate JSON message based on the specified model.

    Args:
        model_name: The model for which to generate the message
        prompt: The text prompt to be included in the message
        role: The role of the message (default: "user")
        skip_image_token: Whether to skip adding image tokens
        skip_audio_token: Whether to skip adding audio tokens
        num_images: Number of image tokens to add
        num_audios: Number of audio tokens to add
        **kwargs: Additional arguments (e.g., video path, max_pixels, fps)

    Returns:
        A dictionary or string representing the message for the specified model
    """
    formatter = MessageFormatter(model_name)

    return formatter.format_message(
        prompt,
        role,
        skip_image_token,
        skip_audio_token,
        num_images,
        num_audios,
        **kwargs,
    )


def get_chat_template(
    processor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
    tokenize: bool = False,
    **kwargs,
) -> Any:
    """Apply chat template using processor's tokenizer."""

    def _get_image_token() -> str:
        if processor is None:
            return "<image>"

        image_token = getattr(processor, "image_token", None)
        if isinstance(image_token, str) and image_token:
            return image_token

        tokenizer = getattr(processor, "tokenizer", None)
        image_token = getattr(tokenizer, "image_token", None)
        if isinstance(image_token, str) and image_token:
            return image_token

        return "<image>"

    def _flatten_content(content: Any, image_token: str) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            multimodal_markers = {image_token, "<audio>", "<video>"}
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type in ("text", "input_text"):
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(str(text))
                    elif item_type in ("image", "image_url", "input_image"):
                        parts.append(image_token)
                    elif item_type in ("audio", "input_audio"):
                        parts.append("<audio>")
                    elif item_type == "video":
                        parts.append("<video>")
                    else:
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))

            stitched_parts = []
            prev_is_marker = False
            for part in parts:
                if not part:
                    continue
                current_is_marker = part in multimodal_markers
                if prev_is_marker and not current_is_marker and not part[0].isspace():
                    stitched_parts.append(" ")
                stitched_parts.append(part)
                prev_is_marker = current_is_marker

            return "".join(stitched_parts).strip()

        if isinstance(content, dict):
            text = content.get("text", "") or content.get("content", "")
            return str(text) if text else ""

        return str(content) if content is not None else ""

    def _messages_to_plain_prompt() -> str:
        image_token = _get_image_token()
        normalized = []

        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": "user", "content": message})
                continue

            if isinstance(message, dict):
                normalized.append(
                    {
                        "role": message.get("role", "user"),
                        "content": _flatten_content(
                            message.get("content", ""), image_token
                        ),
                    }
                )
                continue

            normalized.append({"role": "user", "content": str(message)})

        if not normalized:
            return ""

        if len(normalized) == 1 and normalized[0]["role"] == "user":
            return normalized[0]["content"]

        lines = []
        for message in normalized:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in ("system", "user", "assistant"):
                prefix = role.capitalize()
                lines.append(f"{prefix}: {content}" if content else f"{prefix}:")
            else:
                lines.append(content)

        if add_generation_prompt:
            lines.append("Assistant:")

        return "\n".join(lines).strip()

    def _missing_template_error(error: Exception) -> bool:
        message = str(error)
        return (
            "chat_template is not set" in message
            or "no template argument was passed" in message
        )

    chat_template_override = kwargs.get("chat_template", None)

    try:
        template_processor = None
        if (
            processor is not None
            and hasattr(processor, "apply_chat_template")
            and (
                chat_template_override is not None
                or getattr(processor, "chat_template", None) is not None
            )
        ):
            template_processor = processor
        elif (
            processor is not None
            and hasattr(processor, "tokenizer")
            and hasattr(processor.tokenizer, "apply_chat_template")
            and (
                chat_template_override is not None
                or getattr(processor.tokenizer, "chat_template", None) is not None
            )
        ):
            template_processor = processor.tokenizer
        elif processor is not None and hasattr(processor, "apply_chat_template"):
            # Handles tokenizers passed directly.
            if (
                chat_template_override is not None
                or getattr(processor, "chat_template", None) is not None
            ):
                template_processor = processor

        if template_processor is None:
            return _messages_to_plain_prompt()

        try:
            return template_processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except ValueError as e:
            if chat_template_override is None and _missing_template_error(e):
                return _messages_to_plain_prompt()
            raise
    except AttributeError:
        return _messages_to_plain_prompt()


def apply_chat_template(
    processor,
    config: Union[Dict[str, Any], Any],
    prompt: Union[str, Dict[str, Any], List[Any]],
    add_generation_prompt: bool = True,
    return_messages: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs,
) -> Union[List[Dict[str, Any]], str, Any]:
    """
    Apply chat template to prompts.

    Args:
        processor: The processor with chat template functionality
        config: Model configuration
        prompt: Single prompt string, dict, or list of prompts
        add_generation_prompt: Whether to add generation prompt
        return_messages: Whether to return messages list instead of template
        num_images: Number of images in the input
        num_audios: Number of audio files in the input
        **kwargs: Additional arguments for message formatting

    Returns:
        Formatted messages or chat template
    """
    config = config if isinstance(config, dict) else config.__dict__
    model_type = config["model_type"]

    # Build messages from prompts
    messages = []

    if isinstance(prompt, str):
        # Single string prompt
        messages.append(
            get_message_json(
                model_type,
                prompt,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )
        )
    elif isinstance(prompt, dict):
        # Single dict prompt
        content = extract_text_from_content(prompt["content"])
        messages.append(
            get_message_json(
                model_type,
                content,
                prompt["role"],
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )
        )
    elif isinstance(prompt, list):
        # List of prompts
        for i, p in enumerate(prompt):
            if isinstance(p, str):
                is_first = i == 0
                messages.append(
                    get_message_json(
                        model_type,
                        p,
                        skip_image_token=not is_first,
                        skip_audio_token=not is_first,
                        num_images=num_images,
                        num_audios=num_audios,
                        **kwargs,
                    )
                )
            elif isinstance(p, dict) or isinstance(p, BaseModel):
                role = "user"
                content = ""
                if isinstance(p, dict):
                    role = p.get("role", "user")
                    content = p.get("content")
                else:
                    role = p.role
                    content = p.content
                # Handle multimodal content: extract only text, skip image/audio URLs
                # This prevents base64 image data from being tokenized as text
                content = extract_text_from_content(content)
                is_first = i == 0 or (i == 1 and role not in ["system", "assistant"])
                messages.append(
                    get_message_json(
                        model_type,
                        content,
                        role,
                        skip_image_token=not is_first
                        or role in ["system", "assistant"],
                        skip_audio_token=not is_first
                        or role in ["system", "assistant"],
                        num_images=num_images,
                        num_audios=num_audios,
                        **kwargs,
                    )
                )

    if return_messages:
        return messages

    # Some models only need the last message
    if model_type in ["paligemma", "molmo", "florence2"]:
        return messages[-1]

    return get_chat_template(processor, messages, add_generation_prompt)
