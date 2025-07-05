from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Union


class MessageFormat(Enum):
    """Enum for different message format types."""

    LIST_WITH_IMAGE = "list_with_image"
    LIST_WITH_IMAGE_FIRST = "list_with_image_first"
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
    "idefics2": MessageFormat.LIST_WITH_IMAGE,
    "idefics3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "aya_vision": MessageFormat.LIST_WITH_IMAGE,
    "qwen2_vl": MessageFormat.LIST_WITH_IMAGE,
    "qwen2_5_vl": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "mistral3": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "internvl_chat": MessageFormat.LIST_WITH_IMAGE_TYPE,
    "kimi_vl": MessageFormat.LIST_WITH_IMAGE,
    "gemma3": MessageFormat.START_IMAGE_TOKEN,
    "gemma3n": MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST,
    "llama4": MessageFormat.LIST_WITH_IMAGE,
    "smolvlm": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "llava": MessageFormat.LIST_WITH_IMAGE,
    "llava_next": MessageFormat.LIST_WITH_IMAGE,
    "mllama": MessageFormat.LIST_WITH_IMAGE,
    "pixtral": MessageFormat.LIST_WITH_IMAGE_TYPE,
    # Token-based models
    "llava-qwen2": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "bunny-llama": MessageFormat.IMAGE_TOKEN_NEWLINE,
    "phi3_v": MessageFormat.NUMBERED_IMAGE_TOKENS,
    "multi_modality": MessageFormat.IMAGE_TOKEN,
    "deepseek_vl_v2": MessageFormat.IMAGE_TOKEN_NEWLINE,
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


class MessageBuilder:
    """Builder for creating messages in various formats."""

    @staticmethod
    def text_message(text: str) -> Dict[str, str]:
        """Create a simple text message."""
        return {"type": "text", "text": text}

    @staticmethod
    def content_message(content: str) -> Dict[str, str]:
        """Create a content-type text message."""
        return {"type": "text", "content": content}

    @staticmethod
    def image_message() -> Dict[str, str]:
        """Create an image message."""
        return {"type": "image"}

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
        if self.model_name in ["qwen2_vl", "qwen2_5_vl"] and kwargs.get("video"):
            return self._format_video_message(prompt, kwargs)

        # Route to appropriate formatter
        formatter_map = {
            MessageFormat.LIST_WITH_IMAGE: self._format_list_with_image,
            MessageFormat.LIST_WITH_IMAGE_FIRST: partial(
                self._format_list_with_image, image_first=True
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
        **kwargs,
    ) -> Dict[str, Any]:
        """Format as a list with image tokens."""
        content = [MessageBuilder.text_message(prompt)]

        if role == "user" and not skip_image_token:
            image_tokens = [MessageBuilder.image_message()] * num_images
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
            if not skip_image_token:
                message["content"] = (
                    [MessageBuilder.image_message()] * num_images + message["content"]
                    if image_first
                    else message["content"]
                    + [MessageBuilder.image_message()] * num_images
                )
            if not skip_audio_token:
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

        if role == "user" and not skip_image_token:
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

        if role == "user" and not skip_image_token:
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
    try:
        processor = (
            processor
            if "chat_template" in processor.__dict__.keys()
            else processor.tokenizer
        )

        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    except AttributeError:
        raise ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )


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
        messages.append(
            get_message_json(
                model_type,
                prompt["content"],
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
            elif isinstance(p, dict):
                role = p.get("role", "user")
                is_first = i == 0 or (i == 1 and role not in ["system", "assistant"])
                messages.append(
                    get_message_json(
                        model_type,
                        p["content"],
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
