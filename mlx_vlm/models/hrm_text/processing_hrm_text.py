from typing import Any, Optional

import mlx.core as mx
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch

DIRECT_CONDITION = "<|object_ref_start|>"
SYNTH_COT_CONDITION = "<|quad_end|><|object_ref_end|>"

DEFAULT_CHAT_TEMPLATE = (
    "{%- set condition = '<|quad_end|><|object_ref_end|>' "
    "if enable_thinking else '<|object_ref_start|>' -%}"
    "{{- '<|im_start|>' + condition -}}"
    "{%- for message in messages -%}"
    "{{- message['content'] -}}"
    "{%- if not loop.last -%}{{- '\\n' -}}{%- endif -%}"
    "{%- endfor -%}"
    "{{- '<|im_end|>' -}}"
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("content", "")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", "") or content.get("content", ""))
    return "" if content is None else str(content)


class HrmTextProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        self.model_type = "hrm_text"
        self.tokenizer.chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
        super().__init__(
            tokenizer,
            chat_template=self.tokenizer.chat_template,
            **kwargs,
        )

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        condition: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(conversation, (str, dict)):
            conversation = [conversation]

        texts = []
        for message in conversation:
            if isinstance(message, str):
                texts.append(message)
            elif isinstance(message, dict):
                texts.append(_content_to_text(message.get("content", "")))
            elif message is not None:
                texts.append(str(message))

        condition = condition or (
            SYNTH_COT_CONDITION if enable_thinking else DIRECT_CONDITION
        )
        rendered = f"<|im_start|>{condition}{chr(10).join(texts)}<|im_end|>"
        if not tokenize:
            return rendered
        return self.tokenizer(rendered, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        inputs = self.tokenizer(*args, **kwargs)
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None and "token_type_ids" not in inputs:
            inputs["token_type_ids"] = mx.ones_like(mx.array(input_ids))
        return inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        chat_template = kwargs.pop("chat_template", None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
        except (AttributeError, ValueError):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
        return cls(tokenizer=tokenizer, chat_template=chat_template)


install_auto_processor_patch("hrm_text", HrmTextProcessor)

__all__ = [
    "DEFAULT_CHAT_TEMPLATE",
    "DIRECT_CONDITION",
    "HrmTextProcessor",
    "SYNTH_COT_CONDITION",
]
