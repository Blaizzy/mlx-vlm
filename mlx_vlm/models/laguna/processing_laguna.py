from typing import Any, Optional

from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch


class LagunaProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, tokenizer, chat_template: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        self.model_type = "laguna"
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        super().__init__(
            tokenizer,
            chat_template=getattr(self.tokenizer, "chat_template", None),
            **kwargs,
        )

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        chat_template = kwargs.pop("chat_template", None)
        tokenizer_kwargs = _collect_tokenizer_kwargs(kwargs)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            **tokenizer_kwargs,
        )
        return cls(tokenizer=tokenizer, chat_template=chat_template)


def _collect_tokenizer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    tokenizer_kwargs = {}
    for nested_key in ("tokenizer_config", "processor_config", "processor_kwargs"):
        nested = kwargs.pop(nested_key, None)
        if isinstance(nested, dict):
            tokenizer_kwargs.update(nested)

    for ignored_key in ("quantize_activations",):
        kwargs.pop(ignored_key, None)

    tokenizer_kwargs.update(kwargs)
    tokenizer_kwargs.setdefault("fix_mistral_regex", True)
    return tokenizer_kwargs


install_auto_processor_patch("laguna", LagunaProcessor)


__all__ = ["LagunaProcessor"]
