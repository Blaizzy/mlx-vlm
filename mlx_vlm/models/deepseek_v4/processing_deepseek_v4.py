from pathlib import Path
from typing import Optional

from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch

DEFAULT_CHAT_TEMPLATE = (
    "{{- '<｜begin▁of▁sentence｜>' -}}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{{- messages[0]['content'] -}}"
    "{%- set start = 1 -%}"
    "{%- else -%}"
    "{%- set start = 0 -%}"
    "{%- endif -%}"
    "{%- for m in messages[start:] -%}"
    "{%- if m['role'] == 'user' -%}"
    "{{- '<｜User｜>' + m['content'] -}}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{{- '<｜Assistant｜>' + m['content'] + '<｜end▁of▁sentence｜>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<｜Assistant｜></think>' -}}"
    "{%- endif -%}"
)


def load_deepseek_v4_chat_template(model_path, **kwargs) -> Optional[str]:
    local_path = Path(model_path)
    if local_path.exists():
        template_path = local_path / "chat_template.jinja"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return None

    try:
        from huggingface_hub import hf_hub_download

        download_kwargs = {
            key: kwargs[key]
            for key in ("revision", "token", "local_files_only")
            if key in kwargs
        }
        template_path = hf_hub_download(
            repo_id=str(model_path),
            filename="chat_template.jinja",
            **download_kwargs,
        )
        return Path(template_path).read_text(encoding="utf-8")
    except Exception:
        return None


class DeepseekV4Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        chat_template = (
            chat_template
            or getattr(tokenizer, "chat_template", None)
            or DEFAULT_CHAT_TEMPLATE
        )
        self.tokenizer.chat_template = chat_template
        super().__init__(tokenizer, chat_template=chat_template, **kwargs)

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
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        chat_template = kwargs.pop("chat_template", None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        except (AttributeError, ValueError):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        if chat_template is None:
            chat_template = load_deepseek_v4_chat_template(
                pretrained_model_name_or_path,
                **kwargs,
            )
        return cls(tokenizer=tokenizer, chat_template=chat_template)


install_auto_processor_patch("deepseek_v4", DeepseekV4Processor)

__all__ = [
    "DEFAULT_CHAT_TEMPLATE",
    "DeepseekV4Processor",
    "load_deepseek_v4_chat_template",
]
