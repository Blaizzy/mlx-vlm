import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    BatchFeature,
    ProcessorMixin,
    SiglipImageProcessor,
)

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from ..internvl_chat.processor import dynamic_preprocess

HISTORY_CHAT_TEMPLATE = """
{%- if not messages or messages[0]['role'] != 'system' -%}
{{- '<|start|>system<|message|>You are LLM-jp-VL, a Multimodal LLM trained by LLM-jp.<|end|>' -}}
{%- endif -%}
{%- if tools -%}
{{- '<|start|>system<|message|>Available tools: ' + (tools | tojson) + '<|end|>' -}}
{%- endif -%}
{%- for message in messages -%}
{{- '<|start|>' + message['role'] -}}
{%- if message['role'] == 'assistant' -%}{{- '<|channel|>final' -}}{%- endif -%}
{{- '<|message|>' -}}
{{- message['content'] if message['content'] is string else message['content'] | tojson -}}
{%- for key, value in message.items() if key not in ['role', 'content'] -%}
{{- '\\nMessage metadata: ' + ({key: value} | tojson) -}}
{%- endfor -%}
{{- '<|return|>' if loop.last and message['role'] == 'assistant' and not add_generation_prompt else '<|end|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|start|>assistant<|channel|>final<|message|>' -}}{%- endif -%}
"""


class LLMjpVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_length=256,
        max_dynamic_patch=12,
        min_dynamic_patch=1,
        use_thumbnail=True,
    ):
        self.image_seq_length = image_seq_length
        self.max_dynamic_patch = max_dynamic_patch
        self.min_dynamic_patch = min_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.image_token = "<image>"
        template = tokenizer.chat_template
        if template:
            template += (
                "{% if add_generation_prompt %}<|channel|>final<|message|>{% endif %}"
            )
        super().__init__(image_processor, tokenizer, chat_template=template)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        histories = (
            conversation
            if conversation and isinstance(conversation[0], (list, tuple))
            else [conversation]
        )
        rich_history = bool(kwargs.get("tools"))
        for messages in histories:
            system_positions = [
                i for i, message in enumerate(messages) if message["role"] == "system"
            ]
            rich_history |= system_positions not in ([], [0]) or any(
                set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                for message in messages
            )
        if chat_template is None and rich_history:
            chat_template = HISTORY_CHAT_TEMPLATE
        return super().apply_chat_template(
            conversation, chat_template=chat_template, **kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        load_chat_template(tokenizer, path)
        image_config = json.loads((path / "preprocessor_config.json").read_text())
        image_processor = SiglipImageProcessor.from_dict(image_config)
        config = json.loads((path / "processor_config.json").read_text())
        options = {
            key: config[key]
            for key in (
                "image_seq_length",
                "max_dynamic_patch",
                "min_dynamic_patch",
                "use_thumbnail",
            )
            if key in config
        }
        return cls(image_processor, tokenizer, **options)

    def __call__(
        self,
        images=None,
        text=None,
        padding=False,
        padding_side="left",
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else list(text or [])
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        if not texts:
            raise ValueError("LLM-jp-VL requires text containing image placeholders")
        if sum(t.count(self.image_token) for t in texts) != len(images):
            raise ValueError("Image count does not match image placeholders")
        pixels = []
        image_index = 0
        expanded = []
        size = self.image_processor.size
        image_size = size["height"] if isinstance(size, dict) else size.height
        for prompt in texts:
            count = prompt.count(self.image_token)
            text_tokens = len(
                self.tokenizer.encode(
                    prompt.replace(self.image_token, ""), add_special_tokens=False
                )
            )
            budget = self.tokenizer.model_max_length - text_tokens
            max_num = max(
                1,
                min(
                    self.max_dynamic_patch,
                    (budget // max(count, 1) - 2) // self.image_seq_length - 1,
                ),
            )
            for _ in range(count):
                patches = dynamic_preprocess(
                    images[image_index].convert("RGB"),
                    min_num=self.min_dynamic_patch,
                    max_num=max_num,
                    image_size=image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                image_index += 1
                pixels.append(
                    self.image_processor(images=patches, return_tensors="np")[
                        "pixel_values"
                    ]
                )
                replacement = (
                    "<|image_start|>"
                    + "<|image_pad|>" * (self.image_seq_length * len(patches))
                    + "<|image_end|>"
                )
                prompt = prompt.replace(self.image_token, replacement, 1)
            expanded.append(prompt)
        data = dict(
            self.tokenizer(
                expanded,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )
        )
        if pixels:
            data["pixel_values"] = np.concatenate(pixels)
        return BatchFeature(data=to_mlx(data))

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


install_auto_processor_patch("llmjpvl", LLMjpVLProcessor)
