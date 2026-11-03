"""Use mlx-vlm's Qwen3-VL text backbone; the vision tower is unnecessary."""

import json
import math
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from ..qwen3_vl.config import TextConfig
from ..qwen3_vl.language import Qwen3VLModel


class Flux3TextEncoder(Qwen3VLModel):
    def get_prompt_embeds(self, input_ids, attention_mask, layers):
        h = self.embed_tokens(input_ids)
        positions = mx.cumsum(attention_mask, axis=-1) - 1
        positions = mx.where(attention_mask == 0, 1, positions)
        positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        length = input_ids.shape[1]
        mask = (mx.arange(length)[:, None] >= mx.arange(length)[None, :])[None, None]
        mask = mask & attention_mask[:, None, None, :].astype(mx.bool_)
        features = []
        for index, layer in enumerate(self.layers, start=1):
            h = layer(h, mask=mask, position_ids=positions)
            if index in layers:
                features.append(
                    self.norm(h) if index == self.args.num_hidden_layers else h
                )
            if index == max(layers):
                break
        return mx.concatenate(features, axis=-1).astype(mx.bfloat16)


def text_config(path: Path) -> TextConfig:
    config = json.loads((path / "config.json").read_text())["text_config"]
    return TextConfig.from_dict(config)


class Flux3Tokenizer:
    def __init__(self, path, *, max_length=4096, pad_multiple=80):
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        template = path / "chat_template.json"
        if not self.tokenizer.chat_template and template.exists():
            self.tokenizer.chat_template = json.loads(template.read_text())[
                "chat_template"
            ]
        self.tokenizer.padding_side = "right"
        self.max_length = max_length
        self.pad_multiple = pad_multiple

    def format(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self, prompt):
        return len(self.tokenizer.encode(self.format(prompt)))

    def tokenize(self, prompt, max_length=None):
        cap = min(max_length or self.max_length, self.max_length)
        real = min(self.count_tokens(prompt), cap)
        length = min(math.ceil(real / self.pad_multiple) * self.pad_multiple, cap)
        tokens = self.tokenizer(
            self.format(prompt),
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"]), mx.array(tokens["attention_mask"])
