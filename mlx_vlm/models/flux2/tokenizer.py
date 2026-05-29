from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer


@dataclass(frozen=True, slots=True)
class TokenizerOutput:
    input_ids: mx.array
    attention_mask: mx.array


class Flux2Tokenizer:
    def __init__(self, model_path: str | Path, max_length: int = 512) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).expanduser() / "tokenizer"),
            local_files_only=True,
            use_fast=True,
        )

    def _format_prompts(self, prompt: str | list[str]) -> list[str]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [p if p is not None else "" for p in prompts]
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]

    def count_tokens(self, prompt: str | list[str]) -> int:
        prompts = self._format_prompts(prompt)
        if all(p == "" for p in prompts):
            return 0
        tokens = self.tokenizer(
            prompts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        return max(len(ids) for ids in tokens["input_ids"])

    def tokenize(
        self, prompt: str | list[str], max_length: int | None = None
    ) -> TokenizerOutput:
        prompts = self._format_prompts(prompt)
        if all(p == "" for p in prompts):
            batch = len(prompts)
            return TokenizerOutput(
                input_ids=mx.array(np.empty((batch, 0), dtype=np.int32)),
                attention_mask=mx.array(np.empty((batch, 0), dtype=np.int32)),
            )
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length or self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        return TokenizerOutput(
            input_ids=mx.array(tokens["input_ids"]),
            attention_mask=mx.array(tokens["attention_mask"]),
        )


__all__ = ["Flux2Tokenizer", "TokenizerOutput"]
