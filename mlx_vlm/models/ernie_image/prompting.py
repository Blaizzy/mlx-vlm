from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_vlm.models.cache import KVCache
from mlx_vlm.sample_utils import make_sampler

from .text_encoder import ErnieImageTextEncoder


class ErnieImagePromptEnhancer:
    def __init__(
        self,
        *,
        model: ErnieImageTextEncoder,
        model_path: str | Path,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).expanduser() / "pe_tokenizer"),
            local_files_only=True,
            use_fast=True,
        )
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", 2048))
        self.max_new_tokens = max_new_tokens or min(tokenizer_limit, 2048)

    def enhance(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            formatted,
            add_special_tokens=False,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        caches = [KVCache() for _ in self.model.layers]
        sampler = make_sampler(temp=temperature, top_p=top_p)
        mx.random.seed(seed)
        generated = []
        additional_eos = getattr(
            self.tokenizer, "additional_special_tokens_ids", None
        ) or ()
        eos_ids = {
            int(token)
            for token in (
                getattr(self.tokenizer, "eos_token_id", None),
                *additional_eos,
            )
            if token is not None
        }
        for _ in range(self.max_new_tokens):
            hidden_states = self.model(input_ids, cache=caches, normalize=True)
            logits = self.model.embed_tokens.as_linear(hidden_states[:, -1, :])
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token = sampler(logprobs)
            mx.eval(token)
            token_id = int(token.item())
            if token_id in eos_ids:
                break
            generated.append(token_id)
            input_ids = token.reshape(1, 1)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


__all__ = ["ErnieImagePromptEnhancer"]
