from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx


@dataclass
class QwenLoadOpts:
    model_dir: str
    max_new_tokens: int = 128
    temperature: float = 0.0


class MLXQwen:
    """Minimal wrapper that loads Qwen with mlx-lm."""

    def __init__(self, opts: QwenLoadOpts):
        try:
            from mlx_lm import load
        except Exception as exc:
            raise RuntimeError("mlx-lm not installed; run `pip install mlx-lm`") from exc

        import os
        os.environ.setdefault("TRANSFORMERS_TRUST_REMOTE_CODE", "1")
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

        from transformers import AutoTokenizer

        original_from_pretrained = AutoTokenizer.from_pretrained

        def _from_pretrained_with_trust(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            return original_from_pretrained(*args, **kwargs)

        AutoTokenizer.from_pretrained = _from_pretrained_with_trust

        try:
            self.model, self.tokenizer = load(opts.model_dir)
        finally:
            AutoTokenizer.from_pretrained = original_from_pretrained
        self.opts = opts

    def embed_weight(self):
        model_sub = getattr(self.model, "model", None)
        if model_sub is not None:
            embed_tokens = getattr(model_sub, "embed_tokens", None)
            if embed_tokens is not None:
                return embed_tokens.weight

        transformer = getattr(self.model, "transformer", None)
        if transformer is not None:
            wte = getattr(transformer, "wte", None)
            if wte is not None:
                return wte.weight

        raise AttributeError("Unable to locate embedding weight on loaded model")

    def generate_with_prefix_embeddings(self, fused_emb: mx.array, max_new_tokens: int):
        mx.eval(fused_emb)
        return "[TODO-MLX-Qwen: embedding-level generate not wired yet in this stub]"
