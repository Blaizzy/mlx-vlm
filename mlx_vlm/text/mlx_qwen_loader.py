from __future__ import annotations

from dataclasses import dataclass


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

        self.model, self.tokenizer = load(opts.model_dir)
        self.opts = opts
