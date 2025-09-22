from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

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
            from mlx_lm import generate, load
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

        self._generate = generate
        self.opts = opts
        self._embed_backup: Optional[mx.array] = None
        self._backup_index: Optional[int] = None

    def _set_embed_weight(self, new_weight: mx.array) -> None:
        model_sub = getattr(self.model, "model", None)
        if model_sub is not None:
            embed_tokens = getattr(model_sub, "embed_tokens", None)
            weight = getattr(embed_tokens, "weight", None) if embed_tokens is not None else None
            if weight is not None:
                embed_tokens.weight = new_weight
                return

        transformer = getattr(self.model, "transformer", None)
        if transformer is not None:
            wte = getattr(transformer, "wte", None)
            weight = getattr(wte, "weight", None) if wte is not None else None
            if weight is not None:
                wte.weight = new_weight
                return

        raise AttributeError("Unable to locate embedding weight on loaded model")

    def embed_weight(self) -> mx.array:
        model_sub = getattr(self.model, "model", None)
        if model_sub is not None:
            embed_tokens = getattr(model_sub, "embed_tokens", None)
            if embed_tokens is not None:
                weight = getattr(embed_tokens, "weight", None)
                if weight is not None:
                    return weight

        transformer = getattr(self.model, "transformer", None)
        if transformer is not None:
            wte = getattr(transformer, "wte", None)
            if wte is not None:
                weight = getattr(wte, "weight", None)
                if weight is not None:
                    return weight

        raise AttributeError("Unable to locate embedding weight on loaded model")

    def image_token_id(self) -> int | None:
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<image>", None)

    def set_image_embedding_row(self, image_id: int, new_vec_np: np.ndarray):
        W = self.embed_weight()
        if self._embed_backup is None:
            self._embed_backup = W[image_id].copy()
            self._backup_index = image_id

        new_row = mx.array(new_vec_np.astype(np.float32))
        if new_row.shape != W[image_id].shape:
            raise ValueError(
                f"image embedding shape mismatch: {tuple(new_row.shape)} vs {tuple(W[image_id].shape)}"
            )

        W_np = np.array(W)
        W_np[image_id, :] = np.array(new_row)
        self._set_embed_weight(mx.array(W_np))

    def restore_image_embedding_row(self):
        if self._embed_backup is None or self._backup_index is None:
            return

        W = self.embed_weight()
        W_np = np.array(W)
        W_np[self._backup_index, :] = np.array(self._embed_backup)
        self._set_embed_weight(mx.array(W_np))
        self._embed_backup = None
        self._backup_index = None

    @contextlib.contextmanager
    def temp_image_embedding(self, image_id: int, vec_np: np.ndarray):
        try:
            self.set_image_embedding_row(image_id, vec_np)
            yield
        finally:
            self.restore_image_embedding_row()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        max_new = self.opts.max_new_tokens if max_new_tokens is None else max_new_tokens
        temp = self.opts.temperature if temperature is None else temperature
        return self._generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_new,
            temp=temp,
        )

    def generate_with_image_embedding(
        self,
        prompt: str,
        image_id: int,
        vision_projected_seq: np.ndarray,
        reduce: str = "mean",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if vision_projected_seq.ndim != 2:
            raise ValueError("vision_projected_seq must be [Tv, H]")

        if reduce == "mean":
            vec = vision_projected_seq.mean(axis=0)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")

        with self.temp_image_embedding(image_id, vec):
            return self.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def generate_with_prefix_embeddings(self, fused_emb: mx.array, max_new_tokens: int):
        mx.eval(fused_emb)
        return "[TODO-MLX-Qwen: embedding-level generate not wired yet in this stub]"
