from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mlx.core as mx

VISION_TOKEN_CANDIDATES = [
    "<|vision_start|>",
    "<|vision_end|>",
    "<image>",
    "<img>",
    "<image_patch>",
    "<image_placeholder>",
]


def pick_vision_token_id(tokenizer) -> tuple[str | None, int | None]:
    """Return the first known vision token string and its id, if present."""

    vocab = tokenizer.get_vocab()
    for tok in VISION_TOKEN_CANDIDATES:
        tid = vocab.get(tok)
        if tid is not None:
            return tok, tid
    return None, None


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
            from mlx_lm.sample_utils import make_sampler
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
        self._make_sampler = make_sampler
        self.opts = opts
        self._embed_backup: Optional[np.ndarray] = None
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

    def set_image_embedding_row(
        self,
        image_id: int,
        new_vec_np: np.ndarray,
        blend: float = 0.65,
    ):
        """Blend the stored `<image>` embedding row with a new vector.

        The update follows ``W[image_id] = (1 - blend) * W_orig + blend * new_vec``
        so that the model conditions on the image features without making the
        placeholder token itself overwhelmingly likely.
        """

        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be within [0, 1]; received {blend}")

        W = self.embed_weight()
        target_dtype = getattr(W, "dtype", None)
        W_np = np.array(mx.array(W, dtype=mx.float32))

        if self._embed_backup is None:
            self._embed_backup = W_np[image_id].copy()
            self._backup_index = image_id

        target_shape = tuple(W_np[image_id].shape)
        if new_vec_np.shape != target_shape:
            raise ValueError(
                f"image embedding shape mismatch: {tuple(new_vec_np.shape)} vs {target_shape}"
            )

        orig = self._embed_backup if self._embed_backup is not None else W_np[image_id]
        new_row = new_vec_np.astype(np.float32, copy=False)
        mixed = (1.0 - blend) * orig + blend * new_row

        W_np[image_id, :] = mixed
        updated = mx.array(W_np, dtype=target_dtype) if target_dtype is not None else mx.array(W_np)
        self._set_embed_weight(updated)

    def restore_image_embedding_row(self):
        if self._embed_backup is None or self._backup_index is None:
            return

        W = self.embed_weight()
        target_dtype = getattr(W, "dtype", None)
        W_np = np.array(mx.array(W, dtype=mx.float32))

        W_np[self._backup_index, :] = self._embed_backup.astype(np.float32, copy=False)
        restored = mx.array(W_np, dtype=target_dtype) if target_dtype is not None else mx.array(W_np)
        self._set_embed_weight(restored)
        self._embed_backup = None
        self._backup_index = None

    @contextlib.contextmanager
    def temp_image_embedding(
        self,
        image_id: int,
        vec_np: np.ndarray,
        blend: float = 0.65,
    ):
        try:
            self.set_image_embedding_row(image_id, vec_np, blend=blend)
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
        sampler = self._make_sampler(temp=float(temp))
        return self._generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_new,
            sampler=sampler,
        )

    def generate_with_image_embedding(
        self,
        prompt: str,
        image_id: int,
        vision_projected_seq: np.ndarray,
        reduce: str = "mean",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        blend: float = 0.65,
    ) -> str:
        if vision_projected_seq.ndim != 2:
            raise ValueError("vision_projected_seq must be [Tv, H]")

        if reduce == "mean":
            vec = vision_projected_seq.mean(axis=0)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")

        embed_dim = int(self.embed_weight().shape[1])
        if vec.shape[0] != embed_dim:
            raise ValueError(
                "Projected vision hidden size does not match text embedding dimension: "
                f"{vec.shape[0]} vs {embed_dim}. Load a projector (see convert_dots_ocr.py"
                " convert-projector) or supply a text checkpoint whose hidden size matches"
                " the vision projector output."
            )

        with self.temp_image_embedding(image_id, vec, blend=blend):
            return self.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def generate_with_prefix_embeddings(self, fused_emb: mx.array, max_new_tokens: int):
        mx.eval(fused_emb)
        return "[TODO-MLX-Qwen: embedding-level generate not wired yet in this stub]"
