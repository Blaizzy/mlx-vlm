from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseModelConfig


@dataclass
class AudioModelOutput:
    last_hidden_state: mx.array


class AudioModel(nn.Module):
    """
    Placeholder implementation of the Qwen3-Omni audio encoder.

    The Hugging Face reference implementation exposes a fairly involved audio
    tower (convolutional down-sampling + Transformer encoder).  Re-implementing
    it in MLX requires a one-to-one port of ~1k lines of PyTorch code, which is
    beyond the scope of this initial plumbing change.  For now we keep a very
    small shim that simply forwards pre-computed audio embeddings (already in
    the language model hidden size) while retaining the same call-site API.

    The class is intentionally shaped like the final encoder so we can swap in
    the full port once it is ready without touching the higher level model.
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: Optional[mx.array] = None,
    ) -> AudioModelOutput:
        raise NotImplementedError(
            "Audio inputs are not supported in the initial Qwen3-Omni MLX port. "
            "The Thinker stack, tokenizer and vision tower are wired up, but the "
            "audio encoder still needs to be translated from PyTorch.  "
            "Please keep `input_features=None` for now."
        )
