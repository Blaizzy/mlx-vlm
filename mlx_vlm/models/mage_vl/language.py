"""Mage-VL's decoder is stock Qwen3-4B-Instruct-2507, reused as-is.

The text config is `rope_type: "default"` — plain 1-D rope, not mRoPE — so `models/qwen3`
is the correct base rather than `models/qwen3_vl`. `LanguageModel.__call__` already accepts
`inputs_embeds`, which the vision tower's output is passed through.
"""

from ..qwen3.language import LanguageModel, Qwen3Model, TransformerBlock

__all__ = ["LanguageModel", "Qwen3Model", "TransformerBlock"]
