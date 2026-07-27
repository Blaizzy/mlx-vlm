"""Mage-VL's decoder is stock Qwen3-4B-Instruct-2507 — reused verbatim, not re-translated.

Path A, settled by reading both candidates rather than picking by name:

* `models/qwen3` builds rope through `initialize_rope(base=rope_theta, scaling_config=rope_scaling)`
  and advances positions by `cache.offset` — plain 1-D rope. That is exactly Mage-VL, whose text
  config is `rope_type: "default"` and whose modeling code says, in as many words, "Use simple 1D
  position_ids" (`modeling_mage_vl.py:1269`).
* `models/qwen3_vl` is the closer-sounding name and the WRONG model here: its language half is
  `MRoPERotaryEmbedding`-based and tiles `position_ids` to `(3, 1, 1)`. Mage-VL reserves 3D mRoPE
  as future work and does not use it.

`LanguageModel.__call__` already accepts `inputs_embeds`, which is the whole reason no decoder
work is needed — the vision tower's output goes straight in.
"""

from ..qwen3.language import LanguageModel, Qwen3Model, TransformerBlock

__all__ = ["LanguageModel", "Qwen3Model", "TransformerBlock"]
