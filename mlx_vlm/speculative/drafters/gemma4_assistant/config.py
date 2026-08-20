import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.gemma4.config import TextConfig


@dataclass
class Gemma4AssistantConfig(BaseModelConfig):
    """Drafter config for Gemma 4 Multi-Token Prediction (assistant) models.

    Mirrors the HF ``Gemma4AssistantConfig`` shape (top-level + nested
    ``text_config``). Defaults match ``gg-hf-am/gemma-4-26B-A4B-it-assistant``.
    """

    model_type: str = "gemma4_assistant"
    backbone_hidden_size: int = 1536
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True
    block_size: int = 4
    # Unused by MTP (drafter consumes shared K/V, not per-layer hidden
    # captures) but kept for API parity with DFlash so the round-loop's
    # references to ``draft_model.config.target_layer_ids`` don't crash.
    target_layer_ids: List[int] = field(default_factory=list)
    text_config: Optional[TextConfig] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is not None:
            # HF Gemma4AssistantConfig.__post_init__: when num_kv_shared_layers
            # is unset/0 the assistant shares K/V across all layers.
            if not self.text_config.num_kv_shared_layers:
                self.text_config.num_kv_shared_layers = (
                    self.text_config.num_hidden_layers
                )

    @classmethod
    def from_dict(cls, params: dict) -> "Gemma4AssistantConfig":
        flat = dict(params)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
