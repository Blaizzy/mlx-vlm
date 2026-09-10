import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.deepseek_v41.config import ModelConfig as DeepseekV41Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return DeepseekV41Config.from_dict(params)


@dataclass
class DeepseekV41DsparkConfig(BaseModelConfig):
    """DeepSeek-V4.1 DSpark speculative head.

    DSpark stacks ``n_mtp_layers`` transformer stages (3 in V4.1) that draft a
    whole block of ``dspark_block_size`` tokens per step, conditioned on the
    hidden states of ``dspark_target_layer_ids`` from the target model. Stage 0
    owns ``main_proj``/``main_norm``; only the last stage owns the final norm
    plus ``markov_head`` and ``confidence_head``.
    """

    model_type: str = "deepseek_v41_dspark"
    text_config: Optional[TextConfig] = None
    n_mtp_layers: int = 3
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    block_size: int = 0
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        self.dspark_target_layer_ids = list(self.dspark_target_layer_ids or [])
        self._sync_dspark_from_text_config()
        # The drafter proposes ``dspark_block_size`` tokens plus the bonus.
        if not self.block_size:
            self.block_size = int(self.dspark_block_size) + 1
        if self.runtime_block_size is None:
            self.runtime_block_size = self.block_size

    def _sync_dspark_from_text_config(self):
        """Fall back to the DSpark knobs carried on the text config."""
        text = self.text_config
        if text is None:
            return
        for name in (
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_markov_rank",
        ):
            if not getattr(self, name):
                value = getattr(text, name, 0)
                if value:
                    setattr(self, name, int(value))
        if not self.dspark_target_layer_ids:
            self.dspark_target_layer_ids = list(
                getattr(text, "dspark_target_layer_ids", []) or []
            )

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV41DsparkConfig":
        flat = dict(params)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
