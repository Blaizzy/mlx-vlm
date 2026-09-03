import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return DeepseekV4Config.from_dict(params)


@dataclass
class DeepseekV4DsparkConfig(BaseModelConfig):
    """DeepSeek-V4 DSpark speculative head.

    A DeepSeek-V4-backbone variant of the model-agnostic DSpark drafter: it
    shares the DSpark proposal machinery (``VanillaMarkov`` block sampling, the
    ``dflash`` round loop and its target-hidden tap) but swaps the Qwen-style
    draft layers for DeepSeek-V4 blocks (MLA attention, MoE, Hyper-Connections).
    Field names mirror ``DSparkConfig`` so the shared ``dflash`` dispatch and
    proposal code apply unchanged.
    """

    model_type: str = "deepseek_v4_dspark"
    text_config: Optional[TextConfig] = None
    n_mtp_layers: int = 1

    # DSpark / dflash proposal contract (mirrors DSparkConfig)
    target_layer_ids: List[int] = field(default_factory=list)
    mask_token_id: int = -1
    markov_rank: int = 256
    block_size: int = 0
    runtime_block_size: Optional[int] = None
    block_size_policy: str = "fixed"
    dflash_initial_block_size: Optional[int] = None
    draft_window_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        self.target_layer_ids = list(self.target_layer_ids or [])
        self._sync_from_text_config()
        if not self.block_size:
            self.block_size = self.proposal_length + 1
        if self.runtime_block_size is None:
            self.runtime_block_size = self.block_size

    def _sync_from_text_config(self):
        text = self.text_config
        if text is None:
            return
        if self.mask_token_id < 0:
            self.mask_token_id = int(getattr(text, "dspark_noise_token_id", 0) or 0)
        if not self.markov_rank:
            self.markov_rank = int(getattr(text, "dspark_markov_rank", 256) or 256)
        if not self.target_layer_ids:
            self.target_layer_ids = list(
                getattr(text, "dspark_target_layer_ids", []) or []
            )
        if not self.block_size:
            block = int(getattr(text, "dspark_block_size", 0) or 0)
            if block:
                self.block_size = block + 1

    @property
    def proposal_length(self) -> int:
        return max(int(self.block_size) - 1, 0)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV4DsparkConfig":
        flat = dict(params)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
