from dataclasses import dataclass, field

from ....models.base import BaseModelConfig
from ....models.mistral_large4.config import TextConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """Native DSpark metadata; block_size includes the target's anchor token."""

    model_type: str = "mistral_large4_dspark"
    text_config: TextConfig = field(default_factory=TextConfig)
    block_size: int = 0
    runtime_block_size: int | None = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        mtp = self.text_config.mtp_config or {}
        if not mtp.get("dspark"):
            raise ValueError("Mistral Large 4 DSpark requires mtp_config.dspark")
        self.num_hidden_layers = int(mtp["n_layers"])
        self.proposal_length = int(mtp["depth"])
        self.target_layer_ids = [
            int(i) for i in mtp["target_activation_layers"].split(",")
        ]
        self.markov_rank = int(mtp["dspark"]["markov_rank"])
        self.head_type = mtp["dspark"].get("head_type", "vanilla")
        self.draft_input = mtp["dspark"].get("draft_input", "shared_mask")
        windows = mtp.get("ragged_attention")
        self.windows = (
            [int(w) for w in windows.split(",")]
            if windows
            else [None] * self.num_hidden_layers
        )
        if (
            self.num_hidden_layers < 1
            or self.proposal_length < 1
            or self.markov_rank < 1
            or len(self.windows) != self.num_hidden_layers
            or any(w is not None and w < 1 for w in self.windows)
        ):
            raise ValueError("Invalid DSpark depth, dimensions or attention windows")
        if (
            not self.target_layer_ids
            or self.target_layer_ids != sorted(set(self.target_layer_ids))
            or any(
                i < 0 or i >= self.text_config.num_hidden_layers
                for i in self.target_layer_ids
            )
        ):
            raise ValueError("Invalid DSpark target activation layers")
        if self.head_type not in ("rnn", "vanilla"):
            raise ValueError(f"Unsupported DSpark head: {self.head_type}")
        if self.draft_input not in ("position_specific", "shared_mask"):
            raise ValueError(f"Unsupported DSpark input: {self.draft_input}")
        if mtp.get("norm_strategy", "eagle") != "eagle":
            raise ValueError("DSpark requires the eagle normalization strategy")
        self.block_size = self.block_size or self.proposal_length + 1
        if self.block_size != self.proposal_length + 1:
            raise ValueError("DSpark block_size must equal trained depth + 1")
        self.runtime_block_size = self.runtime_block_size or self.block_size
        if not 2 <= self.runtime_block_size <= self.block_size:
            raise ValueError("Invalid DSpark runtime block size")

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def vocab_size(self):
        return self.text_config.vocab_size
