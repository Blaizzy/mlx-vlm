from dataclasses import dataclass

from ....models.deepseek_v41.config import ModelConfig as TextConfig
from ..deepseek_v4_dspark.config import DeepseekV4DsparkConfig


@dataclass
class ModelConfig(DeepseekV4DsparkConfig):
    model_type: str = "deepseek_v41_dspark"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        super().__post_init__()
