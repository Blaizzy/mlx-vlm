from types import SimpleNamespace

from ....models.deepseek_v41.config import ModelConfig
from ..deepseek_v4_dspark.split import DeepseekV4DsparkSplitter
from .deepseek_v41_dspark import DeepseekV41DsparkDraftModel


class DeepseekV41DsparkSplitter(DeepseekV4DsparkSplitter):
    output_model_type = "deepseek_v41_dspark"
    draft_model_cls = DeepseekV41DsparkDraftModel

    def sanitize_ctx(self, text_config):
        return SimpleNamespace(args=ModelConfig.from_dict(text_config))
