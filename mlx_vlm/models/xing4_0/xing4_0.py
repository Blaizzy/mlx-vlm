from ..deepseek_v3.deepseek_v3 import Model as DeepseekV3Model
from .language import LanguageModel


class Model(DeepseekV3Model):
    def __init__(self, config):
        super(DeepseekV3Model, self).__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
