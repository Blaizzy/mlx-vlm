import mlx.nn as nn

from ..mistral3.mistral3 import Mistral3MultiModalProjector
from ..mistral3.mistral3 import Model as Mistral3Model
from ..pixtral import VisionModel
from .config import ModelConfig
from .language import LanguageModel


class Model(Mistral3Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.vision_feature_layer = config.vision_feature_layer

    def shard(self, group=None):
        self.language_model.shard(group)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, inputs_embeds=features.inputs_embeds, mask=mask, cache=cache
        )

    @property
    def quant_predicate(self):
        # Native expert matrices are already quantized and have no to_quantized.
        return lambda path, module: hasattr(module, "to_quantized")
