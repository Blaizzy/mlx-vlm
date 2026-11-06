import mlx.core as mx

from ..jev.jev import Jev, merge_lora
from ..qwen3_5_text import Model as Qwen3_5TextModel


class Model(Qwen3_5TextModel):
    decision_types = ("choice", "score", "bool", "noul")

    def __call__(self, input_ids, **media):
        length = input_ids.shape[1]
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
        hidden = self.language_model.model(input_ids, position_ids=positions)
        return self.language_model.lm_head(hidden[:, -1])

    def sanitize(self, weights):
        return super().sanitize(merge_lora(weights, self.config.decision_config))

    @property
    def quant_predicate(self):
        base = super().quant_predicate
        return lambda path, module: path != "language_model.lm_head" and (
            base is None or base(path, module)
        )

    def predict(self, processor, state, questions, **kwargs):
        return JevText(self, processor).predict(state, questions, **kwargs)


class JevText(Jev):
    image_token = None
