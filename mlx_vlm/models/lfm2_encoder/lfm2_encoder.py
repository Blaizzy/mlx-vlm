from typing import Optional

import mlx.core as mx

from ..base import LanguageModelOutput
from ..lfm2_embedding.lfm2_embedding import Model as Lfm2BidirectionalModel


class Model(Lfm2BidirectionalModel):
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        h, _ = self._encode(input_ids, attention_mask)
        return LanguageModelOutput(logits=self.model.embed_tokens.as_linear(h))

    def sanitize(self, weights):
        weights = {key.removeprefix("lfm2."): value for key, value in weights.items()}
        return super().sanitize(weights)
