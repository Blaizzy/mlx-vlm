from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..cache import KVCache
from ..pooling import EmbeddingOutput
from ..qwen2_5_vl.qwen2_5_vl import Model as Qwen2_5_VLModel
from .config import ModelConfig


class Model(Qwen2_5_VLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.custom_text_proj = nn.Linear(
            config.text_config.hidden_size, config.embedding_dim
        )
        self.language_model.lm_head = None

    def _last_hidden_state(self, inputs_embeds: mx.array) -> mx.array:
        model = self.language_model.model
        cache = [KVCache() for _ in model.layers]
        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = mx.broadcast_to(
            mx.arange(seq_length).reshape(1, 1, seq_length),
            (3, batch_size, seq_length),
        )
        return model(
            None,
            inputs_embeds=inputs_embeds,
            mask=None,
            cache=cache,
            position_ids=position_ids,
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        last_hidden_state = self._last_hidden_state(features.inputs_embeds)
        proj = self.custom_text_proj(last_hidden_state)
        proj = proj / mx.linalg.norm(proj, axis=-1, keepdims=True)
        if attention_mask is not None:
            proj = proj * attention_mask[:, :, None]
        return EmbeddingOutput(last_hidden_state=last_hidden_state, text_embeds=proj)

    def sanitize(self, weights):
        remapped = {}
        for k, v in weights.items():
            if k.startswith("embedding_proj_layer"):
                k = k.replace("embedding_proj_layer", "custom_text_proj")
            elif k.startswith("vlm.model.language_model."):
                k = k.replace("vlm.model.language_model.", "model.")
            elif k.startswith("vlm.model.visual."):
                k = k.replace("vlm.model.visual.", "visual.")
            elif k.startswith("vlm.model."):
                k = k.replace("vlm.model.", "model.")
            elif k.startswith("vlm."):
                k = k[4:]
            remapped[k] = v
        return super().sanitize(remapped)
