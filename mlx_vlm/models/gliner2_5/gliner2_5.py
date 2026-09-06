import mlx.nn as nn

from .boundary import BoundaryHead
from .config import ModelConfig
from .deberta import DebertaModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        settings = config.boundary_head
        if settings.get("candidate_pool", "shared") != "shared":
            raise ValueError("GLiNER2.5 MLX currently supports shared candidate pools")
        if settings.get("candidate_attention_layers", 0) != 0:
            raise ValueError("candidate attention layers are not supported yet")
        if settings.get("query_attention_layers", 0) != 0:
            raise ValueError("query attention layers are not supported yet")
        self.encoder = DebertaModel(config.encoder_config)
        hidden_size = config.encoder_config.hidden_size
        self.classifier = [
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size * 2, 1),
        ]
        self.boundary_head = BoundaryHead(hidden_size, settings)

    def encode(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)

    def classify(self, choice_states):
        logits = choice_states
        for layer in self.classifier:
            logits = layer(logits)
        return logits.squeeze(-1)

    def extract(self, text_states, text_mask, query_states, query_mask):
        return self.boundary_head(text_states, text_mask, query_states, query_mask)

    def __call__(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)

    def sanitize(self, weights):
        unsupported = (
            "boundary_head.boundary_proposer.",
            "boundary_head.pair_scorer.",
            "boundary_head.candidate_encoder.",
            "record_decoder.",
            "relation_scorer.",
        )
        remapped = {}
        for key, value in weights.items():
            if key.startswith(unsupported):
                continue
            key = key.replace("encoder.encoder.layer.", "encoder.encoder.layers.")
            key = key.replace(".attention.self.", ".attention.self_attn.")
            key = key.replace(".LayerNorm.", ".layer_norm.")
            remapped[key] = value
        return remapped


__all__ = ["Model", "ModelConfig"]
