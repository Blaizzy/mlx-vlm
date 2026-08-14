import mlx.core as mx
import mlx.nn as nn

from ..base import SequenceClassifierOutput, scaled_dot_product_attention
from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids, token_type_ids=None):
        position_ids = mx.arange(input_ids.shape[1])[None, :]
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def _shape(self, x, B, L):
        return x.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(self, hidden_states, mask=None):
        B, L, _ = hidden_states.shape
        q = self._shape(self.query(hidden_states), B, L)
        k = self._shape(self.key(hidden_states), B, L)
        v = self._shape(self.value(hidden_states), B, L)
        out = scaled_dot_product_attention(q, k, v, None, self.scale, mask)
        return out.transpose(0, 2, 1, 3).reshape(B, L, self.all_head_size)


class BertSelfOutput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(self, hidden_states, mask=None):
        return self.output(self.self(hidden_states, mask), hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def __call__(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(self, hidden_states, mask=None):
        attention_output = self.attention(hidden_states, mask)
        return self.output(self.intermediate(attention_output), attention_output)


class BertEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, mask)
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def _encode(self, input_ids, attention_mask=None, token_type_ids=None):
        B, L = input_ids.shape
        h = self.embeddings(input_ids, token_type_ids)
        if attention_mask is None:
            attention_mask = mx.ones((B, L))
        mask = (1.0 - attention_mask[:, None, None, :].astype(h.dtype)) * mx.finfo(
            h.dtype
        ).min
        return self.encoder(h, mask.astype(h.dtype)), attention_mask

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        h, attention_mask = self._encode(input_ids, attention_mask, token_type_ids)
        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "mean"
        }
        text_embeds = normalize_embeddings(
            pool_by_config(h, attention_mask, pooling_config)
        )
        return EmbeddingOutput(last_hidden_state=h, text_embeds=text_embeds)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("bert."):
                k = k[len("bert.") :]
            if "position_ids" in k or k.startswith("pooler.") or k.startswith("cls."):
                continue
            out[k] = v
        return out

    @property
    def layers(self):
        return self.encoder.layer


class BertPooler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, hidden_states):
        return mx.tanh(self.dense(hidden_states[:, 0]))


class SequenceClassificationModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        hidden_states, _ = self._encode(input_ids, attention_mask, token_type_ids)
        return SequenceClassifierOutput(
            logits=self.classifier(self.pooler(hidden_states))
        )

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            if key.startswith("bert."):
                key = key[len("bert.") :]
            if "position_ids" in key or key.startswith("cls."):
                continue
            out[key] = value
        return out
