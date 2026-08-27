import mlx.core as mx
import mlx.nn as nn

from ..base import SequenceClassifierOutput, scaled_dot_product_attention
from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias
        )
        self.act = nn.GELU(approx="precise")
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, hidden_states):
        x = self.Wi(hidden_states)
        split = x.shape[-1] // 2
        inp, gate = x[:, :, :split], x[:, :, split:]
        return self.Wo(self.act(inp) * gate)


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5
        self.local = layer_id % config.global_attn_every_n_layers != 0
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias
        )
        rope_theta = config.global_rope_theta
        if self.local and config.local_rope_theta is not None:
            rope_theta = config.local_rope_theta
        self.rotary_emb = nn.RoPE(dims=self.head_dim, base=rope_theta)
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, hidden_states, global_mask=None, sliding_mask=None):
        bs = hidden_states.shape[0]
        qkv = self.Wqkv(hidden_states)
        qkv = mx.reshape(qkv, (bs, -1, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, [0, 3, 2, 1, 4])
        q, k, v = mx.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        mask = sliding_mask if self.local else global_mask
        out = scaled_dot_product_attention(q, k, v, None, self.scale, mask)
        out = mx.transpose(out, [0, 2, 1, 3])
        out = mx.reshape(out, (bs, -1, self.all_head_size))
        return self.Wo(out)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = ModernBertAttention(config, layer_id)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def __call__(self, hidden_states, global_mask=None, sliding_mask=None):
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), global_mask, sliding_mask
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.local_attention = config.local_attention
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def _masks(self, attention_mask, dtype):
        B, L = attention_mask.shape
        additive = mx.where(attention_mask == 1, 0.0, -1e9)[:, None, None, :]
        global_mask = mx.broadcast_to(additive, (B, 1, L, L))
        rows = mx.arange(L)[None, :]
        distance = mx.abs(rows - rows.T)
        window = distance <= (self.local_attention // 2)
        window = mx.broadcast_to(window[None, None, :, :], global_mask.shape)
        sliding_mask = mx.where(window, global_mask, -1e9)
        return global_mask.astype(dtype), sliding_mask.astype(dtype)

    def _encode(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((B, L))
        h = self.embeddings(input_ids)
        global_mask, sliding_mask = self._masks(attention_mask, h.dtype)
        for layer in self.layers:
            h = layer(h, global_mask, sliding_mask)
        return self.final_norm(h), attention_mask

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        h, attention_mask = self._encode(input_ids, attention_mask)
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
            if k.startswith("model."):
                k = k[len("model.") :]
            if (
                k.startswith("head.")
                or k.startswith("decoder.")
                or k.startswith("classifier.")
                or k.startswith("pooler.")
            ):
                continue
            out[k] = v
        return out


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.classifier_activation != "gelu":
            raise ValueError(
                f"Unsupported ModernBERT classifier activation: "
                f"{config.classifier_activation!r}."
            )
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.classifier_bias
        )
        self.act = nn.GELU(approx="precise")
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, hidden_states):
        return self.norm(self.act(self.dense(hidden_states)))


class SequenceClassificationModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.head = ModernBertPredictionHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        hidden_states, attention_mask = self._encode(input_ids, attention_mask)
        if self.config.classifier_pooling == "cls":
            pooled = hidden_states[:, 0]
        elif self.config.classifier_pooling == "mean":
            mask = attention_mask[..., None].astype(hidden_states.dtype)
            pooled = mx.sum(hidden_states * mask, axis=1) / mx.maximum(
                mx.sum(mask, axis=1), 1
            )
        else:
            raise ValueError(
                f"Unsupported ModernBERT classifier pooling: "
                f"{self.config.classifier_pooling!r}."
            )
        return SequenceClassifierOutput(logits=self.classifier(self.head(pooled)))

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("decoder.") or key.startswith("pooler."):
                continue
            out[key] = value
        return out
