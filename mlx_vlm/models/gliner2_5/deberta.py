import math

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig


def _relative_positions(length: int, bucket_size: int, max_position: int):
    positions = mx.arange(length)
    relative = positions[:, None] - positions[None, :]
    if bucket_size <= 0 or max_position <= 0:
        return relative.astype(mx.int32)
    middle = bucket_size // 2
    absolute = mx.where(
        (relative < middle) & (relative > -middle),
        middle - 1,
        mx.abs(relative),
    )
    absolute_f = absolute.astype(mx.float32)
    log_position = (
        mx.ceil(
            mx.log(absolute_f / middle)
            / math.log((max_position - 1) / middle)
            * (middle - 1)
        )
        + middle
    )
    bucketed = mx.where(
        absolute <= middle,
        relative.astype(mx.float32),
        log_position * mx.sign(relative),
    )
    return bucketed.astype(mx.int32)


class DebertaEmbeddings(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids, attention_mask):
        states = self.layer_norm(self.word_embeddings(input_ids))
        return states * attention_mask[..., None].astype(states.dtype)


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.position_buckets = config.position_buckets
        self.max_relative_positions = config.max_relative_positions
        self.pos_att_type = config.pos_att_type
        self.share_att_key = config.share_att_key

    def _heads(self, states):
        batch, length, _ = states.shape
        return states.reshape(
            batch, length, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

    def __call__(self, states, attention_mask, relative_embeddings):
        query = self._heads(self.query_proj(states))
        key = self._heads(self.key_proj(states))
        value = self._heads(self.value_proj(states))
        scale_factor = (
            1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
        )
        scale = math.sqrt(self.head_dim * scale_factor)
        scores = (query @ key.swapaxes(-1, -2)) / scale

        length = states.shape[1]
        span = self.position_buckets
        relative = _relative_positions(
            length, self.position_buckets, self.max_relative_positions
        )
        relative_embeddings = relative_embeddings[: 2 * span][None]
        if not self.share_att_key:
            raise ValueError("GLiNER2.5 requires share_att_key=True")
        position_query = self._heads(self.query_proj(relative_embeddings))
        position_key = self._heads(self.key_proj(relative_embeddings))

        batch = states.shape[0]
        if "c2p" in self.pos_att_type:
            c2p = query @ position_key.swapaxes(-1, -2)
            indices = mx.clip(relative + span, 0, 2 * span - 1)[None, None]
            indices = mx.broadcast_to(
                indices, (batch, self.num_attention_heads, length, length)
            )
            scores = scores + mx.take_along_axis(c2p, indices, axis=-1) / scale
        if "p2c" in self.pos_att_type:
            p2c = key @ position_query.swapaxes(-1, -2)
            indices = mx.clip(-relative + span, 0, 2 * span - 1)[None, None]
            indices = mx.broadcast_to(
                indices, (batch, self.num_attention_heads, length, length)
            )
            scores = (
                scores
                + mx.take_along_axis(p2c, indices, axis=-1).swapaxes(-1, -2) / scale
            )

        keep = attention_mask[:, None, :, None].astype(mx.bool_) & attention_mask[
            :, None, None, :
        ].astype(mx.bool_)
        scores = mx.where(keep, scores, mx.array(-1e4, dtype=scores.dtype))
        probabilities = mx.softmax(scores, axis=-1)
        context = probabilities @ value
        return context.transpose(0, 2, 1, 3).reshape(
            states.shape[0], length, self.all_head_size
        )


class DebertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states, residual):
        return self.layer_norm(self.dense(hidden_states) + residual)


class DebertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)

    def __call__(self, states, attention_mask, relative_embeddings):
        return self.output(
            self.self_attn(states, attention_mask, relative_embeddings), states
        )


class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, states):
        return nn.gelu(self.dense(states))


class DebertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, states, residual):
        return self.layer_norm(self.dense(states) + residual)


class DebertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

    def __call__(self, states, attention_mask, relative_embeddings):
        attended = self.attention(states, attention_mask, relative_embeddings)
        return self.output(self.intermediate(attended), attended)


class DebertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rel_embeddings = nn.Embedding(
            config.position_buckets * 2, config.hidden_size
        )
        self.layers = [DebertaLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, states, attention_mask):
        relative_embeddings = self.layer_norm(self.rel_embeddings.weight)
        for layer in self.layers:
            states = layer(states, attention_mask, relative_embeddings)
        return states


class DebertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)

    def __call__(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.bool_)
        states = self.embeddings(input_ids, attention_mask)
        return self.encoder(states, attention_mask)


__all__ = ["DebertaModel"]
