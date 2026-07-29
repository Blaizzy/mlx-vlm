from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx


@dataclass
class EmbeddingOutput:
    last_hidden_state: Optional[mx.array] = None
    text_embeds: Optional[mx.array] = None


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )


def mean_pooling(token_embeddings: mx.array, attention_mask: mx.array):
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(token_embeddings: mx.array, attention_mask: mx.array) -> mx.array:
    first_indices = mx.argmax(attention_mask, axis=1)
    batch_size = token_embeddings.shape[0]
    hidden_dim = token_embeddings.shape[-1]
    gather_idx = mx.broadcast_to(
        first_indices[:, None, None], (batch_size, 1, hidden_dim)
    )
    return mx.squeeze(mx.take_along_axis(token_embeddings, gather_idx, axis=1), axis=1)


def max_pooling(token_embeddings: mx.array, attention_mask: mx.array) -> mx.array:
    mask = mx.expand_dims(attention_mask, -1)
    mask = mx.broadcast_to(mask, token_embeddings.shape).astype(token_embeddings.dtype)
    masked = mx.where(mask == 0, -float("inf"), token_embeddings)
    return mx.max(masked, axis=1)


def lasttoken_pooling(token_embeddings: mx.array, attention_mask: mx.array) -> mx.array:
    batch_size, seq_len, hidden_dim = token_embeddings.shape
    flipped = attention_mask[:, ::-1]
    flip_indices = mx.argmax(flipped, axis=1)
    has_any_real = mx.max(flipped, axis=1)
    flip_indices = mx.where(has_any_real == 0, seq_len - 1, flip_indices)
    last_indices = seq_len - flip_indices - 1
    gather_idx = mx.broadcast_to(
        last_indices[:, None, None], (batch_size, 1, hidden_dim)
    )
    mask = mx.broadcast_to(attention_mask[:, :, None], token_embeddings.shape).astype(
        token_embeddings.dtype
    )
    return mx.squeeze(
        mx.take_along_axis(token_embeddings * mask, gather_idx, axis=1), axis=1
    )


_LEGACY_POOLING_MODE_KWARGS = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}

_SUPPORTED_POOL_MODES = {"cls", "mean", "max", "lasttoken"}
_KNOWN_UNSUPPORTED_POOL_MODES = {"weightedmean", "mean_sqrt_len_tokens"}


def _normalize_pooling_config(pooling_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(pooling_config)
    found = [k for k in _LEGACY_POOLING_MODE_KWARGS if k in cfg]
    if not found:
        return cfg
    if "pooling_mode" not in cfg:
        active = tuple(
            name
            for key, name in _LEGACY_POOLING_MODE_KWARGS.items()
            if cfg.get(key, False)
        )
        if not active:
            active = ("mean",)
        cfg["pooling_mode"] = active[0] if len(active) == 1 else active
    for k in found:
        del cfg[k]
    return cfg


def pool_by_config(
    token_embeddings: mx.array,
    attention_mask: mx.array,
    pooling_config: Dict[str, Any],
) -> mx.array:
    cfg = _normalize_pooling_config(pooling_config)
    mode = cfg["pooling_mode"]
    if not cfg.get("include_prompt", True):
        raise NotImplementedError(
            "Prompt-aware pooling (include_prompt=False) is not supported. "
            "This affects INSTRUCTOR-style models."
        )
    if isinstance(mode, (tuple, list)):
        raise NotImplementedError(
            f"Concatenated pooling mode {mode!r} is not supported; "
            "only a single pooling mode is allowed."
        )
    if mode in _KNOWN_UNSUPPORTED_POOL_MODES:
        raise NotImplementedError(
            f"Pooling mode {mode!r} is not supported. "
            f"Supported modes: {sorted(_SUPPORTED_POOL_MODES)}."
        )
    if mode == "cls":
        return cls_pooling(token_embeddings, attention_mask)
    if mode == "max":
        return max_pooling(token_embeddings, attention_mask)
    if mode == "lasttoken":
        return lasttoken_pooling(token_embeddings, attention_mask)
    if mode == "mean":
        return mean_pooling(token_embeddings, attention_mask)
    raise ValueError(
        f"Unknown pooling mode {mode!r}. "
        f"Supported modes: {sorted(_SUPPORTED_POOL_MODES)}."
    )
