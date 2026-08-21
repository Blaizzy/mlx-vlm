from collections.abc import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_dflash.dflash import DFlashDraftModel
from .config import DSparkConfig


class VanillaMarkov(nn.Module):
    """Low-rank token-transition correction used by DSpark proposals."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def apply_step_logits(
        self,
        logits: mx.array,
        token_ids: mx.array,
    ) -> mx.array:
        return logits + self.markov_w2(self.get_prev_embeddings(token_ids))

    def sample_block(
        self,
        base_logits: mx.array,
        *,
        first_prev_tokens: mx.array,
        sampler: Callable[[mx.array], mx.array],
    ) -> mx.array:
        proposal_length = int(base_logits.shape[1])
        if proposal_length == 0:
            return mx.zeros((base_logits.shape[0], 0), dtype=mx.int32)

        prev_tokens = first_prev_tokens.reshape(-1)
        sampled = []
        for step in range(proposal_length):
            step_logits = self.apply_step_logits(base_logits[:, step], prev_tokens)
            prev_tokens = sampler(step_logits).reshape(-1)
            sampled.append(prev_tokens[:, None])
        return mx.concatenate(sampled, axis=1)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, markov_rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        markov_embeddings: mx.array,
    ) -> mx.array:
        features = mx.concatenate([hidden_states, markov_embeddings], axis=-1)
        return mx.sigmoid(self.proj(features).squeeze(-1).astype(mx.float32))


def validate_lfm2_dspark_target(config: DSparkConfig, target_model) -> None:
    language_model = getattr(target_model, "language_model", target_model)
    target_config = getattr(language_model, "config", None)
    inner = getattr(language_model, "model", language_model)
    layers = getattr(inner, "layers", None)

    model_type = getattr(target_config, "model_type", None)
    if model_type not in {"lfm2", "lfm2_moe"}:
        raise ValueError(
            "LFM2 DSpark requires an LFM2 target (dense or MoE), got "
            f"model_type={model_type!r}."
        )
    hidden_size = getattr(target_config, "hidden_size", None)
    if hidden_size != config.hidden_size:
        raise ValueError(
            "LFM2 DSpark target hidden-size mismatch: "
            f"draft={config.hidden_size}, target={hidden_size}."
        )
    layer_count = (
        len(layers)
        if layers is not None
        else getattr(target_config, "num_hidden_layers", None)
    )
    if layer_count != config.num_target_layers:
        raise ValueError(
            "LFM2 DSpark target layer-count mismatch: "
            f"draft={config.num_target_layers}, target={layer_count}."
        )
    vocab_size = getattr(target_config, "vocab_size", None)
    if vocab_size != config.vocab_size:
        raise ValueError(
            "LFM2 DSpark target vocabulary mismatch: "
            f"draft={config.vocab_size}, target={vocab_size}."
        )
    if not hasattr(language_model, "rollback_speculative_cache"):
        raise ValueError(
            "The LFM2 target does not expose speculative cache rollback support."
        )


class DSparkDraftModel(DFlashDraftModel):
    """MLX port of Liquid AI's LFM2.5 DSpark draft model."""

    requires_greedy_sampling = True
    # DSpark is trained for a fixed semi-autoregressive proposal block. The
    # generic DFlash acceptance controller is counterproductive here because
    # it quickly collapses the nine-token block to only 2-4 proposals.
    prefer_requested_block_size = True

    def __init__(self, config: DSparkConfig):
        super().__init__(config)
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)
        self.confidence_head: DSparkConfidenceHead | None = (
            DSparkConfidenceHead(config.hidden_size, config.markov_rank)
            if config.enable_confidence_head
            else None
        )

    def validate_target_compatibility(self, target_model) -> None:
        validate_lfm2_dspark_target(self.config, target_model)

    def bind(self, target_model) -> "DSparkDraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def confidence(
        self,
        draft_hidden: mx.array,
        anchor_tokens: mx.array,
        draft_tokens: mx.array,
    ) -> mx.array | None:
        if self.confidence_head is None:
            return None
        previous = mx.concatenate(
            [anchor_tokens.reshape(-1, 1), draft_tokens[:, :-1]], axis=1
        )
        markov_embeddings = self.markov_head.get_prev_embeddings(previous)
        return self.confidence_head(draft_hidden, markov_embeddings)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        # mlx-vlm's block_size includes the anchor; DSpark produces one token
        # per draft hidden position, so its denoising input has gamma positions.
        proposal_length = int(block_size) - 1
        if proposal_length <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)

        if isinstance(last_bonus, int):
            anchor = mx.array([last_bonus], dtype=token_dtype)
        else:
            anchor = last_bonus.reshape(-1).astype(token_dtype)
        masks = mx.full(
            (anchor.shape[0], proposal_length - 1),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = self._hidden(draft_inputs, hidden, cache)
        base_logits = self._logits(draft_hidden)
        return self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            sampler=sampler,
        ).astype(token_dtype)

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in normalized:
                raise ValueError(
                    f"Duplicate LFM2 DSpark weight key after sanitization: {key}"
                )
            normalized[key] = value
        return normalized


Model = DSparkDraftModel


__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftModel",
    "Model",
    "VanillaMarkov",
    "validate_lfm2_dspark_target",
]
