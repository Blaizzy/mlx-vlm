import mlx.core as mx
import mlx.nn as nn

from ....models.cache import KVCache
from ..qwen3_dflash.dflash import DFlashDraftModel
from .config import DSparkConfig


class VanillaMarkov(nn.Module):
    """Low-rank previous-token bias used by DSpark draft heads."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.prev_embeddings(token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class DSparkDraftModel(DFlashDraftModel):
    """Native DSpark drafter using mlx-vlm's verify and rollback engine."""

    prefer_requested_block_size = False
    fixed_requested_block_size = True
    dflash_initial_block_size = 4
    dflash_min_block_size = 3
    quantize_on_load = True

    def __init__(self, config: DSparkConfig):
        super().__init__(config)
        self.config = config
        # DSpark's anchor position also predicts the first proposal, so a
        # trained width of N can draft N tokens and verifies N + 1 positions.
        self.max_runtime_block_size = int(config.block_size) + 1
        # DSpark prefill retains only the requested target hidden-state taps.
        # Budget four bytes per value conservatively; the Qwen target normally
        # emits BF16 activations, but this also covers FP32 target variants.
        self.prefill_memory_bytes_per_token = (
            int(config.hidden_size) * len(config.target_layer_ids) * 4
        )
        if config.markov_head_type != "vanilla":
            raise ValueError(
                "Only DSpark's vanilla Markov head is currently supported."
            )
        self.markov_head = (
            VanillaMarkov(config) if int(config.markov_rank) > 0 else None
        )
        self.confidence_head = None
        if config.enable_confidence_head:
            input_dim = config.hidden_size
            if config.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov requires markov_rank > 0."
                    )
                input_dim += config.markov_rank
            self.confidence_head = ConfidenceHead(input_dim)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: list[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        # mlx-vlm's runtime block size includes the known bonus token. The
        # checkpoint width counts predictive block positions, including the
        # anchor position that predicts the first proposal.
        draft_count = min(
            max(0, int(block_size) - 1),
            int(self.config.block_size) - int(self.config.logits_start),
        )
        if draft_count == 0:
            if isinstance(last_bonus, int):
                return mx.zeros((1, 0), dtype=token_dtype)
            return mx.zeros((int(last_bonus.shape[0]), 0), dtype=token_dtype)

        if isinstance(last_bonus, int):
            bonus = mx.array([last_bonus], dtype=token_dtype)
        else:
            bonus = last_bonus.reshape(-1).astype(token_dtype)

        batch_size = int(bonus.shape[0])
        trained_width = int(self.config.block_size)
        masks = mx.full(
            (batch_size, trained_width - 1),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        block = mx.concatenate([bonus[:, None], masks], axis=1)

        # The published Qwen3.8 head is bidirectional inside its seven-token
        # block. Always run that full width; truncating it changes every row.
        draft_hidden = self._hidden(block, hidden, cache)
        start = int(self.config.logits_start)
        head_hidden = draft_hidden[:, start : start + draft_count]
        base_logits = self._logits(head_hidden)

        if self.markov_head is None:
            return sampler(base_logits).astype(token_dtype)

        tokens = []
        previous = bonus
        for position in range(draft_count):
            logits = base_logits[:, position] + self.markov_head.step_bias(previous)
            next_token = sampler(logits[:, None, :]).reshape(batch_size)
            next_token = next_token.astype(token_dtype)
            tokens.append(next_token[:, None])
            previous = next_token
        return mx.concatenate(tokens, axis=1)

    def validate_target_compatibility(self, target_model) -> None:
        target_config = getattr(target_model, "config", None)
        target_config = getattr(target_config, "text_config", target_config)
        if target_config is None:
            return

        for field in ("hidden_size", "vocab_size"):
            expected = getattr(self.config, field)
            actual = getattr(target_config, field, None)
            if actual is not None and int(actual) != int(expected):
                raise ValueError(
                    "DSpark drafter is incompatible with the target model: "
                    f"{field}={expected} for the drafter, {actual} for the target."
                )

        target_layers = getattr(target_config, "num_hidden_layers", None)
        last_capture = max(self.config.target_layer_ids)
        if target_layers is not None and last_capture >= int(target_layers):
            raise ValueError(
                "DSpark drafter capture layer is outside the target model: "
                f"layer {last_capture}, target has {target_layers} layers."
            )
