from collections.abc import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn

from ..compatibility import validate_dflash_target
from ..qwen3_dflash.dflash import DFlashDecoderLayer, DFlashDraftModel
from .config import DFlash2Config


def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else mx.concatenate(
                [mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]], axis=1
            )
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + (kernel + dynamic[:, :, offset]) * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        prepared = _grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
        )
        return prepared, dynamic[..., 1, :, :]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlash2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache, projected_context=None):
        residual = x
        x, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(
            self.self_attn(x, x_ctx, rope, cache, projected_context), kernel
        )
        residual = x
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(
            config.vocab_size, config.selector_rank
        )
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        sampler: Callable[[mx.array], mx.array],
    ) -> mx.array:
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids.reshape(-1)
        path = []
        sample_proposal = getattr(sampler, "sample_proposal", None)
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            selected = (
                sample_proposal(scores)
                if callable(sample_proposal)
                else mx.argmax(scores, axis=-1)
            ).reshape(-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)


class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer
    prefer_requested_block_size = False
    dflash_initial_block_size = 3
    dflash_min_block_size = 3

    def __init__(self, config: DFlash2Config):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def validate_target_compatibility(self, target_model) -> None:
        validate_dflash_target(self.config, target_model, "DFlash2")

    def bind(self, target_model) -> "DFlash2DraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return (
            self.embed_tokens(inputs)
            * self.embed_scale
            * self.config.input_embedding_scale
        )

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden) * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        proposal_length = int(block_size) - 1
        if proposal_length <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        masks = mx.full(
            (anchor.shape[0], proposal_length),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = self._hidden(draft_inputs, hidden, cache)[:, 1:]
        return self.candidate_selector.select(
            draft_hidden,
            self._logits(draft_hidden),
            anchor,
            sampler,
        ).astype(token_dtype)

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        codebooks = {
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        }
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in codebooks:
                key = f"{key}.weight"
            if key in normalized:
                raise ValueError(
                    f"Duplicate DFlash2 weight key after sanitization: {key}"
                )
            normalized[key] = value
        return normalized


Model = DFlash2DraftModel


__all__ = [
    "CandidateSelector",
    "DFlash2DecoderLayer",
    "DFlash2DraftModel",
    "GroupedDynamicCausalConv",
    "Model",
    "_grouped_dynamic_convolve",
]
