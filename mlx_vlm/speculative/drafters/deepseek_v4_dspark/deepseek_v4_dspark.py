"""DeepSeek-V4 DSpark speculative drafter.

A DeepSeek-V4-backbone variant of the model-agnostic DSpark drafter (``..dspark``).
It reuses the shared DSpark proposal machinery — ``VanillaMarkov`` block sampling
and the ``dflash`` round loop with its target-hidden tap — and only swaps the
Qwen-style draft layers for DeepSeek-V4 blocks (MLA attention, MoE,
Hyper-Connections), mirroring DeepSeek-V4-Flash-0731's native DSpark head.

Like the base DSpark drafter, it conforms to the DFlash drafter contract
(``reset`` / ``_hidden`` / ``_logits`` / ``draft_block``): the projected target
hidden (``main_proj`` over the concatenated ``target_layer_ids`` hiddens) is the
accumulating attention context, and the drafted block — seeded with
``mask_token_id`` — supplies the queries.
"""

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Callable, List

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import (
    HyperConnection,
    HyperHead,
    hc_expand,
)
from ....models.deepseek_v4.language import DeepseekV4MoE, LocalAttention
from ..dspark.dspark import VanillaMarkov
from .config import DeepseekV4DsparkConfig


class DSparkMLACrossAttention(LocalAttention):
    """DeepSeek-V4 MLA attention in the DFlash cross-attention layout.

    The accumulating KV context is the projected target hidden ``x_ctx``
    (cached); the drafted block supplies the queries and a transient block KV.
    Block self-attention is non-causal (the whole block is denoised at once).
    """

    def __call__(
        self, x: mx.array, x_ctx: mx.array, cache: RotatingKVCache
    ) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        eps = self.config.rms_norm_eps
        offset = cache.offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset + S)

        ctx_kv = self.kv_norm(self.wkv(x_ctx)).reshape(B, 1, S, self.head_dim)
        ctx_kv = self.rope(ctx_kv, offset)
        block_kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        block_kv = self.rope(block_kv, offset + S)

        kv, _ = cache.update_and_fetch(ctx_kv, mx.zeros((B, 1, S, 0)))
        kv = mx.concatenate([kv, block_kv], axis=2)

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset + S, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class DSparkStage(nn.Module):
    """One DSpark transformer stage: a DeepSeek-V4 HC block whose attention reads
    the projected target hidden. Same parameter names as ``DeepseekV4Block`` so
    the checkpoint maps straight in, plus the stage-specific input/output modules."""

    def __init__(self, config: DeepseekV4DsparkConfig, stage_id: int):
        super().__init__()
        text_config = config.text_config
        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            compress_ratios=[0],
            num_hash_layers=0,
        )
        self.is_first = stage_id == 0
        self.is_last = stage_id == config.n_mtp_layers - 1

        self.attn = DSparkMLACrossAttention(layer_config, layer_idx=0)
        self.ffn = DeepseekV4MoE(layer_config, layer_idx=0)
        self.attn_norm = nn.RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.ffn_norm = nn.RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(layer_config)
        self.ffn_hc = HyperConnection(layer_config)

        if self.is_first:
            n_targets = max(len(config.target_layer_ids), 1)
            self.main_proj = nn.Linear(
                text_config.hidden_size * n_targets,
                text_config.hidden_size,
                bias=False,
            )
            self.main_norm = nn.RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
        if self.is_last:
            self.hc_head = HyperHead(layer_config)
            self.norm = nn.RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )

    def __call__(
        self, h: mx.array, main_x: mx.array, cache: RotatingKVCache
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), main_x, cache)
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), None)
        return hc_expand(x, residual, post, comb)


class DeepseekV4DsparkDraftModel(nn.Module):
    def __init__(self, config: DeepseekV4DsparkConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV4DsparkConfig.text_config must be set")
        self.args = text_config
        self.hc_mult = text_config.hc_mult

        self.stages = [
            DSparkStage(config, stage_id) for stage_id in range(config.n_mtp_layers)
        ]
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)

        self.prefer_requested_block_size = config.block_size_policy == "fixed"
        self.dflash_initial_block_size = config.dflash_initial_block_size
        self._input_embed = None
        self._lm_head_fn = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "DeepseekV4DsparkDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self._input_embed = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self) -> List[RotatingKVCache]:
        return [RotatingKVCache(max_size=self.args.sliding_window) for _ in self.stages]

    def reset(self, target_model) -> List[RotatingKVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def _hidden(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[RotatingKVCache],
    ) -> mx.array:
        first = self.stages[0]
        main_x = first.main_norm(first.main_proj(target_hidden))

        h = self._input_embed(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[-1])
        )
        h = mx.contiguous(h)
        for stage, stage_cache in zip(self.stages, cache):
            h = stage(h, main_x, stage_cache)

        last = self.stages[-1]
        return last.norm(last.hc_head(h))

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[RotatingKVCache],
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
            )

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
            (anchor.shape[0], proposal_length - 1),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        base_logits = self._logits(self._hidden(draft_inputs, hidden, cache))
        return self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            sampler=sampler,
        ).astype(token_dtype)

    def sanitize(self, weights: dict) -> dict:
        """Map the ``mtp.<stage>.*`` checkpoint layout onto the drafter.

        Reuses the proven ``deepseek_v4_mtp`` block sanitizer per stage, then
        reprefixes to ``stages.<i>.*``. The markov head is model-level, and the
        confidence head (unused by the ``dflash`` loop) is dropped.
        """
        from ..deepseek_v4_mtp.deepseek_v4_mtp import DeepseekV4MTPDraftModel

        context = SimpleNamespace(args=self.args)
        by_stage: dict = {}
        out: dict = {}
        for key, value in weights.items():
            match = re.match(r"mtp\.(\d+)\.", key)
            if match:
                by_stage.setdefault(int(match.group(1)), {})[key] = value
            else:
                out[key] = value

        for stage_id, stage_weights in by_stage.items():
            mapped = DeepseekV4MTPDraftModel.sanitize(context, stage_weights)
            for key, value in mapped.items():
                body = key[len("decoder.") :] if key.startswith("decoder.") else key
                if body.startswith("confidence_head."):
                    continue
                if body.startswith("markov_head."):
                    out[body] = value
                else:
                    out[f"stages.{stage_id}.{body}"] = value
        return out


Model = DeepseekV4DsparkDraftModel
