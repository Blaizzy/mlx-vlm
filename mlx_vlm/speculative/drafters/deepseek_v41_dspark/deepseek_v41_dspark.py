"""DeepSeek-V4.1 DSpark speculative drafter.

A DeepSeek-V4.1-backbone variant of the model-agnostic DSpark drafter: it reuses
the shared DSpark proposal machinery (``VanillaMarkov`` block sampling and the
``dflash`` round loop with its target-hidden tap) and only swaps the Qwen-style
draft layers for DeepSeek-V4.1 MLA cross-attention, MoE, and single-pass
Hyper-Connections, mirroring the native DSpark head.

Like the base DSpark drafter, it conforms to the DFlash drafter contract
(``reset`` / ``_hidden`` / ``_logits`` / ``draft_block``): the projected target
hidden (``main_proj`` over the concatenated ``target_layer_ids`` hiddens) is the
accumulating attention context, and the drafted block — seeded with
``mask_token_id`` — supplies the queries. Pre-mix coefficients thread through
the stages single-pass style, carried internally from an identity start.
"""

from typing import Callable, List

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.language import DeepseekV4RoPE
from ....models.deepseek_v41.dspark import (  # noqa: F401 (documents the unused native head)
    DSparkConfidenceHead,
)
from ....models.deepseek_v41.language import (
    DeepseekV41Block,
    DeepseekV41MoE,
    hc_mix_coeffs,
    make_identity_pre_mix,
)
from ....models.mla import MultiLinear
from ..dspark.dspark import VanillaMarkov
from .config import DeepseekV41DsparkConfig


class DSparkMLACrossAttention(nn.Module):
    """DeepSeek-V4.1 MLA attention in the DFlash cross-attention layout.

    The accumulating KV context is the projected target hidden ``x_ctx``
    (cached); the drafted block supplies the queries and a transient block KV.
    Block self-attention is non-causal (the whole block is denoised at once).
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

    def __call__(
        self, x: mx.array, x_ctx: mx.array, cache: RotatingKVCache
    ) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        offset = cache.offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
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


class DeepseekV41DsparkStage(nn.Module):
    """One DSpark transformer stage: a DeepSeek-V4.1 HC block whose attention reads
    the projected target hidden. Same parameter names as the backbone block (minus
    compressor/indexer, plus the stage input/output modules) so the checkpoint maps
    straight in."""

    def __init__(self, config: DeepseekV41DsparkConfig, stage_id: int):
        super().__init__()
        text_config = config.text_config
        self.stage_id = stage_id
        self.is_first = stage_id == 0
        self.is_last = stage_id == config.n_mtp_layers - 1
        self.hc_mult = text_config.hc_mult
        self.hc_sinkhorn_iters = text_config.hc_sinkhorn_iters
        self.hc_eps = text_config.hc_eps
        self.norm_eps = text_config.rms_norm_eps

        self.attn = DSparkMLACrossAttention(text_config)
        self.ffn = DeepseekV41MoE(
            text_config,
            moe_intermediate_size=text_config.moe_intermediate_size,
            n_routed_experts=text_config.dspark_n_routed_experts,
            num_experts_per_tok=text_config.dspark_num_experts_per_tok,
        )
        self.attn_norm = nn.RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.ffn_norm = nn.RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * text_config.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.ones((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.ones((3,), dtype=mx.float32)

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
            self.norm = nn.RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )

    def __call__(
        self,
        h: mx.array,
        main_x: mx.array,
        pre_mix: mx.array,
        cache: RotatingKVCache,
    ):
        residual = h
        attn_pre, attn_post, attn_comb = hc_mix_coeffs(
            h,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )
        x = DeepseekV41Block.hc_pre(h, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, main_x, cache)
        h = hc_post(x, residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = hc_mix_coeffs(
            h,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )
        x = DeepseekV41Block.hc_pre(h, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        h = hc_post(x, residual, ffn_post, ffn_comb)
        return h, ffn_pre


def hc_post(x, residual, post, comb):
    """Expand the sublayer output back to hc copies through `comb`."""
    y = post[..., None].astype(mx.float32) * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(
        comb.swapaxes(-1, -2).astype(mx.float32), residual.astype(mx.float32)
    )
    return y.astype(x.dtype)


class DeepseekV41DsparkDraftModel(nn.Module):
    prefer_requested_block_size = True

    def __init__(self, config: DeepseekV41DsparkConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV41DsparkConfig.text_config must be set")
        self.args = text_config
        self.hc_mult = text_config.hc_mult

        self.stages = [
            DeepseekV41DsparkStage(config, stage_id)
            for stage_id in range(config.n_mtp_layers)
        ]
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)

        self._input_embed = None
        self._lm_head_fn = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "DeepseekV41DsparkDraftModel":
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
            if hasattr(target_model, "language_model") and hasattr(
                target_model.language_model, "embed_tokens"
            ):
                inner = target_model.language_model
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
        from ....models.cache import RotatingKVCache as _RotCache

        return [_RotCache(max_size=512) for _ in self.stages]

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
        pre_mix = make_identity_pre_mix(h.shape[0], h.shape[1], self.hc_mult)
        for stage, stage_cache in zip(self.stages, cache):
            h, pre_mix = stage(h, main_x, pre_mix, stage_cache)

        last = self.stages[-1]
        return last.norm(DeepseekV41Block.hc_pre(h, pre_mix))

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

        Reuses the proven per-stage mapping, then renames the model-level
        native markov head onto ``VanillaMarkov``. That head is stored inside
        the last stage in released checkpoints and at the top level in split
        ones, so both spellings are accepted. The native confidence head is
        unused by the ``dflash`` loop and is dropped.
        """
        import re

        from ....models.deepseek_v41.language import sanitize_moe_weights
        from .split import sanitize_dspark_weights

        weights = sanitize_dspark_weights(weights)
        markov_re = re.compile(r"^(?:stages\.\d+\.)?markov_head\.(embed|head)\.(.+)$")
        for key in [k for k in weights if markov_re.match(k)]:
            m = markov_re.match(key)
            slot = "markov_w1" if m.group(1) == "embed" else "markov_w2"
            weights[f"markov_head.{slot}.{m.group(2)}"] = weights.pop(key)
        confidence_re = re.compile(r"^(?:stages\.\d+\.)?confidence_head\.")
        weights = {k: v for k, v in weights.items() if not confidence_re.match(k)}
        stages = {
            int(m.group(1))
            for k in weights
            if (m := re.match(r"stages\.(\d+)\.ffn\.experts\.0\.w1\.weight", k))
        }
        for stage in sorted(stages):
            prefix = f"stages.{stage}.ffn"
            n_routed = (
                max(
                    int(m.group(1))
                    for k in weights
                    if (
                        m := re.match(
                            rf"{re.escape(prefix)}\.experts\.(\d+)\.w1\.weight", k
                        )
                    )
                )
                + 1
            )
            weights = sanitize_moe_weights(weights, prefix, n_routed)
        text_config = self.config.text_config
        for stage in sorted(stages):
            wo_prefix = f"stages.{stage}.attn.wo_a"
            for key in (
                f"{wo_prefix}.weight",
                f"{wo_prefix}.scales",
                f"{wo_prefix}.biases",
            ):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        text_config.o_groups, text_config.o_lora_rank, -1
                    )
        return weights


Model = DeepseekV41DsparkDraftModel
