"""DeepSeek-V4.1 DSpark speculative drafter.

DSpark is DeepSeek-V4.1-Flash's native multi-token-prediction head: a stack of
``n_mtp_layers`` (3) transformer stages drafting ``dspark_block_size`` (5)
tokens per step, conditioned on the concatenated hidden states of the target's
``dspark_target_layer_ids`` (the ``main_hidden`` window).

Structure mirrors the reference ``inference/model.py`` ``DSparkBlock``: stage 0
owns ``main_proj``/``main_norm`` over the target hidden, every stage is an MLA
attention (KV from the projected target hidden plus the block's own tokens) +
MoE block under single-pass mHC, and only the last stage owns the final norm
plus ``markov_head`` (low-rank bigram bias) and ``confidence_head``. The
attention here is a dense mirror of the reference sparse windowed kernel; the
markov/confidence heads are imported from the model path, whose
``embed/head/proj`` names already match this checkpoint layout.
"""

from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.deepseek_v4.hyper_connection import hc_expand
from ....models.deepseek_v4.language import DeepseekV4RoPE
from ....models.deepseek_v41.dspark import DSparkConfidenceHead, DSparkMarkovHead
from ....models.deepseek_v41.language import (
    DeepseekV41Block,
    DeepseekV41MoE,
    hc_mix_coeffs,
    make_identity_pre_mix,
)
from ....models.mla import MultiLinear
from .config import DeepseekV41DsparkConfig


class DeepseekV41DsparkAttention(nn.Module):
    """MLA attention whose KV context is the projected target hidden ``main_x``.

    Queries come from the drafted block; keys/values are the concatenation of
    the target context (``main_x``) and the block's own tokens, with a causal
    in-block mask. Dense mirror of the reference sparse kernel.
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
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )

    def __call__(
        self, x: mx.array, main_x: mx.array, block_offset: int = 0
    ) -> mx.array:
        batch, block = x.shape[0], x.shape[1]
        ctx = main_x.shape[1]

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(batch, block, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, block_offset)

        main_kv = self.kv_norm(self.wkv(main_x)).reshape(batch, 1, ctx, self.head_dim)
        main_kv = self.rope(main_kv, 0)
        block_kv = self.kv_norm(self.wkv(x)).reshape(batch, 1, block, self.head_dim)
        block_kv = self.rope(block_kv, block_offset)
        kv = mx.concatenate([main_kv, block_kv], axis=2)

        cols = mx.arange(ctx + block)[None, None, :]
        rows = mx.arange(block)[None, :, None]
        mask = (cols < ctx) | ((cols - ctx) <= rows)

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, block_offset, inverse=True)

        out = out.reshape(batch, self.o_groups, -1, block, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class DeepseekV41DsparkStage(nn.Module):
    """One DSpark transformer stage under single-pass mHC.

    Structurally a backbone block (same parameter names, so the checkpoint maps
    straight in) apart from the cross-attention and the stage-specific
    input/output modules.
    """

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

        self.attn = DeepseekV41DsparkAttention(text_config)
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
            n_targets = max(len(config.dspark_target_layer_ids), 1)
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
            self.markov_head = DSparkMarkovHead(text_config)
            self.confidence_head = DSparkConfidenceHead(text_config)

    def __call__(
        self, h: mx.array, main_x: mx.array, pre_mix: mx.array, block_offset: int = 0
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
        x = self.attn(x, main_x, block_offset)
        x = hc_expand(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mix_coeffs(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )
        x = DeepseekV41Block.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = hc_expand(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


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
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id

        self.stages = [
            DeepseekV41DsparkStage(config, stage_id)
            for stage_id in range(config.n_mtp_layers)
        ]

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

    def reset(self, target_model) -> None:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []

    def _forward_embed(
        self, main_hidden: mx.array, bonus_token: mx.array, token_dtype: mx.Dtype
    ) -> Tuple[mx.array, mx.array]:
        first = self.stages[0]
        main_x = first.main_norm(first.main_proj(main_hidden))

        batch = main_hidden.shape[0]
        draft_ids = mx.full(
            (batch, self.block_size), self.noise_token_id, dtype=token_dtype
        )
        draft_ids[:, 0] = bonus_token.astype(token_dtype)
        x = self._input_embed(draft_ids)
        x = mx.broadcast_to(
            x[:, :, None, :], (batch, self.block_size, self.hc_mult, x.shape[-1])
        )
        return mx.contiguous(x), main_x

    def _forward_head(
        self,
        x: mx.array,
        pre_mix: mx.array,
        bonus_token: mx.array,
        sampler: Optional[Callable[[mx.array], mx.array]],
        token_dtype: mx.Dtype,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        last = self.stages[-1]
        hidden = DeepseekV41Block.hc_pre(x, pre_mix)
        base_logits = self._lm_head_fn(last.norm(hidden))

        output_ids = [bonus_token.astype(token_dtype)]
        markov_embeds = []
        logits_out = []
        for i in range(self.block_size):
            bias, embed = last.markov_head(output_ids[i])
            logits_i = base_logits[:, i, :] + bias
            markov_embeds.append(embed)
            logits_out.append(logits_i)
            token = (
                mx.argmax(logits_i, axis=-1) if sampler is None else sampler(logits_i)
            )
            output_ids.append(token.astype(token_dtype))

        confidence = last.confidence_head(hidden, mx.stack(markov_embeds, axis=1))
        return (
            mx.stack(output_ids, axis=1),
            mx.stack(logits_out, axis=1),
            confidence,
        )

    def draft_block(
        self,
        main_hidden: mx.array,
        bonus_token: mx.array,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        block_offset: int = 0,
        token_dtype: mx.Dtype = mx.int32,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Draft one block.

        ``main_hidden``: [B, ctx, hidden_size * len(target_layer_ids)] target
        context. ``bonus_token``: [B] accepted token. Returns
        ``(output_ids [B, block_size + 1], logits [B, block_size, vocab],
        confidence [B, block_size])``.
        """
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
            )

        x, main_x = self._forward_embed(main_hidden, bonus_token, token_dtype)
        pre_mix = make_identity_pre_mix(
            main_hidden.shape[0], self.block_size, self.hc_mult
        )
        for stage in self.stages:
            x, pre_mix = stage(x, main_x, pre_mix, block_offset)
        return self._forward_head(x, pre_mix, bonus_token, sampler, token_dtype)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Map the ``mtp.<stage>.*`` checkpoint layout onto ``stages.<i>.*``,
        stacking per-expert tensors for SwitchGLU like the backbone."""
        import re

        from ....models.deepseek_v41.language import sanitize_moe_weights
        from .split import sanitize_dspark_weights

        weights = sanitize_dspark_weights(weights)
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
        return weights


Model = DeepseekV41DsparkDraftModel
