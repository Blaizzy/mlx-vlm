import re
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from ....models.deepseek_v41.deepseek_v41 import _pack_source_weight
from ....models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
from ....models.deepseek_v41.language import (
    DeepseekV41Attention,
    DeepseekV41Block,
    hc_expand,
    make_identity_pre_mix,
    sanitize_moe_weights,
)
from ..deepseek_v4_dspark.deepseek_v4_dspark import DeepseekV4DsparkDraftModel


class DSparkAttention(DeepseekV41Attention):
    def __call__(self, x, context, cache):
        batch, length, _ = x.shape
        count = context.shape[1]
        offset = cache.offset
        kv = self.kv_norm(self.wkv(fake_quant_fp8_ue8m0(context)))[:, None]
        kv = fake_quant_fp8_ue8m0(self.rope(kv, offset))
        kv, _ = cache.update_and_fetch(kv, mx.zeros((*kv.shape[:-1], 0)))
        kv = kv[..., -self.window_size :, :]
        q = self.wq_b(
            fake_quant_fp8_ue8m0(self.q_norm(self.wq_a(fake_quant_fp8_ue8m0(x))))
        )
        q = q.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q, offset + count)
        block_kv = self.kv_norm(self.wkv(fake_quant_fp8_ue8m0(x)))[:, None]
        block_kv = fake_quant_fp8_ue8m0(self.rope(block_kv, offset + count))
        kv = mx.concatenate([kv, block_kv], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            q, kv, kv, scale=self.scale, sinks=self.attn_sink.astype(q.dtype)
        )
        out = self.rope(out, offset + count, inverse=True)
        out = out.reshape(batch, self.o_groups, -1, length, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(fake_quant_fp8_ue8m0(out))
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(fake_quant_fp8_ue8m0(out))


class DSparkStage(DeepseekV41Block):
    def __init__(self, config, stage_id):
        text = config.text_config
        draft = replace(
            text,
            num_hidden_layers=1,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            n_routed_experts=text.dspark_n_routed_experts or text.n_routed_experts,
            num_experts_per_tok=text.dspark_num_experts_per_tok
            or text.num_experts_per_tok,
        )
        super().__init__(draft, 0)
        self.attn = DSparkAttention(draft, 0)
        if stage_id == 0:
            self.main_proj = nn.Linear(
                text.hidden_size * len(config.target_layer_ids),
                text.hidden_size,
                bias=False,
            )
            self.main_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        if stage_id == config.n_mtp_layers - 1:
            self.norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)

    def __call__(self, h, pre_mix, context, cache):
        pre, post, comb = self.hc_mixes(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn(self.attn_norm(self.hc_pre(h, pre_mix)), context, cache)
        h = hc_expand(x, h, post, comb)
        next_pre, post, comb = self.hc_mixes(
            h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn(self.ffn_norm(self.hc_pre(h, pre)), None)
        return hc_expand(x, h, post, comb), next_pre


class DeepseekV41DsparkDraftModel(DeepseekV4DsparkDraftModel):
    stage_class = DSparkStage

    def bind(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        self._input_embed = target.embed_tokens
        self._lm_head_fn = target.head
        return self

    def _hidden(self, inputs, target_hidden, cache):
        first = self.stages[0]
        context = first.main_norm(first.main_proj(fake_quant_fp8_ue8m0(target_hidden)))
        h = self._input_embed(inputs)
        pre_mix = make_identity_pre_mix(h.shape[0], h.shape[1], self.hc_mult)
        h = mx.broadcast_to(h[:, :, None], (*h.shape[:2], self.hc_mult, h.shape[-1]))
        for stage, layer_cache in zip(self.stages, cache):
            h, pre_mix = stage(h, pre_mix, context, layer_cache)
        return self.stages[-1].norm(DeepseekV41Block.hc_pre(h, pre_mix))

    def sanitize(self, weights):
        weights = dict(weights)
        for key in list(weights):
            if key.endswith(".scale"):
                prefix = key[: -len(".scale")]
                packed, scales, _ = _pack_source_weight(
                    weights[prefix + ".weight"], weights.pop(key)
                )
                weights[prefix + ".weight"] = packed
                weights[prefix + ".scales"] = scales
        out = {}
        stages = set()
        for key, value in weights.items():
            match = re.match(r"mtp\.(\d+)\.(.*)", key)
            if match is None:
                out[key] = value
                continue
            stage, body = match.groups()
            stages.add(int(stage))
            if body.startswith("confidence_head."):
                continue
            if body.startswith("markov_head."):
                body = body.replace("markov_head.embed.", "markov_head.markov_w1.")
                body = body.replace("markov_head.head.", "markov_head.markov_w2.")
                out[body] = value.astype(mx.float32)
            else:
                out[f"stages.{stage}.{body}"] = value
        experts = self.args.dspark_n_routed_experts or self.args.n_routed_experts
        for stage in stages:
            prefix = f"stages.{stage}"
            out = sanitize_moe_weights(out, f"{prefix}.ffn", experts)
            gate = f"{prefix}.ffn.gate.weight"
            if gate in out:
                out[gate] = out[gate].astype(mx.float32)
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}.attn.wo_a.{suffix}"
                if key in out and out[key].ndim == 2:
                    out[key] = out[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )
        return out
