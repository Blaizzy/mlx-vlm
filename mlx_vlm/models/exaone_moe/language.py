from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from ..transformer_block import BlockSpec, MoESpec, TransformerBlock
from .config import ModelConfig


def block_spec(args: ModelConfig, layer_idx: int) -> BlockSpec:
    has_sliding = "sliding_attention" in args.layer_types
    is_sliding = args.layer_types[layer_idx] == "sliding_attention"
    use_rope = is_sliding or not has_sliding
    head_dim = args.head_dim

    rope = None
    if use_rope:
        rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    moe = None
    if args.is_moe_layer[layer_idx]:
        num_shared = args.num_shared_experts or 0
        moe = MoESpec(
            hidden_size=args.hidden_size,
            moe_intermediate_size=args.moe_intermediate_size,
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            scoring="sigmoid",
            use_correction_bias=True,
            n_group=args.n_group,
            topk_group=args.topk_group,
            norm_topk_prob=args.norm_topk_prob,
            norm_guard_topk=True,
            norm_denom="add",
            norm_eps=1e-20,
            routed_scaling_factor=args.routed_scaling_factor,
            num_shared_experts=num_shared,
            shared_intermediate_size=(
                args.moe_intermediate_size * num_shared if num_shared else None
            ),
        )

    return BlockSpec(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        intermediate_size=args.intermediate_size,
        rope=rope,
        rms_norm_eps=args.rms_norm_eps,
        qk_norm=True,
        use_rope=use_rope,
        use_sliding=is_sliding,
        moe=moe,
    )


class ExaoneMoEModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(block_spec(args, i)) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.is_sliding = [
            args.layer_types[i] == "sliding_attention"
            for i in range(args.num_hidden_layers)
        ]
        self.swa_idx = self.is_sliding.index(True) if any(self.is_sliding) else None
        self.ga_idx = self.is_sliding.index(False) if not all(self.is_sliding) else None
        self.window_size = args.sliding_window

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h, cache[self.ga_idx] if self.ga_idx is not None else cache[0]
        )
        swa_mask = create_attention_mask(
            h,
            cache[self.swa_idx] if self.swa_idx is not None else cache[0],
            window_size=self.window_size,
        )

        for is_sliding, layer, c in zip(self.is_sliding, self.layers, cache):
            mask = swa_mask if is_sliding else global_mask
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExaoneMoEModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        new_weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            if not self.args.is_moe_layer[l]:
                continue

            prefix = f"model.layers.{l}"

            bias_key = f"{prefix}.mlp.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = weights.pop(
                    bias_key
                )

            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    last_key = (
                        f"{prefix}.mlp.experts.{self.args.num_experts - 1}.{m}.{k}"
                    )
                    if first_key in weights and last_key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for is_sliding in self.model.is_sliding:
            if is_sliding:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
