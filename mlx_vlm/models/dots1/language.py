import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from ..rope_utils import initialize_rope
from ..transformer_block import BlockSpec, MoESpec, TransformerBlock
from .config import ModelConfig


def _moe_spec(args: ModelConfig) -> MoESpec:
    return MoESpec(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        scoring="sigmoid",
        use_correction_bias=True,
        n_group=args.n_group or 1,
        topk_group=args.topk_group or 1,
        norm_topk_prob=args.norm_topk_prob,
        norm_guard_topk=True,
        norm_denom="add",
        norm_eps=0.0,
        routed_scaling_factor=args.routed_scaling_factor,
        expert_attr="experts",
        num_shared_experts=args.n_shared_experts,
        shared_intermediate_size=args.moe_intermediate_size * args.n_shared_experts,
        shared_bias=args.mlp_bias,
    )


def _block_spec(args: ModelConfig, layer_idx: int) -> BlockSpec:
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
    rope = initialize_rope(
        head_dim,
        base=args.rope_theta,
        traditional=False,
        scaling_config=args.rope_scaling,
        max_position_embeddings=args.max_position_embeddings,
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
        mlp_bias=args.mlp_bias,
        moe=_moe_spec(args) if layer_idx >= args.first_k_dense_replace else None,
    )


class Dots1Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(_block_spec(args, layer_idx))
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Dots1Model(args)
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
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            if l >= self.args.first_k_dense_replace:
                for m in ("gate_proj", "down_proj", "up_proj"):
                    for k in ("weight", "scales", "biases"):
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.n_routed_experts)
                            ]
                            weights[f"{prefix}.mlp.experts.{m}.{k}"] = mx.stack(to_join)

        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]

    @property
    def layers(self):
        return self.model.layers
