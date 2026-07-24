import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from ..rope_utils import initialize_rope
from ..transformer_block import BlockSpec, MoESpec, TransformerBlock
from .config import ModelConfig


def _is_moe_layer(args: ModelConfig, layer_idx: int) -> bool:
    start = (
        min(args.moe_layer_start_index)
        if isinstance(args.moe_layer_start_index, (tuple, list))
        else args.moe_layer_start_index
    )
    if args.moe_layer_end_index is None:
        end = args.num_hidden_layers - 1
    else:
        end = (
            max(args.moe_layer_end_index)
            if isinstance(args.moe_layer_end_index, (tuple, list))
            else args.moe_layer_end_index
        )
    return (
        (layer_idx + 1) % args.moe_layer_interval == 0
        and layer_idx >= start
        and layer_idx <= end
    )


def _moe_spec(args: ModelConfig) -> MoESpec:
    moe_intermediate_size = args.moe_intermediate_size or args.intermediate_size
    if args.moe_num_shared_experts > 0:
        shared_intermediate_size = (
            args.moe_intermediate_size * args.moe_num_shared_experts
            if args.moe_intermediate_size
            else args.intermediate_size * args.moe_num_shared_experts
        )
    else:
        shared_intermediate_size = None
    return MoESpec(
        hidden_size=args.hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=args.moe_num_experts,
        num_experts_per_tok=args.moe_k,
        scoring=args.moe_gate_act,
        expert_bias=args.use_bias,
        num_shared_experts=args.moe_num_shared_experts,
        shared_intermediate_size=shared_intermediate_size,
        shared_bias=args.use_bias,
    )


def _block_spec(args: ModelConfig, layer_idx: int) -> BlockSpec:
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
    rope = initialize_rope(
        head_dim,
        base=args.rope_theta,
        traditional=True,
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
        attn_bias=args.use_bias,
        mlp_bias=args.use_bias,
        moe=_moe_spec(args) if _is_moe_layer(args, layer_idx) else None,
    )


class Ernie45Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(_block_spec(args, i))
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
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
        self.model = Ernie45Model(args)
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
        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
            "e_score_correction_bias",
        ]
        weights = {
            key: value
            for key, value in weights.items()
            if not any(pattern in key for pattern in remove_patterns)
        }

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                if f"{prefix}.mlp.experts.0.{m}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.weight")
                        for e in range(self.args.moe_num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.weight"] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
