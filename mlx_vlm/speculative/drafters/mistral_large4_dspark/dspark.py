"""Mistral's bundled BF16 DSpark head, following vLLM's native implementation.

The shared DFlash loop verifies proposals and commits only accepted target
positions. Draft caches store projected target features, never rejected queries.
"""

import mlx.core as mx
import mlx.nn as nn

from ....models.cache import KVCache, RotatingKVCache
from ....models.mistral_large4.language import MLP, LeRoPE, rotate
from .config import ModelConfig


class DraftAttention(nn.Module):
    def __init__(self, args, window):
        super().__init__()
        self.heads = args.num_attention_heads
        self.nope_dim = args.qk_nope_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.head_dim = self.nope_dim + self.rope_dim
        self.latent = args.kv_lora_rank
        self.value_dim = args.v_head_dim
        self.window = window
        self.wq = nn.Linear(args.hidden_size, self.heads * self.head_dim, bias=False)
        self.wkv_a_with_mqa = nn.Linear(
            args.hidden_size, self.latent + self.rope_dim, bias=False
        )
        self.kv_a_norm = nn.RMSNorm(self.latent, eps=args.rms_norm_eps)
        self.wkv_b = nn.Linear(
            self.latent, self.heads * (self.nope_dim + self.value_dim), bias=False
        )
        self.wo = nn.Linear(self.heads * self.value_dim, args.hidden_size, bias=False)
        # The drafter uses plain YaRN: no trained LeRoPE alpha or SSMax scaling.
        self._inv_freq = LeRoPE(args)._inv_freq

    def _rope(self, x, offset):
        positions = mx.arange(x.shape[2], dtype=mx.float32) + offset
        angles = positions[:, None] * self._inv_freq
        return rotate(x, mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype))

    def _kv(self, x, offset):
        kv = self.wkv_a_with_mqa(x)
        latent = self.kv_a_norm(kv[..., : self.latent])[:, None]
        rotary = self._rope(kv[..., self.latent :][:, None], offset)
        return latent, rotary

    def __call__(self, x, context, cache):
        b, length, _ = x.shape
        offset = cache.offset
        context_length = context.shape[1]
        # Skip evicted prompt context before projection, retaining absolute RoPE.
        if self.window is not None and context_length > self.window:
            skip = context_length - self.window
            context = context[:, skip:]
            cache.offset += skip
            offset += skip
            context_length -= skip
        latent, rotary = cache.update_and_fetch(*self._kv(context, offset))
        if self.window is not None:
            latent, rotary = (
                latent[..., -self.window :, :],
                rotary[..., -self.window :, :],
            )
        query_offset = offset + context_length
        block_latent, block_rotary = self._kv(x, query_offset)
        latent = mx.concatenate([latent, block_latent], axis=2)
        rotary = mx.concatenate([rotary, block_rotary], axis=2)
        q = (
            self.wq(x)
            .reshape(b, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = mx.concatenate(
            [
                q[..., : self.nope_dim],
                self._rope(q[..., self.nope_dim :], query_offset),
            ],
            axis=-1,
        )
        total = latent.shape[2]
        kv = (
            self.wkv_b(latent)
            .reshape(b, total, self.heads, self.nope_dim + self.value_dim)
            .transpose(0, 2, 1, 3)
        )
        k = mx.concatenate(
            [
                kv[..., : self.nope_dim],
                mx.broadcast_to(rotary, (b, self.heads, total, self.rope_dim)),
            ],
            axis=-1,
        )
        # All queries see the retained context and the entire draft block.
        out = mx.fast.scaled_dot_product_attention(
            q, k, kv[..., self.nope_dim :], scale=self.head_dim**-0.5, mask=None
        )
        return self.wo(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class DraftLayer(nn.Module):
    def __init__(self, args, window):
        super().__init__()
        self.attention = DraftAttention(args, window)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, context, cache):
        x = x + self.attention(self.attention_norm(x), context, cache)
        return x + self.feed_forward(self.ffn_norm(x))


class DraftBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = config.text_config
        self.target_projection = (
            nn.Linear(
                args.hidden_size * len(config.target_layer_ids),
                args.hidden_size,
                bias=False,
            )
            if len(config.target_layer_ids) > 1
            else None
        )
        self.target_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        mask_shape = (
            (config.proposal_length - 1, args.hidden_size)
            if config.draft_input == "position_specific"
            else (args.hidden_size,)
        )
        self.mask_embedding = mx.zeros(mask_shape)
        self.layers = [DraftLayer(args, window) for window in config.windows]
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class AutoregressiveHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank = config.markov_rank
        self.embedding = nn.Embedding(config.vocab_size, rank)
        self.output = nn.Linear(rank, config.vocab_size, bias=False)
        self.joint_proj = (
            nn.Linear(config.hidden_size + 2 * rank, 3 * rank, bias=True)
            if config.head_type == "rnn"
            else None
        )

    def step(self, tokens, hidden, state):
        embed = self.embedding(tokens)
        if self.joint_proj is None:
            return self.output(embed), state
        if state is None:
            state = mx.zeros_like(embed)
        gate, candidate, output = mx.split(
            self.joint_proj(mx.concatenate([state, embed, hidden], axis=-1)), 3, axis=-1
        )
        gate = mx.sigmoid(gate)
        state = gate * state + (1 - gate) * mx.tanh(candidate)
        return self.output(mx.tanh(output)), state


class Model(nn.Module):
    prefer_requested_block_size = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = DraftBackbone(config)
        self.autoregressive_head = AutoregressiveHead(config)
        self._input_embed = None
        self._lm_head = None
        self.accept_lens = []
        self.draft_lens = []

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "args", None)
        if (
            args is None
            or args.model_type != "mistral_large4_text"
            or any(
                getattr(args, name) != getattr(self.config.text_config, name)
                for name in ("hidden_size", "vocab_size", "num_hidden_layers")
            )
        ):
            raise ValueError(
                "The DSpark head requires its matching Mistral Large 4 target"
            )

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        target = getattr(target_model, "language_model", target_model)
        self._input_embed = target.model.embed_tokens
        self._lm_head = target.lm_head
        return self

    def make_cache(self):
        return [
            KVCache() if w is None else RotatingKVCache(max_size=w)
            for w in self.config.windows
        ]

    def reset(self, target_model):
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def prepare_target_hidden(self, hidden):
        projection = self.backbone.target_projection
        return self.backbone.target_norm(
            projection(hidden) if projection is not None else hidden
        )

    def draft_block(
        self,
        last_bonus,
        hidden,
        cache,
        block_size,
        sampler,
        token_dtype=mx.int32,
        *,
        target_hidden_prepared=False,
    ):
        count = int(block_size) - 1
        if not 0 <= count <= self.config.proposal_length:
            raise ValueError("Requested DSpark block exceeds the trained depth")
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1)
        )
        if count == 0:
            return mx.zeros((anchor.size, 0), dtype=token_dtype)
        if self._input_embed is None:
            raise RuntimeError("Bind the target before drafting")
        context = (
            hidden if target_hidden_prepared else self.prepare_target_hidden(hidden)
        )
        embed = self._input_embed(anchor[:, None])
        # Always execute the trained block, even when the output budget is smaller.
        masks = mx.broadcast_to(
            self.backbone.mask_embedding.astype(embed.dtype),
            (anchor.size, self.config.proposal_length - 1, embed.shape[-1]),
        )
        x = mx.concatenate([embed, masks], axis=1)
        for layer, entry in zip(self.backbone.layers, cache):
            x = layer(x, context, entry)
        return self._sample_block(x, anchor, count, sampler, token_dtype)

    def draft_batch(self, anchors, hidden, caches, block_size, sampler, token_dtype):
        """Batch dense draft work while retaining independent context caches."""
        count = int(block_size) - 1
        if not 0 <= count <= self.config.proposal_length:
            raise ValueError("Requested DSpark block exceeds the trained depth")
        if count == 0:
            return mx.zeros((len(hidden), 0), dtype=token_dtype)
        lengths = [h.shape[1] for h in hidden]
        context = self.prepare_target_hidden(mx.concatenate(hidden, axis=1))
        boundaries, total = [], 0
        for length in lengths[:-1]:
            total += length
            boundaries.append(total)
        contexts = mx.split(context, boundaries, axis=1)
        anchor = mx.array(anchors, dtype=token_dtype).reshape(-1)
        embed = self._input_embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.backbone.mask_embedding.astype(embed.dtype),
            (anchor.size, self.config.proposal_length - 1, embed.shape[-1]),
        )
        x = mx.concatenate([embed, masks], axis=1)
        for index, layer in enumerate(self.backbone.layers):
            normalized = layer.attention_norm(x)
            attended = mx.concatenate(
                [
                    layer.attention(normalized[row : row + 1], ctx, caches[row][index])
                    for row, ctx in enumerate(contexts)
                ]
            )
            x = x + attended
            x = x + layer.feed_forward(layer.ffn_norm(x))
        return self._sample_block(x, anchor, count, sampler, token_dtype)

    def _sample_block(self, x, anchor, count, sampler, token_dtype):
        x = self.backbone.final_norm(x)
        logits = self._lm_head(x)
        cap = self.config.text_config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        state, tokens, previous = None, [], anchor
        for i in range(count):
            bias, state = self.autoregressive_head.step(previous, x[:, i], state)
            # Only the base logits are soft-capped, never the Markov correction.
            previous = sampler(logits[:, i] + bias).reshape(-1)
            tokens.append(previous[:, None])
        return mx.concatenate(tokens, axis=1).astype(token_dtype)

    def sanitize(self, weights):
        return {k.removeprefix("mtp_dspark."): v for k, v in weights.items()}
