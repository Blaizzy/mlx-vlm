"""MLX port of the DFlash block-diffusion drafter.

Reference: huggingface.co/z-lab/Qwen3.5-4B-DFlash (dflash.py).

The drafter is a stateful head that runs on top of a target Qwen3.5 model:
    * ``inputs`` — ``[B, S, V]`` token ids for the noise block of length
       ``block_size`` (first slot = previous bonus token ``b``, remaining
       ``block_size - 1`` slots = mask tokens). The drafter embeds them via
       the target model's tied ``embed_tokens``.
    * ``target_hidden`` — ``[B, T_new, num_target_layers * H]`` concatenated
       hidden states from the target's selected ``target_layer_ids`` for the
       newly committed positions since the previous drafter call.
    * ``cache`` — list of ``mlx_lm.models.cache.KVCache`` (one per drafter
       layer) that absorbs both the new target features and the noise K/V.

The drafter returns logits ``[B, block_size, V]`` via the target's tied
``lm_head``/``embed_tokens.as_linear``. Call ``trim_prompt_cache`` on the
drafter cache after each round to drop the transient noise slots, leaving
only committed target features behind.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import MLP as Qwen3MLP

from .config import DFlashConfig


class DFlashAttention(nn.Module):
    """Cross-attention where Q comes from the noise block and K/V come from
    the concatenation of new target context + noise. Non-causal. Uses the
    offset-aware ``nn.RoPE`` so RoPE positions are automatically consistent
    with the drafter's stateful ``KVCache``.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,          # [B, L, H]   noise block hidden states
        x_ctx: mx.array,      # [B, S, H]   new fused target context features
        rope: nn.RoPE,
        cache: KVCache,
    ) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        c = mx.concatenate([x_ctx, x], axis=1)  # [B, S+L, H]
        q = self.q_proj(x)
        k = self.k_proj(c)
        v = self.v_proj(c)
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(k.reshape(B, S + L, self.n_kv_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, S + L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Q sits right after the existing cache + new target context.
        q = rope(q, offset=cache.offset + S)
        # K starts at cache.offset (new target features, then noise).
        k = rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """Stateful block-diffusion drafter. See module docstring."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)
        # Filled in by ``bind`` — the drafter uses the target's tied
        # embed_tokens for both input embedding and output projection.
        self.embed_tokens = None
        self.lm_head = None
        #: Per-round accepted drafted-token counts for the current
        #: generation. Reset by :meth:`reset`; read by callers after
        #: generation to report mean acceptance.
        self.accept_lens: List[int] = []

    def bind(self, target_model) -> "DFlashDraftModel":
        """Attach the target model's ``embed_tokens`` and ``lm_head``.

        Handles three common layouts:
            * plain mlx_lm Qwen3 (``target_model.model.embed_tokens``)
            * mlx_lm hybrid (``target_model.model.embed_tokens``)
            * mlx_vlm VLM (``target_model.language_model.model.embed_tokens``)
        """
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
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def reset(self, target_model) -> List[KVCache]:
        """Prepare for a fresh generation.

        Binds the drafter to the target's tied ``embed_tokens`` /
        ``lm_head``, clears per-round acceptance stats, and returns a
        fresh drafter cache list.
        """
        self.bind(target_model)
        self.accept_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        """Run one drafter round.

        Builds the noise block ``[bonus, mask, mask, …]`` of length
        ``block_size``, runs the drafter forward, trims the transient
        noise K/V tail off ``cache``, and samples the drafted tokens.

        ``last_bonus`` may be a scalar ``int`` (B=1) or an
        ``mx.array`` of shape ``[B]`` for batch speculative decoding.
        Returns shape ``(B, block_size - 1)``.
        """
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_logits = self(block, hidden, cache)
        for c in cache:
            c.trim(block_size)
        return sampler(draft_logits[:, 1 - block_size :])

    def __call__(
        self,
        inputs: mx.array,           # [B, block_size] token ids
        target_hidden: mx.array,    # [B, T_new, num_target_layers * H]
        cache: List[KVCache],
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.lm_head(self.norm(h))

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model."):]
            out[k] = v
        return out


# Backwards-compat alias (earlier module versions exposed a ``DFlashKVCache``
# shim; we now use mlx_lm.KVCache directly).
DFlashKVCache = KVCache
