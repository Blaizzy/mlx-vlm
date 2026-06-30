"""DSpark self-speculative drafter for a DeepSeek-V4 target.

Ports the reference ``inference/model.py`` draft stack (``Transformer.forward_spec`` +
``DSparkBlock``) into mlx-vlm, reusing the in-tree DeepSeek-V4 primitives that are
numerically identical to the reference:

* :class:`DeepseekV4MoE` — the hash/score MoE (draft blocks sit at ``layer_id >=
  num_hash_layers`` so they score-route); exact-match verified vs the reference MoE.
* :class:`HyperConnection` / :class:`HyperHead` / :func:`hc_expand` — the Hyper-Connection
  pre/post/head mixing; match the reference ``hc_pre``/``hc_post``/``hc_head`` to ~1e-6.
* :class:`DSparkMarkovHead` / :class:`DSparkConfidenceHead` — the low-rank Markov logit bias
  and (bias-free) confidence projection.

Only the windowed MLA cross-attention (``DSparkAttention``) and the RoPE/RMSNorm helpers are
ported fresh (no in-tree equivalent). The drafter exposes two drivers over the same weights:

* ``forward_spec`` / ``advance`` — the reference windowed API (parity oracle target).
* ``draft_block`` / ``reset`` / ``make_cache`` — the mlx-vlm speculative round-loop contract
  (``speculative/dspark.py``). Losslessness comes from the target verify+walk, so the eager
  ``draft_block`` conditioning need not match ``forward_spec`` bit-for-bit.
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.deepseek_v4.hyper_connection import (
    HyperConnection,
    HyperHead,
    hc_expand,
)
from ....models.deepseek_v4.language import DeepseekV4MoE
from .attention import DSparkAttention, DSparkKVCache, RMSNorm
from .config import DeepseekV4DSparkConfig
from .heads import DSparkConfidenceHead, DSparkMarkovHead


def _sample(logits: mx.array, temperature: float) -> mx.array:
    """argmax at temperature 0, else Gumbel-max (matches the reference sampler)."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    probs = mx.softmax(logits.astype(mx.float32) / max(temperature, 1e-5), axis=-1)
    g = mx.random.uniform(shape=probs.shape).astype(mx.float32)
    return mx.argmax(probs / (-mx.log(g + 1e-20)), axis=-1).astype(mx.int32)


class _DSparkWindowCache:
    """Per-sequence draft state: one sliding window per block + the committed offset."""

    def __init__(self, n_blocks: int, window_size: int):
        self.windows = [DSparkKVCache(window_size) for _ in range(n_blocks)]
        self.offset = 0


class DeepseekV4DSparkBlock(nn.Module):
    """One DSpark draft stage: HC(attn) + HC(MoE), with windowed cross-attention.

    Mirrors the reference ``DSparkBlock``/``Block``. Reuses ``DeepseekV4MoE`` for the FFN and
    ``HyperConnection``/``HyperHead`` for the residual mixing; only ``DSparkAttention`` is the
    ported windowed MLA. Stage 0 carries ``main_proj``/``main_norm`` (projects the captured
    target hiddens); the last stage carries the LM-head norm + Markov/confidence heads.
    """

    def __init__(
        self, config: DeepseekV4DSparkConfig, stage_id: int, max_seq_len: int = 8192
    ):
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.is_last = stage_id == config.n_mtp_layers - 1
        self.block_size = config.block_size
        self.noise_token_id = config.noise_token_id
        self.hc_mult = config.hc_mult
        self.temperature = getattr(config, "temperature", 0.0)
        layer_id = (
            config.num_hidden_layers + stage_id
        )  # >= num_hash_layers → score routing

        self.attn = DSparkAttention(config, max_seq_len)
        self.ffn = DeepseekV4MoE(config, layer_id)
        # The official shared expert applies the SwiGLU clamp (reference Expert with
        # swiglu_limit); DeepseekV4MoE leaves it at the no-clamp default, so set it here.
        self.ffn.shared_experts.swiglu_limit = config.swiglu_limit
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

        if stage_id == 0:
            self.main_proj = nn.Linear(config.fc_in, config.hidden_size, bias=False)
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config.vocab_size, config.markov_rank)
            self.confidence_head = DSparkConfidenceHead(
                config.confidence_in, bias=False
            )
            self.hc_head = HyperHead(config)

    def _hc_block(self, x: mx.array, attn_call, input_ids: mx.array) -> mx.array:
        residual = x
        h, post, comb = self.attn_hc(x)
        h = attn_call(self.attn_norm(h))
        x = hc_expand(h, residual, post, comb)

        residual = x
        h, post, comb = self.ffn_hc(x)
        h = self.ffn(self.ffn_norm(h), input_ids)
        return hc_expand(h, residual, post, comb)

    def __call__(
        self,
        x: mx.array,
        start_pos: int,
        input_ids: mx.array,
        main_x: mx.array,
        window: DSparkKVCache,
    ) -> mx.array:
        if start_pos == 0:
            return self.attn(x, 0, main_x, window)  # prefill: seed window only
        return self._hc_block(
            x, lambda h: self.attn(h, start_pos, main_x, window), input_ids
        )

    def forward_draft(
        self,
        x: mx.array,
        block_start: int,
        win_valid: int,
        input_ids: mx.array,
        window: DSparkKVCache,
    ) -> mx.array:
        return self._hc_block(
            x, lambda h: self.attn.draft(h, block_start, win_valid, window), input_ids
        )

    def advance(self, main_x: mx.array, position: int, window: DSparkKVCache) -> None:
        self.attn.advance_window(main_x, position, window)

    def forward_embed(
        self, main_hidden: mx.array, input_ids: mx.array, embed: nn.Module
    ) -> Tuple[mx.array, mx.array]:
        """Stage-0 only: project the main hidden + embed the draft block (anchor + noise)."""
        main_x = self.main_norm(self.main_proj(main_hidden))
        b = input_ids.shape[0]
        anchor = input_ids.reshape(b, 1).astype(mx.int32)
        noise = mx.full((b, self.block_size - 1), self.noise_token_id, dtype=mx.int32)
        draft_ids = mx.concatenate([anchor, noise], axis=1)
        x = embed(draft_ids)
        x = mx.broadcast_to(
            x[:, :, None, :], (b, self.block_size, self.hc_mult, x.shape[-1])
        )
        return mx.contiguous(x), main_x

    def forward_head(
        self, x: mx.array, input_ids: mx.array, head: nn.Module
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Last-stage only: draft tokens + (Markov-biased) logits + confidence."""
        x = self.hc_head(x)
        logits = head(self.norm(x).astype(mx.float32))  # [b, block_size, vocab]
        prev = input_ids.astype(mx.int32)
        out_ids, biased, markov_embeds = [prev], [], []
        for i in range(self.block_size):
            bias, membed = self.markov_head(prev)
            li = logits[:, i] + bias
            biased.append(li)
            markov_embeds.append(membed)
            prev = _sample(li, self.temperature)
            out_ids.append(prev)
        output_ids = mx.stack(out_ids, axis=1)
        logits_out = mx.stack(biased, axis=1)
        markov_embed = mx.stack(markov_embeds, axis=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits_out, confidence


class DeepseekV4DSparkDraftModel(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig, max_seq_len: int = 8192):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.blocks = [
            DeepseekV4DSparkBlock(config, i, max_seq_len)
            for i in range(config.n_mtp_layers)
        ]
        self._spec_cache = self.make_cache()
        self._last_confidence: Optional[mx.array] = None
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    # ---- reference windowed API (parity oracle target) ----------------------------

    def forward_spec(
        self,
        input_ids: mx.array,
        main_hidden: mx.array,
        start_pos: int = 0,
        cache: Optional[_DSparkWindowCache] = None,
    ) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
        """Prefill (start_pos==0) seeds the window and returns None; decode drafts a block."""
        cache = cache if cache is not None else self._spec_cache
        h, main_x = self.blocks[0].forward_embed(main_hidden, input_ids, self.embed)
        for blk, window in zip(self.blocks, cache.windows):
            h = blk(h, start_pos, input_ids, main_x, window)
        if start_pos == 0:
            return None
        return self.blocks[-1].forward_head(h, input_ids, self.head)

    def advance(
        self,
        main_hidden: mx.array,
        position: int,
        cache: Optional[_DSparkWindowCache] = None,
    ) -> None:
        """Slide every block's window over one committed token at ``position``."""
        cache = cache if cache is not None else self._spec_cache
        mh = main_hidden.reshape(main_hidden.shape[0], -1)
        b0 = self.blocks[0]
        main_x = b0.main_norm(b0.main_proj(mh))
        for blk, window in zip(self.blocks, cache.windows):
            blk.advance(main_x, position, window)

    # ---- mlx-vlm speculative round-loop contract ----------------------------------

    def make_cache(self) -> _DSparkWindowCache:
        return _DSparkWindowCache(len(self.blocks), self.config.sliding_window)

    def reset(self, target_model: Any = None) -> _DSparkWindowCache:
        # DSpark is standalone (own embed/head) — nothing to bind to the target.
        self.accept_lens = []
        self.draft_lens = []
        self._last_confidence = None
        return self.make_cache()

    def _extend_window(self, cache: _DSparkWindowCache, main_x: mx.array) -> None:
        """Fold the committed context (projected ``main_x`` [B, S, dim]) into each window."""
        S = main_x.shape[1]
        if cache.offset == 0:
            for blk, window in zip(self.blocks, cache.windows):
                blk.attn.seed_window(main_x, window)
        else:
            for pos in range(S):
                for blk, window in zip(self.blocks, cache.windows):
                    blk.advance(main_x[:, pos], cache.offset + pos, window)
        cache.offset += S

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: _DSparkWindowCache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        if isinstance(last_bonus, int):
            anchor = mx.array([last_bonus], dtype=token_dtype)
        else:
            anchor = last_bonus.reshape(-1).astype(token_dtype)
        B = hidden.shape[0]

        b0 = self.blocks[0]
        main_x = b0.main_norm(b0.main_proj(hidden))  # [B, S, dim]
        self._extend_window(cache, main_x)
        block_start = win_valid = cache.offset

        noise = mx.full(
            (B, block_size - 1), int(self.config.noise_token_id), dtype=token_dtype
        )
        draft_ids = mx.concatenate([anchor[:, None], noise], axis=1)  # [B, block_size]
        h = self.embed(draft_ids)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :], (B, block_size, self.config.hc_mult, h.shape[-1])
            )
        )
        for blk, window in zip(self.blocks, cache.windows):
            h = blk.forward_draft(h, block_start, win_valid, draft_ids, window)

        last = self.blocks[-1]
        x = last.hc_head(h)  # [B, block_size, dim]
        base_logits = self.head(last.norm(x).astype(mx.float32))  # [B, block_size, V]

        prev = anchor
        drafts, markov_embeds = [], []
        for i in range(block_size):
            bias, embed = last.markov_head(prev)
            prev = sampler(base_logits[:, i] + bias).astype(token_dtype)
            drafts.append(prev)
            markov_embeds.append(embed)
        draft_tokens = mx.stack(drafts, axis=1)  # [B, block_size]
        markov_embed = mx.stack(markov_embeds, axis=1)
        self._last_confidence = last.confidence_head(x, markov_embed)  # [B, block_size]
        return draft_tokens

    # ---- checkpoint loading --------------------------------------------------------

    def sanitize(self, weights: dict) -> dict:
        return _sanitize(weights, self.config)


def _sanitize(weights: dict, config: DeepseekV4DSparkConfig) -> dict:
    """Map a DeepSeek-V4-DSpark checkpoint (``mtp.N.*`` + ``embed``/``head``) to this
    drafter's param tree, reusing the target's MTP conventions:

    * ``mtp.N.*`` -> ``blocks.N.*``; base-model keys dropped.
    * hc renames ``hc_{attn,ffn}_{fn,base,scale}`` -> ``{attn,ffn}_hc.{...}``,
      ``hc_head_*`` -> ``hc_head.*``.
    * ``ffn.gate.bias`` -> ``ffn.gate.e_score_correction_bias``;
      ``ffn.shared_experts.{w1,w2,w3}`` -> ``{gate,down,up}_proj``;
      per-expert ``ffn.experts.{e}.{w1,w2,w3}`` -> stacked
      ``ffn.switch_mlp.{gate,down,up}_proj``.

    fp8/fp4 dequant of the bundled quantized weights is handled at load time
    (Phase 3 / real-weight path); ``.scale``/``.scales`` siblings pass through here.
    """
    w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    # 1. keep only embed/head/mtp.*; rename mtp.N -> blocks.N with the inline subs.
    remapped = {}
    for k, v in weights.items():
        if k in ("embed.weight", "head.weight"):
            remapped[k] = v
            continue
        if not k.startswith("mtp."):
            continue
        nk = "blocks." + k[len("mtp.") :]
        nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        for sub in ("attn", "ffn"):
            for param in ("fn", "base", "scale"):
                nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
        nk = nk.replace(".hc_head_fn", ".hc_head.fn")
        nk = nk.replace(".hc_head_base", ".hc_head.base")
        nk = nk.replace(".hc_head_scale", ".hc_head.scale")
        for old, new in w_remap.items():
            nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
        remapped[nk] = v
    weights = remapped

    # 2. stack per-expert routed weights into switch_mlp.{gate,down,up}_proj.
    for stage in range(config.n_mtp_layers):
        prefix = f"blocks.{stage}.ffn.experts"
        for src, dst in w_remap.items():
            for suffix in ("weight", "scales"):
                key0 = f"{prefix}.0.{src}.{suffix}"
                if key0 in weights:
                    stacked = [
                        weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                        for e in range(config.n_routed_experts)
                    ]
                    weights[f"blocks.{stage}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                        stacked
                    )

    return weights
