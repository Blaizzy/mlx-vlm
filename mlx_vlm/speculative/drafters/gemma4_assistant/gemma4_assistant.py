from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import RMSNorm

from ....models.gemma4.config import TextConfig
from ....models.gemma4.language import DecoderLayer
from .config import Gemma4AssistantConfig
from .masks import make_drafter_masks


class _DraftInner(nn.Module):
    """Minimal Gemma-4 text model holding the drafter's per-layer modules.

    Mirrors ``mlx_vlm.models.gemma4.language.Gemma4TextModel`` weight-key
    shape (``embed_tokens``, ``layers.{i}.*``, ``norm``) so the HF drafter
    checkpoint loads as-is. Skips the target's KV-sharing/per-layer-input
    bookkeeping — the drafter has its own forward in
    ``Gemma4AssistantDraftModel.__call__`` and consumes ``shared_kv_states``
    coming from the target rather than computing or sharing K/V internally.
    """

    def __init__(self, text_config: TextConfig):
        super().__init__()
        self.config = text_config
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = [
            DecoderLayer(text_config, layer_idx=i, kv_shared_only=True)
            for i in range(text_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)


class Gemma4AssistantDraftModel(nn.Module):
    """Gemma 4 Multi-Token Prediction drafter.

    Surface mirrors ``DFlashDraftModel``: ``bind`` / ``make_cache`` /
    ``reset`` / ``draft_block`` / ``__call__`` / ``sanitize`` / ``accept_lens``.
    The drafter has no KV cache of its own — its recurrent state is the
    ``post_projection``-projected hidden carried across draft steps.

    ``draft_block``'s ``cache`` argument is repurposed to thread the
    target's ``shared_kv_states`` dict into the drafter; the round-loop
    in ``mlx_vlm/generate.py`` sets this via ``reset(target, shared_kv)``
    before each block.
    """

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__()
        self.config = config
        text_cfg = config.text_config
        if text_cfg is None:
            raise ValueError("Gemma4AssistantConfig.text_config must be set")

        self.model = _DraftInner(text_cfg)
        self.pre_projection = nn.Linear(
            2 * config.backbone_hidden_size, text_cfg.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(
            text_cfg.hidden_size, config.backbone_hidden_size, bias=False
        )
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                text_cfg.hidden_size, text_cfg.vocab_size, bias=False
            )

        # Bound at runtime via bind().
        self._lm_head_fn = None
        self._input_embed = None  # target's embed_tokens (backbone_hidden_size)
        self._input_embed_scale: float = 1.0
        # Threaded by reset() / round-loop before each draft_block.
        self._shared_kv: Optional[dict] = None
        self._kv_offset: int = 0
        self._position: int = 0

        self.accept_lens: List[int] = []

        if config.use_ordered_embeddings:
            raise NotImplementedError(
                "Gemma4Assistant `use_ordered_embeddings=True` (centroid LM "
                "head) is not yet ported to mlx-vlm."
            )

    # ------------------------------------------------------------------
    # binding / cache management
    # ------------------------------------------------------------------
    def bind(self, target_model) -> "Gemma4AssistantDraftModel":
        """Wire input embeddings (target's, backbone_hidden_size) and lm_head
        (drafter's tied, hidden_size).

        Per the HF ``SinglePositionMultiTokenCandidateGenerator`` the input
        pathway uses the *target*'s embed table (because ``inputs_embeds``
        feeds into ``pre_projection`` which expects ``2 *
        backbone_hidden_size``), while the LM head uses the drafter's own
        tied embedding.
        """
        # LM head — drafter's own tied embed.
        if self.config.tie_word_embeddings:
            self._lm_head_fn = self.model.embed_tokens.as_linear
        else:
            self._lm_head_fn = self.lm_head

        # Input embeddings — target's. Walk the standard wrappers.
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
        if inner is not None:
            self._input_embed = inner.embed_tokens
            # HF's target uses ``Gemma4TextScaledWordEmbedding`` which
            # multiplies the lookup by ``sqrt(hidden_size)`` *inside* the
            # module. In MLX-VLM the same scale is applied externally
            # (``language.py: h = self.embed_tokens(inputs) * self.embed_scale``),
            # so the bare ``embed_tokens`` here returns *unscaled* vectors.
            # The drafter's ``pre_projection`` was trained against scaled
            # target embeddings, so we replicate the scale at call time.
            self._input_embed_scale = float(
                getattr(inner, "embed_scale", 1.0)
            )

        # Stash the target's layer_types so the round-loop knows which keys
        # the drafter expects in shared_kv_states.
        try:
            tcfg = getattr(target_model, "language_model", target_model)
            tcfg = getattr(tcfg, "config", None)
            if tcfg is not None:
                self.config.target_layer_types = list(tcfg.layer_types)
        except Exception:
            pass
        return self

    def make_cache(self) -> List:
        """Drafter has no cache of its own."""
        return []

    def reset(self, target_model) -> List:
        self.bind(target_model)
        self.accept_lens = []
        self._shared_kv = None
        self._kv_offset = 0
        return self.make_cache()

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset: int,
        position: Optional[int] = None,
    ) -> None:
        """Threaded by the round-loop before each ``draft_block`` call.

        ``kv_offset`` is the length of the target's KV cache (= number of
        target-side tokens already written to KV; the just-sampled bonus
        is *not* yet in the cache). ``position`` is the absolute position
        id used for RoPE on the drafter's queries; HF
        ``SinglePositionMultiTokenCandidateGenerator`` locks it to
        ``input_ids.shape[1] - 1`` which equals the bonus's position —
        i.e. ``kv_offset`` (one past the last cached token).
        """
        self._shared_kv = shared_kv_states
        self._kv_offset = int(kv_offset)
        self._position = int(position) if position is not None else int(kv_offset)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: dict,
        position_ids: mx.array,
        cache: Any = None,
    ) -> Tuple[mx.array, mx.array]:
        del cache  # drafter has no own KV cache
        text_cfg = self.config.text_config

        h = self.pre_projection(inputs_embeds)

        query_len = h.shape[1]
        query_offset = int(position_ids[0, 0].item())
        masks = make_drafter_masks(
            shared_kv_states,
            query_len=query_len,
            query_offset=query_offset,
            sliding_window=text_cfg.sliding_window,
            dtype=h.dtype,
        )

        # Drafter queries always live one slot past the end of the KV; the
        # absolute position is what RoPE needs.
        offset = mx.array(query_offset)

        for layer in self.model.layers:
            kv = shared_kv_states[layer.layer_type]
            mask = masks[layer.layer_type]
            h, _, _ = layer(
                h,
                mask=mask,
                cache=None,
                per_layer_input=None,
                shared_kv=kv,
                offset=offset,
            )

        h = self.model.norm(h)
        last_hidden = self.post_projection(h)
        logits = self._lm_head_fn(h) if self._lm_head_fn is not None else (
            self.model.embed_tokens.as_linear(h)
        )
        return last_hidden, logits

    # ------------------------------------------------------------------
    # block drafting
    # ------------------------------------------------------------------
    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        """Autoregressive K-step drafting.

        Returns an ``[B, block_size - 1]`` token tensor — drop-in
        compatible with ``_speculative_walk(_batch)`` in ``generate.py``.
        ``cache`` is unused (drafter has no own cache); ``shared_kv`` is
        threaded via ``set_shared_kv`` before this call.

        Mirrors HF ``SinglePositionMultiTokenCandidateGenerator``: input
        is ``[target_embed(last_token), last_hidden_state]`` (in that
        order), ``position_ids`` is held constant across all draft steps.
        """
        del cache
        if self._shared_kv is None:
            raise RuntimeError(
                "set_shared_kv() must be called before draft_block(). "
                "The round-loop in generate.py is responsible for this."
            )
        if self._input_embed is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target's input embeddings."
            )
        shared_kv = self._shared_kv
        position_ids = mx.array([[self._position]])

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden  # [B, 1, backbone_hidden_size]
        tokens: List[mx.array] = []

        for _ in range(block_size - 1):
            tok_embed = self._input_embed(tok) * self._input_embed_scale
            inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
            h_prev, logits = self(inputs_embeds, shared_kv, position_ids)
            tok = sampler(logits)  # [B, 1]
            tokens.append(tok)

        return mx.concatenate(tokens, axis=1)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            # Drop any (rare) tied lm_head copy when we'll bind to the
            # embedding's as_linear at runtime.
            if k == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            out[k] = v
        return out
