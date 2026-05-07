from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import RMSNorm

from ....models.gemma4.config import TextConfig
from ....models.gemma4.language import DecoderLayer
from .config import Gemma4AssistantConfig
from .masked_embedder import MaskedEmbedder
from .masks import make_drafter_masks, normalize_batched_shared_kv_states


class _DraftInner(nn.Module):

    def __init__(self, text_config: TextConfig):
        super().__init__()
        self.config = text_config
        self.embed_tokens = nn.Embedding(
            text_config.vocab_size, text_config.hidden_size
        )
        self.layers = [
            DecoderLayer(text_config, layer_idx=i, kv_shared_only=True)
            for i in range(text_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)


class Gemma4AssistantDraftModel(nn.Module):

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

        self._lm_head_fn = None
        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._shared_kv: Optional[dict] = None
        self._kv_offset: int = 0
        self._position: int = 0

        self.accept_lens: List[int] = []

        if config.use_ordered_embeddings:
            self.masked_embedding = MaskedEmbedder(config)
        else:
            self.masked_embedding = None

    def bind(self, target_model) -> "Gemma4AssistantDraftModel":

        if self.masked_embedding is not None:
            embed_w = self.model.embed_tokens.weight
            masked = self.masked_embedding
            self._lm_head_fn = lambda h: masked(h, embed_w)
        elif self.config.tie_word_embeddings:
            self._lm_head_fn = self.model.embed_tokens.as_linear
        else:
            self._lm_head_fn = self.lm_head

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
            self._input_embed_scale = float(getattr(inner, "embed_scale", 1.0))

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
        kv_offset,
        position=None,
        left_padding=None,
    ) -> None:
        if isinstance(kv_offset, int):
            self._kv_offset = kv_offset
        else:
            self._kv_offset = (
                int(mx.array(kv_offset).max().item())
                if not isinstance(kv_offset, mx.array)
                else int(kv_offset.max().item())
            )
        if position is None:
            position = kv_offset
        if left_padding is not None:
            shared_kv_states = normalize_batched_shared_kv_states(
                shared_kv_states,
                kv_valid_len=position,
                left_padding=left_padding,
            )
        self._shared_kv = shared_kv_states
        if isinstance(position, int):
            self._position = position
        elif isinstance(position, mx.array):
            self._position = position
        else:
            self._position = mx.array(position)

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: dict,
        position_ids: mx.array,
        cache: Any = None,
    ) -> Tuple[mx.array, mx.array]:
        del cache
        text_cfg = self.config.text_config

        h = self.pre_projection(inputs_embeds)

        query_len = h.shape[1]
        query_offset = (
            int(position_ids[0, 0].item())
            if position_ids.shape[0] == 1
            else position_ids[:, 0]
        )
        masks = make_drafter_masks(
            shared_kv_states,
            query_len=query_len,
            query_offset=query_offset,
            sliding_window=text_cfg.sliding_window,
            dtype=h.dtype,
        )

        if position_ids.shape[0] == 1:
            offset = mx.array(query_offset)
        else:
            offset = position_ids[:, 0]

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
        logits = (
            self._lm_head_fn(h)
            if self._lm_head_fn is not None
            else (self.model.embed_tokens.as_linear(h))
        )
        return last_hidden, logits

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
                "Gemma 4 assistant drafter requires the MTP round-loop, but "
                "no shared K/V was set before draft_block() — this typically "
                "means the DFlash round-loop ran instead. Pass "
                "--draft-kind mtp on the CLI (or MLX_VLM_DRAFT_KIND=mtp on "
                "the server). If you call load_drafter() directly, use "
                "kind='mtp' and pass draft_kind='mtp' through to "
                "generate_step()."
            )
        if self._input_embed is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target's input embeddings."
            )
        shared_kv = self._shared_kv
        if isinstance(self._position, int):
            position_ids = mx.array([[self._position]])
        else:
            position_ids = self._position[:, None]

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []

        for _ in range(block_size - 1):
            tok_embed = self._input_embed(tok) * self._input_embed_scale
            inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
            h_prev, logits = self(inputs_embeds, shared_kv, position_ids)
            tok = sampler(logits)
            tokens.append(tok)

        return mx.concatenate(tokens, axis=1)

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k == "masked_embedding.token_ordering":
                v = v.astype(mx.int32)

            if k == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            out[k] = v
        return out
