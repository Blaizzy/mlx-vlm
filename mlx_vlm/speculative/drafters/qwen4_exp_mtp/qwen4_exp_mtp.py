from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.qwen3_5.language import _create_qwen3_5_attention_mask
from ....models.qwen4_exp.language import (
    QSAKVCache,
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
)
from ..deepseek_v4_mtp.deepseek_v4_mtp import DeepseekV4MTPDraftModel
from .config import Qwen4ExpMTPConfig


class Qwen4ExpMTPDraftModel(DeepseekV4MTPDraftModel):
    """Standalone runtime for Qwen4's native hyper-connection MTP head.

    The draft lifecycle is shared with the DeepSeek-V4 hyper-connection head,
    while input fusion and the decoder block follow Qwen4's released tensors.
    """

    supports_greedy_draft_argmax = True
    # filter_batch keeps owned KV and per-row speculative state aligned.
    supports_continuous_batching = True
    # Unlike shared-KV-only assistants, this drafter owns speculative KV and
    # seed state. External serving loops must reconcile every verified round
    # and apply the same row filters as the target batch.
    requires_verified_token_reconciliation = True
    # A caller-provided block size is an adaptive ceiling. Longer
    # autoregressive tails are useful only after the native one-token prefix
    # has demonstrated enough acceptance to amortize them.
    prefer_requested_block_size = False
    requires_uniform_batch_acceptance = True

    def __init__(self, config: Qwen4ExpMTPConfig):
        nn.Module.__init__(self)
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Qwen4ExpMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        hc_hidden_size = text_config.hc_count * hidden_size
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            hidden_size, eps=text_config.rms_norm_eps
        )
        # The released head applies one global RMS normalization across all
        # hyper-connection streams before projecting them independently.
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            hc_hidden_size, eps=text_config.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            layer_types=["qwen_sparse_attention"],
            full_attention_interval=1,
            ple_layer_ids=[],
        )
        self.layers = [Qwen4ExpDecoderLayer(layer_config, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            layer_config, use_combine=False
        )

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[QSAKVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position = 0
        self._round_appended = 0
        self._kv_valid_len = 0
        self._position = 0
        self._draft_round = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "args", None)
        model_type = getattr(args, "model_type", "")
        if not str(model_type).startswith("qwen4_exp"):
            raise ValueError(
                "Qwen4-Exp MTP requires a Qwen4-Exp target model, got "
                f"model_type={model_type!r}."
            )
        if getattr(args, "hc_count", None) != self.args.hc_count:
            raise ValueError("Qwen4-Exp target and MTP hc_count do not match.")

    def make_cache(self) -> List[QSAKVCache]:
        return [QSAKVCache() for _ in self.layers]

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        expected = self.args.hc_count * self.args.hidden_size
        if hidden.ndim == 4:
            hidden = hidden.reshape(*hidden.shape[:-2], expected)
        if hidden.ndim != 3 or hidden.shape[-1] != expected:
            raise ValueError(
                "Qwen4-Exp MTP expects target hidden shape "
                "[batch, tokens, hc_count * hidden_size]."
            )
        return hidden

    def fuse_inputs(
        self,
        token_embed: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        hidden = self._target_hidden(hidden)
        projected_embedding = self.fc_embedding(self.pre_fc_norm_embedding(token_embed))
        hidden_streams = self.pre_fc_norm_hidden(hidden).reshape(
            *hidden.shape[:-1], self.args.hc_count, self.args.hidden_size
        )
        projected_hidden = self.fc_hidden(hidden_streams)
        return (projected_embedding[..., None, :] + projected_hidden).reshape(
            hidden.shape
        )

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[QSAKVCache]],
    ) -> Tuple[mx.array, mx.array]:
        hidden = self.fuse_inputs(token_embed, hidden)
        if cache is None:
            cache = [None] * len(self.layers)
        position_ids = self._position_ids(length=tokens.shape[1])
        mask = _create_qwen3_5_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(
                hidden,
                tokens,
                mask=mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
        return self.hyper_connection_mixer(hidden), hidden

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]
        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = dict(weights)
        stripped = {}
        for key, value in weights.items():
            for prefix in ("language_model.mtp.", "model.mtp.", "mtp."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            stripped[key] = value

        gate_up_key = "layers.0.mlp.experts.gate_up_proj"
        down_key = "layers.0.mlp.experts.down_proj"
        if gate_up_key in stripped:
            gate_up = stripped.pop(gate_up_key)
            gate, up = mx.split(gate_up, 2, axis=-2)
            stripped["layers.0.mlp.switch_mlp.gate_proj.weight"] = gate
            stripped["layers.0.mlp.switch_mlp.up_proj.weight"] = up
        if down_key in stripped:
            stripped["layers.0.mlp.switch_mlp.down_proj.weight"] = stripped.pop(
                down_key
            )
        return stripped
