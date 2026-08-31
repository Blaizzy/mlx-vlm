import logging
import re

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
from .config import ModelConfig
from .fp8 import convert_qwen4_exp_fp8_weights
from .language import LanguageModel
from .vision import VisionModel

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")

ZERO_CENTERED_NORM_MODULES = frozenset(
    {
        "hc_norm",
        "k_layernorm",
        "k_norm",
        "norm_conv",
        "norm_key",
        "norm_query",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "q_layernorm",
        "q_norm",
    }
)

_PRESHIFTED_GAIN_THRESHOLD = 0.5


def zero_centered_norm_keys(weights):
    """Keys whose values Qwen4ExpRMSNorm reads as ``1 + w``.

    These are the gains the checkpoint stores centered at zero, matching
    upstream ``Qwen4ExpTextRMSNorm``. ``norm`` is excluded on purpose: it belongs
    to ``Qwen4ExpRMSNormGated``, whose gains are already centered at one and must
    be left alone. Matching is on the owning module's name rather than a dotted
    suffix so that the MTP head's unprefixed ``pre_fc_norm_*`` keys also match.
    """
    keys = []
    for key, value in weights.items():
        parts = key.split(".")
        if (
            len(parts) >= 2
            and parts[-1] == "weight"
            and parts[-2] in ZERO_CENTERED_NORM_MODULES
            and getattr(value, "ndim", 0) == 1
        ):
            keys.append(key)
    return keys


def norm_weights_are_preshifted(weights, keys=None):
    """Whether a checkpoint already folded Qwen4ExpRMSNorm's ``+1`` into its gains.

    Conversions produced before #2032 applied the offset at convert time, so
    applying it again at load time doubles it and inflates every gain (~2.6x on
    the hyper-connection norms), which degenerates generation into noise. Such a
    checkpoint carries the same keys, shapes and dtypes as a correct one, so only
    the values tell them apart: the released gains average ~0.15 and pre-shifted
    ones ~1.15, so the two are separated by the midpoint with a wide margin
    either side. Per-tensor sign is not usable -- a quarter of the pre-shifted
    tensors still contain negative values.
    """
    keys = zero_centered_norm_keys(weights) if keys is None else keys
    if not keys:
        return False
    total = sum(weights[key].size for key in keys)
    sums = mx.stack([mx.sum(weights[key].astype(mx.float32)) for key in keys])
    return mx.sum(sums).item() / total > _PRESHIFTED_GAIN_THRESHOLD


def unshift_preshifted_norm_weights(weights, preshifted=None):
    """Subtract a ``+1`` that a checkpoint already folded into its norm gains.

    ``preshifted`` overrides the value-based detection when it is not ``None``.
    Correcting a checkpoint moves it into the untouched class, so running this
    over its own output is a no-op and ``sanitize`` stays idempotent.

    Recovery is only as good as the precision the sum was stored at: a bf16
    checkpoint rounds ``w + 1`` to the coarser exponent around one, so gains
    within an ulp of zero come back as zero. The error is bounded by one ulp of
    the stored value, which re-converting from the released weights avoids.
    """
    keys = zero_centered_norm_keys(weights)
    if not keys:
        return weights
    if preshifted is None:
        preshifted = norm_weights_are_preshifted(weights, keys)
    if not preshifted:
        return weights

    logging.warning(
        "qwen4_exp: this checkpoint's RMSNorm gains are centered at one, so "
        "Qwen4ExpRMSNorm's +1 offset is already folded in -- the signature of a "
        "conversion made before #2032. Subtracting it back out of %d tensors; "
        "re-convert from Qwen/Qwen3.8-Flash-Next to avoid relying on this "
        "fixup, or set text_config.preshifted_norm_weights to override the "
        "detection.",
        len(keys),
    )
    for key in keys:
        weights[key] = weights[key] - 1.0
    return weights


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # The MTP predictor is a separate speculative artifact and is not part
        # of the base conditional-generation model.
        weights = {
            key: value for key, value in weights.items() if not key.startswith("mtp.")
        }
        weights = convert_qwen4_exp_fp8_weights(weights)
        if self.config.text_config.ple_storage:
            weights = {
                key: value
                for key, value in weights.items()
                if ".ple.ple_embedding.ngram_embedding." not in key
            }

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        weights = unshift_preshifted_norm_weights(
            weights, self.config.text_config.preshifted_norm_weights
        )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                midpoint = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :midpoint, :
                ]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., midpoint:, :
                ]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )

        sanitized = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
