import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ....models.base import create_attention_mask
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import HyperHead
from ....models.deepseek_v4.language import DeepseekV4Block
from ..mtp_base import AutoregressiveMTPDraftModel
from .config import DeepseekV4MTPConfig


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if "decoder.ffn.switch_mlp." in k and k.endswith("_proj")
    }
    mxfp8_modules = {
        k: mxfp8
        for k, _ in flat_modules
        if k in ("e_proj", "h_proj")
        or "decoder.ffn.shared_experts." in k
        or "decoder.attn.w" in k
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **mxfp8_modules,
    }


class DeepseekV4MTPDraftModel(AutoregressiveMTPDraftModel):
    def __init__(self, config: DeepseekV4MTPConfig):
        super().__init__(config)
        text_config = config.text_config
        if text_config is None:
            raise ValueError("DeepseekV4MTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            compress_ratios=[0],
            num_hash_layers=0,
        )
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder = DeepseekV4Block(layer_config, layer_idx=0)
        self.hc_head = HyperHead(text_config)
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

    @property
    def quant_predicate(self):
        quantization_config = make_quantization_config(self)

        def predicate(path, _):
            return quantization_config.get(path, True)

        return predicate

    def make_cache(self) -> List[RotatingKVCache]:
        return [RotatingKVCache(max_size=self.args.sliding_window)]

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        if (
            hidden.ndim == 3
            and hidden.shape[-1] == self.args.hc_mult * self.args.hidden_size
        ):
            hidden = hidden.reshape(*hidden.shape[:-1], self.args.hc_mult, -1)
        if hidden.ndim != 4:
            raise ValueError(
                "DeepSeek-V4 MTP expects target hidden shape "
                "[batch, tokens, hc_mult, hidden_size]."
            )
        return hidden

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[RotatingKVCache]],
    ) -> Tuple[mx.array, mx.array]:
        hidden = self._target_hidden(hidden)
        B, L, H, D = hidden.shape
        h_flat = hidden.reshape(B * L * H, D)
        h_proj = self.h_proj(self.hnorm(h_flat)).reshape(B, L, H, D)
        e_proj = self.e_proj(self.enorm(token_embed))[:, :, None, :]
        h = e_proj + h_proj

        if cache is None:
            cache = [None]
        mask = create_attention_mask(
            h[:, :, 0, :],
            cache[0],
            window_size=self.args.sliding_window,
            return_array=True,
        )
        h = self.decoder(
            h,
            mask,
            cache[0],
            tokens,
            position_offset=self._next_position,
        )
        logits_hidden = self.norm(self.hc_head(h))
        return logits_hidden, h

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = dict(weights)
        new_weights = {}
        for k, v in weights.items():
            # The single MTP block lives under `mtp.<layer_idx>.` (index 1 for
            # DeepSeek-V4-Flash, 0 for others) or bare `mtp.` — strip any of them
            # so the block's tensors land at the drafter's top level.
            m = re.match(r"mtp\.(?:\d+\.)?", k)
            if m:
                k = k[m.end() :]
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ("ffn.experts." in wk or ".ffn.experts." in wk)
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            if nk.startswith("attn.") or nk.startswith("attn_norm."):
                nk = f"decoder.{nk}"
            elif nk.startswith("ffn.") or nk.startswith("ffn_norm."):
                nk = f"decoder.{nk}"
            if nk.endswith(".ffn.gate.bias"):
                nk = nk[: -len(".ffn.gate.bias")] + ".ffn.gate.e_score_correction_bias"
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
                    nk = nk.replace(f"hc_{sub}_{param}", f"decoder.{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            nk = nk.replace("hc_head_fn", "hc_head.fn")
            nk = nk.replace("hc_head_base", "hc_head.base")
            nk = nk.replace("hc_head_scale", "hc_head.scale")
            remapped[nk] = v
        weights = remapped

        prefix = "decoder.ffn.experts"
        for src, dst in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            for suffix in ("weight", "scales"):
                key0 = f"{prefix}.0.{src}.{suffix}"
                if key0 in weights:
                    stacked = [
                        weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"decoder.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                        stacked
                    )

        prefix = "decoder.attn.wo_a"
        for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
            if key in weights and weights[key].ndim == 2:
                weights[key] = weights[key].reshape(
                    self.args.o_groups, self.args.o_lora_rank, -1
                )

        return weights
