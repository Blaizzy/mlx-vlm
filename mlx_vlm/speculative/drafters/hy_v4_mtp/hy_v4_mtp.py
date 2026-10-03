from dataclasses import replace
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, CacheList
from ....models.deepseek_v32.language import DeepseekV32DecoderLayer
from ....models.hy_v4.cache import HyV4KVCache
from ....models.hy_v4.fused_switch_glu import FusedSwitchGLU
from ....models.hy_v4.language import HyV4Attention
from ....models.hy_v4.moe import HyV4MoE
from ..qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel
from .config import HyV4MTPConfig


class HyV4MTPDecoderLayer(DeepseekV32DecoderLayer):
    """Hy4's rank-3 native MTP block, which has no hyper-connection streams."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HyV4Attention(config, layer_idx)
        if hasattr(self.mlp, "switch_mlp"):
            self.mlp = HyV4MoE(config)
            self.mlp.switch_mlp = FusedSwitchGLU(
                config.hidden_size,
                config.moe_intermediate_size,
                config.n_routed_experts,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ) -> mx.array:
        r, _ = self.self_attn(self.input_layernorm(x), mask, cache, prev_topk_indices)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class HyV4MTPDraftModel(Qwen3_5MTPDraftModel):
    """Standalone runtime for Hy4's single native next-token prediction layer."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True
    supports_ragged_batch_acceptance = False

    def __init__(self, config: HyV4MTPConfig):
        nn.Module.__init__(self)
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("HyV4MTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            indexer_types=["full"],
        )
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.decoder = HyV4MTPDecoderLayer(layer_config, layer_idx=0)
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

        self._input_embed = None
        self._input_embed_scale = 1.0
        self._lm_head_fn = None
        self._cache: List[CacheList] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position: Any = 0
        self._round_appended = 0
        self._kv_valid_len: Any = 0
        self._position: Any = 0
        self._draft_round = 0
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("decoder.mlp.gate")

        return predicate

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        target_args = getattr(target, "args", None)
        model_type = getattr(target_args, "model_type", "")
        if model_type != "hy_v4":
            raise ValueError(
                "Hy4 MTP requires a Hy4 target model, "
                f"got model_type={model_type!r}."
            )
        for field in (
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "qk_nope_head_dim",
            "v_head_dim",
            "kv_lora_rank",
            "n_routed_experts",
            "moe_intermediate_size",
        ):
            if getattr(target_args, field, None) != getattr(self.args, field, None):
                raise ValueError(f"Hy4 target and MTP {field} do not match.")

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[CacheList]:
        cache_cls = BatchKVCache if left_padding is not None else HyV4KVCache
        if left_padding is None:
            return [CacheList(cache_cls(), cache_cls())]
        return [CacheList(cache_cls(left_padding), cache_cls(left_padding))]

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[CacheList]],
        position_ids: mx.array,
    ) -> mx.array:
        del position_ids
        h = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )
        if cache is None:
            cache = [None]
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] is not None else None, return_array=True
        )
        return self.norm(self.decoder(h, mask=mask, cache=cache[0]))

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = dict(weights)

        def pack_mxfp8(weight: mx.array) -> mx.array:
            if weight.dtype != mx.uint8:
                weight = weight.view(mx.uint8)
            return weight.view(mx.uint32)

        def unpack_mxfp8(weight: mx.array, scales: mx.array) -> mx.array:
            return mx.dequantize(
                pack_mxfp8(weight),
                scales,
                None,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )

        def repack_mxfp8(prefix: str, weight: mx.array) -> None:
            packed, scales = mx.quantize(weight, group_size=32, bits=8, mode="mxfp8")
            weights[f"{prefix}.weight"] = packed
            weights[f"{prefix}.scales"] = scales

        mlp_prefix = "decoder.mlp"
        fused_weight_key = f"{mlp_prefix}.experts.gate_up_proj"
        fused_scale_key = f"{mlp_prefix}.experts.gate_up_proj_scale"
        if fused_weight_key in weights:
            gate_up = weights.pop(fused_weight_key)
            gate_up_scale = weights.pop(fused_scale_key)
            weights[f"{mlp_prefix}.switch_mlp.gate_up_proj.weight"] = pack_mxfp8(
                gate_up
            )
            weights[f"{mlp_prefix}.switch_mlp.gate_up_proj.scales"] = gate_up_scale

        switch_prefix = f"{mlp_prefix}.switch_mlp"
        for suffix in ("weight", "scales", "biases"):
            gate_key = f"{switch_prefix}.gate_proj.{suffix}"
            up_key = f"{switch_prefix}.up_proj.{suffix}"
            if gate_key in weights and up_key in weights:
                weights[f"{switch_prefix}.gate_up_proj.{suffix}"] = mx.concatenate(
                    [weights.pop(gate_key), weights.pop(up_key)], axis=1
                )

        down_weight_key = f"{mlp_prefix}.experts.down_proj"
        down_scale_key = f"{mlp_prefix}.experts.down_proj_scale"
        if down_weight_key in weights:
            weights[f"{mlp_prefix}.switch_mlp.down_proj.weight"] = pack_mxfp8(
                weights.pop(down_weight_key)
            )
            weights[f"{mlp_prefix}.switch_mlp.down_proj.scales"] = weights.pop(
                down_scale_key
            )

        attn_prefix = "decoder.self_attn"
        kv_weight_key = f"{attn_prefix}.kv_b_proj.weight"
        kv_scale_key = f"{attn_prefix}.kv_b_proj.weight_scale"
        if kv_weight_key in weights:
            kv = unpack_mxfp8(weights.pop(kv_weight_key), weights.pop(kv_scale_key))
            kv = kv.reshape(
                self.args.num_attention_heads,
                self.args.qk_nope_head_dim + self.args.v_head_dim,
                self.args.kv_lora_rank,
            )
            repack_mxfp8(
                f"{attn_prefix}.embed_q",
                mx.contiguous(kv[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)),
            )
            repack_mxfp8(
                f"{attn_prefix}.unembed_out",
                mx.contiguous(kv[:, self.args.qk_nope_head_dim :, :]),
            )

        transformed = {}
        for key, value in weights.items():
            if key.endswith(".weight_scale"):
                weight_key = key[: -len("_scale")]
                if weight_key not in weights:
                    raise ValueError(f"Missing MXFP8 weight for {weight_key}.")
                transformed[weight_key] = pack_mxfp8(weights[weight_key])
                transformed[f"{weight_key[:-len('weight')]}scales"] = value
            elif key.endswith(".weight") and f"{key}_scale" in weights:
                continue
            else:
                transformed[key] = value
        return transformed
