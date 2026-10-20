# Copyright © 2026 Apple Inc.

import re

import mlx.core as mx

from .language import residual_gate_coefficients


def norm_gain(weight: mx.array) -> mx.array:
    return (1.0 + weight.astype(mx.float32)).astype(mx.bfloat16)


def interleave_rope_rows(w: mx.array, head_dim: int) -> mx.array:
    """Hugging Face orders the q and k rows of a head for RoPE pairs (i, i +
    head_dim / 2). The model pairs neighbor rows (2i, 2i + 1)."""
    shape = w.shape
    w = w.reshape(-1, 2, head_dim // 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


_UNUSED_QUANT_TENSORS = (
    "mlp.experts.gate_up_proj_input_scale",
    "mlp.experts.down_proj_input_scale",
)

# Norm weights the checkpoint stores centered at zero; sanitize stores the gain.
_ZERO_CENTERED_NORMS = (
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "mlp.pre_expert_norm.weight",
    "mlp.experts.post_expert_norm.weight",
)


class CheckpointMixin:
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        drop_prefixes = ()
        if self.config.audio_config is None:
            drop_prefixes += ("model.audio_tower.", "audio_tower.")
        if self.args.vision is None:
            drop_prefixes += (
                "model.vision_tower.",
                "model.vision_adapter.",
                "model.vision_projection.",
            )
        # The gates are fixed, so compute their weights once.
        gates = {
            "post_attention_residual_gate.gate": "add_attn",
            "post_feedforward_residual_gate.gate": "add_ffn",
        }
        layer_re = re.compile(r"^(model\.language_model\.layers\.\d+\.)(.+)$")
        # Weights from mlx_lm.convert already store the gate weights and the
        # norm gains.
        converted = any(k.endswith("_alpha") for k in weights)

        sanitized = {}
        for k, v in weights.items():
            if k.startswith(drop_prefixes):
                continue
            m = layer_re.match(k)
            if m is None or converted:
                sanitized[k] = v
                continue

            prefix, suffix = m.groups()
            experts = prefix + "mlp.experts."
            if suffix in gates:
                alpha, beta = residual_gate_coefficients(v)
                sanitized[prefix + gates[suffix] + "_alpha"] = alpha
                sanitized[prefix + gates[suffix] + "_beta"] = beta
            elif suffix in _ZERO_CENTERED_NORMS:
                sanitized[k] = norm_gain(v)
            elif suffix == "mlp.experts.gate_up_proj":
                gate_proj, up_proj = mx.split(v, 2, axis=1)
                if v.dtype == mx.uint8:
                    # In mlx quantized weights are packed into uint32
                    gate_proj = gate_proj.view(mx.uint32)
                    up_proj = up_proj.view(mx.uint32)
                sanitized[experts + "gate_proj.weight"] = gate_proj
                sanitized[experts + "up_proj.weight"] = up_proj
            elif suffix == "mlp.experts.gate_up_proj_weight_scale":
                # One e4m3 block scale per 16 inputs
                gate_scales, up_scales = mx.split(v, 2, axis=1)
                sanitized[experts + "gate_proj.scales"] = gate_scales
                sanitized[experts + "up_proj.scales"] = up_scales
            elif suffix == "mlp.experts.gate_up_proj_weight_scale_2":
                # Exports use either a shared/per-expert factor or separate
                # factors for the gate and up halves.
                if v.ndim == 2 and v.shape[-1] == 2:
                    gate_gs, up_gs = v[:, 0], v[:, 1]
                else:
                    gate_gs = up_gs = v
                sanitized[experts + "gate_proj.weight_scale_2"] = gate_gs
                sanitized[experts + "up_proj.weight_scale_2"] = up_gs
            elif suffix == "mlp.experts.down_proj":
                sanitized[experts + "down_proj.weight"] = (
                    v.view(mx.uint32) if v.dtype == mx.uint8 else v
                )
            elif suffix == "mlp.experts.down_proj_weight_scale":
                sanitized[experts + "down_proj.scales"] = v
            elif suffix == "mlp.experts.down_proj_weight_scale_2":
                sanitized[experts + "down_proj.weight_scale_2"] = v
            elif suffix in _UNUSED_QUANT_TENSORS:
                # we don't quantize activations
                continue
            else:
                sanitized[k] = v

        # The model runs q, k, v and the output gate as one matmul.
        head_dim = self.args.text.head_dim
        projections = ("q_proj", "k_proj", "v_proj", "gate_proj")
        for k in [k for k in sanitized if k.endswith("self_attn.q_proj.weight")]:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "scales", "biases"):
                keys = [f"{attn}{p}.{param}" for p in projections]
                if all(key in sanitized for key in keys):
                    q, k_, v, g = (sanitized.pop(key) for key in keys)
                    q = interleave_rope_rows(q, head_dim)
                    k_ = interleave_rope_rows(k_, head_dim)
                    sanitized[f"{attn}qkvg_proj.{param}"] = mx.concatenate(
                        [q, k_, v, g]
                    )

        vision_q = [
            k
            for k in sanitized
            if k.startswith("model.vision_tower.") and k.endswith(".attn.q_proj.weight")
        ]
        for k in vision_q:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "bias"):
                keys = [f"{attn}{p}_proj.{param}" for p in "qkv"]
                q, k_, v = (sanitized.pop(key) for key in keys)
                q = interleave_rope_rows(q, self.args.vision.head_dim)
                k_ = interleave_rope_rows(k_, self.args.vision.head_dim)
                sanitized[f"{attn}qkv_proj.{param}"] = mx.concatenate([q, k_, v])
        result = {}
        for key, value in sanitized.items():
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            elif key.startswith("model."):
                key = key[len("model.") :]
            if (
                key.startswith("audio_tower.")
                and key.endswith(".conv.weight")
                and value.shape[-1] != 1
            ):
                # PyTorch: (channels, 1, kernel), MLX: (channels, kernel, 1).
                value = value.transpose(0, 2, 1)
            if key.endswith(".global_scale"):
                key = key.removesuffix(".global_scale") + ".weight_scale_2"
                value = value.astype(mx.float32) / (6.0 * 448.0)
            result[key] = value
        return result
