"""Load the packed NVFP4 experts in MiniMax M3.1 Flash."""

import mlx.core as mx


def sanitize_nvfp4_experts(weights: dict, args):
    """Stack routed expert projections without altering their quantization."""
    for layer_idx in range(args.num_hidden_layers):
        prefix = f"language_model.model.layers.{layer_idx}.block_sparse_moe"
        for source, target in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            if f"{prefix}.experts.0.{source}.weight_packed" not in weights:
                continue
            packed, scales, global_scales = [], [], []
            for expert_idx in range(args.num_local_experts):
                key = f"{prefix}.experts.{expert_idx}.{source}"
                packed.append(weights.pop(f"{key}.weight_packed").view(mx.uint32))
                scales.append(weights.pop(f"{key}.weight_scale"))
                global_scales.append(
                    weights.pop(f"{key}.weight_global_scale").reshape(())
                )
                # Activation quantization metadata is not needed for BF16 inputs.
                weights.pop(f"{key}.input_global_scale", None)
            target = f"{prefix}.switch_mlp.{target}"
            weights[f"{target}.weight"] = mx.stack(packed)
            weights[f"{target}.scales"] = mx.stack(scales)
            weights[f"{target}.weight_global_scale"] = mx.stack(global_scales)
