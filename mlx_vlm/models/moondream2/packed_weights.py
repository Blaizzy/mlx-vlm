import mlx.core as mx


def unpack_checkpoint_weights(weights):
    unpacked = {}
    for key, value in weights.items():
        if key.startswith("model.region."):
            continue
        if key.endswith((".weight.scale", ".weight.zero_point")):
            continue
        if key.endswith(".weight.packed"):
            base = key.removesuffix(".packed")
            bias = weights[base.removesuffix(".weight") + ".bias"]
            scale = weights[base + ".scale"]
            zero = weights[base + ".zero_point"]
            codes = mx.concatenate([value >> 4, value & 15], axis=0)
            # The source decoder rounds after subtraction and multiplication.
            centered = (codes.astype(mx.bfloat16) - zero).astype(mx.bfloat16)
            value = (centered * scale).astype(mx.bfloat16).reshape(bias.size, -1)
            key = base
        elif (
            key.endswith(".bias")
            and key.removesuffix(".bias") + ".weight.packed" in weights
        ):
            value = value.astype(mx.bfloat16)
        key = key.removeprefix("model.")
        if key == "text.wte":
            key = "text.model.embed_tokens.weight"
        elif key.startswith("text.blocks."):
            key = key.replace("text.blocks.", "text.model.layers.", 1)
        elif key.startswith("text.post_ln."):
            key = key.replace("text.post_ln.", "text.model.post_ln.", 1)
        elif key.startswith("vision.") and not key.startswith("vision.proj_mlp."):
            key = key.replace("vision.", "vision.encoder.", 1)
        unpacked[key] = value
    return unpacked
