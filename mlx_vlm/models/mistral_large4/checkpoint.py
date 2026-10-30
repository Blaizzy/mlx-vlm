"""Read native Mistral NVFP4 weights, retaining packed expert bytes."""

import json
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import ModelConfig, TextConfig, VisionConfig, config_from_params
from .mistral_large4 import Model


def dequantize_fp8(weight, scale):
    """Mistral's FP8_PB_WO tensors use 128 x 128 blocks."""
    w = mx.from_fp8(weight, dtype=mx.float32)
    if scale.ndim == 4:
        o, _, i, _ = scale.shape
        w = (w.reshape(o, 128, i, 128) * scale).reshape(weight.shape)
    elif scale.ndim == 0:
        w = w * scale
    else:
        raise ValueError(f"Unsupported Mistral FP8 scale shape: {scale.shape}")
    return w.astype(mx.bfloat16)


def load_native(model_path, *, group=None, lazy=False, progress=None):
    from ...utils import _load_safetensors

    path = Path(model_path)
    params = json.loads((path / "params.json").read_text())
    config = config_from_params(params)
    manifest = (
        json.loads((path / "rank.json").read_text())
        if (path / "rank.json").exists()
        else None
    )
    n = config["text_config"]["n_routed_experts"]
    if manifest:
        start, end = manifest["expert_start"], manifest["expert_end"]
        if group is not None and (group.rank(), group.size()) != (
            manifest["rank"],
            manifest["size"],
        ):
            raise ValueError("Staged checkpoint belongs to another distributed rank")
    elif group is not None:
        start, end = (
            n * group.rank() // group.size(),
            n * (group.rank() + 1) // group.size(),
        )
    else:
        start, end = 0, n
    config["text_config"].update(expert_start=start, expert_end=end)
    args = ModelConfig.from_dict(config)
    args.text_config = TextConfig.from_dict(config["text_config"])
    args.vision_config = VisionConfig.from_dict(config["vision_config"])
    if not args.text_config.nvfp4:
        raise ValueError(
            "Native Mistral Large 4 loading currently requires NVFP4 experts"
        )
    model = Model(args)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        if shard.name.startswith(("consolidated", "model")):
            for key, value in _load_safetensors(str(shard)).items():
                expert = re.match(r"layers\.\d+\.experts\.(\d+)\.", key)
                if key.startswith("mtp") or (
                    expert and not start <= int(expert[1]) < end
                ):
                    continue
                weights[key] = value

    def take(key):
        value = weights.pop(key)
        if key.endswith(".weight") and value.dtype == mx.uint8:
            scale = weights.pop(key[: -len(".weight")] + ".weight_scale")
            return dequantize_fp8(value, scale)
        return value

    def assign(module, pairs):
        module.load_weights(pairs, strict=True)
        if not lazy:
            mx.eval(module.parameters())

    for i, layer in enumerate(model.language_model.layers):
        prefix = f"layers.{i}."
        pairs = []
        for name, _ in tree_flatten(layer.parameters()):
            expert = re.match(
                r"feed_forward.experts\.(w[123])\.(weight|scales|global_scale)$", name
            )
            if expert:
                proj, kind = expert.groups()
                suffix = {
                    "weight": "weight",
                    "scales": "weight_scale",
                    "global_scale": "weight_scale_2",
                }[kind]
                tensors = [
                    weights.pop(f"{prefix}experts.{e}.{proj}.{suffix}")
                    for e in range(start, end)
                ]
                value = mx.stack(tensors)
                if kind == "weight":
                    value = value.view(mx.uint32)
                if kind == "global_scale":
                    value = value.reshape(-1).astype(mx.float32)
                pairs.append((name, value))
            else:
                native = name
                if i >= args.text_config.first_k_dense_replace:
                    native = native.removeprefix("feed_forward.")
                pairs.append((name, take(prefix + native)))
        assign(layer, pairs)
        if progress:
            progress(i + 1, len(model.language_model.layers))

    lm = model.language_model
    lm.model.embed_tokens.weight = take("tok_embeddings.weight")
    lm.model.norm.weight = take("norm.weight")
    lm.model.rotary_emb.alpha = take("lerope.alpha")
    lm.lm_head.weight = take("output.weight")
    vision = []
    for name, _ in tree_flatten(model.vision_tower.parameters()):
        native = name.replace("vision_model.", "vision_encoder.", 1)
        for target, source in (
            ("q_proj", "wq"),
            ("k_proj", "wk"),
            ("v_proj", "wv"),
            ("o_proj", "wo"),
            ("gate_proj", "w1"),
            ("down_proj", "w2"),
            ("up_proj", "w3"),
        ):
            native = native.replace("." + target + ".", "." + source + ".")
        value = take(native)
        if native.endswith("patch_conv.weight"):
            value = value.transpose(0, 2, 3, 1)
        vision.append((name, value))
    assign(model.vision_tower, vision)
    projector = {
        "norm.weight": "pre_mm_projector_norm.weight",
        "patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
        "linear_1.weight": "vision_language_adapter.w_in.weight",
        "linear_2.weight": "vision_language_adapter.w_out.weight",
    }
    assign(model.multi_modal_projector, [(k, take(v)) for k, v in projector.items()])
    unused = [k for k in weights if not k.endswith(".input_scale")]
    if unused:
        raise ValueError(f"Unconsumed model weights: {unused[:20]}")
    if group is not None:
        model.shard(group)
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    return model
