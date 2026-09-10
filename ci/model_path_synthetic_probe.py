from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.change_rules import load_yaml_mapping
from ci.components.model_path import SYNTHETIC_ADAPTERS


def _bert(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx

    from mlx_vlm.models import bert

    text = profile["text"]
    inputs = profile["inputs"]
    config = bert.ModelConfig(
        model_type="bert",
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        intermediate_size=text["intermediate_size"],
        max_position_embeddings=text["max_position_embeddings"],
    )
    mx.random.seed(0)
    model = bert.Model(config)
    batch_size = inputs["batch_size"]
    sequence_length = inputs["sequence_length"]
    input_ids = mx.arange(batch_size * sequence_length).reshape(
        batch_size, sequence_length
    )
    input_ids = input_ids % text["vocab_size"]
    attention_mask = mx.ones((batch_size, sequence_length))
    return model, model(input_ids, attention_mask=attention_mask).text_embeds


def _qwen2_vl(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx

    from mlx_vlm.models import qwen2_vl

    text = profile["text"]
    vision = profile["vision"]
    inputs = profile["inputs"]
    text_config = qwen2_vl.TextConfig(
        model_type="qwen2_vl",
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        intermediate_size=text["intermediate_size"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        rms_norm_eps=1e-6,
        vocab_size=text["vocab_size"],
        max_position_embeddings=text["max_position_embeddings"],
        rope_theta=10000,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        tie_word_embeddings=False,
    )
    vision_config = qwen2_vl.VisionConfig(
        model_type="qwen2_vl",
        depth=vision["num_hidden_layers"],
        embed_dim=vision["hidden_size"],
        hidden_size=text["hidden_size"],
        image_size=vision["image_size"],
        num_heads=vision["num_attention_heads"],
        patch_size=vision["patch_size"],
        mlp_ratio=vision["intermediate_size"] / vision["hidden_size"],
        in_channels=vision["num_channels"],
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    image_token_id = text["vocab_size"] - 2
    vision_start_token_id = text["vocab_size"] - 3
    config = qwen2_vl.ModelConfig(
        model_type="qwen2_vl",
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=image_token_id,
        video_token_id=text["vocab_size"] - 1,
        vision_start_token_id=vision_start_token_id,
        vocab_size=text["vocab_size"],
    )
    mx.random.seed(0)
    model = qwen2_vl.Model(config)
    grid = mx.array([[1, 2, 2]])
    patch_values = mx.arange(
        4
        * vision["num_channels"]
        * vision_config.temporal_patch_size
        * vision["patch_size"]
        * vision["patch_size"],
        dtype=mx.float32,
    ).reshape(4, -1)
    patch_values = patch_values / max(1, patch_values.size)
    tokens = [vision_start_token_id, image_token_id]
    tokens.extend(range(1, inputs["sequence_length"] - len(tokens) + 1))
    output = model(
        mx.array([tokens]),
        pixel_values=patch_values,
        image_grid_thw=grid,
    )
    return model, output.logits


def _qwen2_5_vl(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx

    from mlx_vlm.models import qwen2_5_vl

    text = profile["text"]
    vision = profile["vision"]
    inputs = profile["inputs"]
    text_config = qwen2_5_vl.TextConfig(
        model_type="qwen2_5_vl",
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        intermediate_size=text["intermediate_size"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        rms_norm_eps=1e-6,
        vocab_size=text["vocab_size"],
        max_position_embeddings=text["max_position_embeddings"],
        rope_theta=10000,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        tie_word_embeddings=False,
    )
    vision_config = qwen2_5_vl.VisionConfig(
        model_type="qwen2_5_vl",
        depth=vision["num_hidden_layers"],
        hidden_size=vision["hidden_size"],
        intermediate_size=vision["intermediate_size"],
        out_hidden_size=text["hidden_size"],
        image_size=vision["image_size"],
        num_heads=vision["num_attention_heads"],
        patch_size=vision["patch_size"],
        in_channels=vision["num_channels"],
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=vision["image_size"],
        fullatt_block_indexes=[0, 1],
    )
    image_token_id = text["vocab_size"] - 2
    vision_start_token_id = text["vocab_size"] - 3
    config = qwen2_5_vl.ModelConfig(
        model_type="qwen2_5_vl",
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=image_token_id,
        video_token_id=text["vocab_size"] - 1,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=text["vocab_size"] - 4,
        vision_token_id=text["vocab_size"] - 5,
        vocab_size=text["vocab_size"],
    )
    mx.random.seed(0)
    model = qwen2_5_vl.Model(config)
    grid = mx.array([[1, 2, 2]])
    patch_values = mx.arange(
        4
        * vision["num_channels"]
        * vision_config.temporal_patch_size
        * vision["patch_size"]
        * vision["patch_size"],
        dtype=mx.float32,
    ).reshape(4, -1)
    patch_values = patch_values / max(1, patch_values.size)
    tokens = [vision_start_token_id, image_token_id]
    tokens.extend(range(1, inputs["sequence_length"] - len(tokens) + 1))
    output = model(
        mx.array([tokens]),
        pixel_values=patch_values,
        image_grid_thw=grid,
    )
    return model, output.logits


def _granite_vision(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx

    from mlx_vlm.models import granite_vision

    text = profile["text"]
    vision = profile["vision"]
    inputs = profile["inputs"]
    text_config = granite_vision.TextConfig(
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        intermediate_size=text["intermediate_size"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        vocab_size=text["vocab_size"],
        max_position_embeddings=text["max_position_embeddings"],
        tie_word_embeddings=False,
    )
    vision_config = granite_vision.VisionConfig(
        num_hidden_layers=vision["num_hidden_layers"],
        hidden_size=vision["hidden_size"],
        intermediate_size=vision["intermediate_size"],
        num_attention_heads=vision["num_attention_heads"],
        image_size=vision["image_size"],
        patch_size=vision["patch_size"],
        num_channels=vision["num_channels"],
    )
    image_token_id = text["vocab_size"] - 1
    config = granite_vision.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=image_token_id,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        vocab_size=text["vocab_size"],
    )
    mx.random.seed(0)
    model = granite_vision.Model(config)
    tokens = [image_token_id, image_token_id]
    tokens.extend(range(1, inputs["sequence_length"] - len(tokens) + 1))
    image_values = mx.arange(
        vision["num_channels"] * vision["image_size"] * vision["image_size"],
        dtype=mx.float32,
    ).reshape(1, 1, vision["num_channels"], vision["image_size"], vision["image_size"])
    output = model(mx.array([tokens]), image_values / max(1, image_values.size), None)
    return model, output.logits


def _internvl_chat(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx

    from mlx_vlm.models import internvl_chat

    text = profile["text"]
    vision = profile["vision"]
    inputs = profile["inputs"]
    text_config = internvl_chat.TextConfig(
        model_type="qwen2",
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        intermediate_size=text["intermediate_size"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        rms_norm_eps=1e-6,
        vocab_size=text["vocab_size"],
        max_position_embeddings=text["max_position_embeddings"],
        tie_word_embeddings=False,
        attention_bias=False,
    )
    vision_config = internvl_chat.VisionConfig(
        model_type="intern_vit_6b",
        hidden_size=vision["hidden_size"],
        num_hidden_layers=vision["num_hidden_layers"],
        intermediate_size=vision["intermediate_size"],
        num_attention_heads=vision["num_attention_heads"],
        image_size=vision["image_size"],
        patch_size=vision["patch_size"],
        num_channels=vision["num_channels"],
        drop_path_rate=0.0,
    )
    image_token_id = text["vocab_size"] - 1
    config = internvl_chat.ModelConfig(
        model_type="internvl_chat",
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=image_token_id,
        video_token_index=text["vocab_size"] - 2,
        vocab_size=text["vocab_size"],
        downsample_ratio=0.5,
    )
    mx.random.seed(0)
    model = internvl_chat.Model(config)
    tokens = [image_token_id]
    tokens.extend(range(1, inputs["sequence_length"] - len(tokens) + 1))
    image_values = mx.arange(
        vision["num_channels"] * vision["image_size"] * vision["image_size"],
        dtype=mx.float32,
    ).reshape(1, 1, vision["num_channels"], vision["image_size"], vision["image_size"])
    output = model(mx.array([tokens]), image_values / max(1, image_values.size), None)
    return model, output.logits


def _deepseek_vl_v2(profile: Mapping[str, Any]) -> tuple[Any, Any]:
    import mlx.core as mx
    import numpy as np

    from mlx_vlm.models import deepseek_vl_v2

    text = profile["text"]
    vision = profile["vision"]
    moe = profile["mixture_of_experts"]
    text_config = deepseek_vl_v2.TextConfig(
        model_type="deepseek_v2",
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        moe_intermediate_size=moe["expert_intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        n_shared_experts=1,
        n_routed_experts=moe["num_experts"],
        num_experts_per_tok=moe["num_experts_per_token"],
        first_k_dense_replace=0,
        moe_layer_freq=1,
        max_position_embeddings=text["max_position_embeddings"],
        qk_nope_head_dim=0,
        qk_rope_head_dim=8,
        v_head_dim=text["head_dim"],
        kv_lora_rank=16,
        q_lora_rank=16,
    )
    vision_config = deepseek_vl_v2.VisionConfig(
        model_type="vision",
        layers=vision["num_hidden_layers"],
        width=vision["hidden_size"],
        intermediate_size=vision["intermediate_size"],
        num_attention_heads=vision["num_attention_heads"],
        image_size=vision["image_size"],
        patch_size=vision["patch_size"],
        num_channels=vision["num_channels"],
        mlp_ratio=vision["intermediate_size"] / vision["hidden_size"],
    )
    projector_config = deepseek_vl_v2.ProjectorConfig(
        projector_type="downsample_mlp_gelu",
        input_dim=vision["hidden_size"],
        n_embed=text["hidden_size"],
        depth=2,
        mlp_ratio=1,
        downsample_ratio=2,
        token_pooling=False,
    )
    image_token_id = text["vocab_size"] - 1
    config = deepseek_vl_v2.ModelConfig(
        model_type="deepseek_vl_v2",
        text_config=text_config,
        vision_config=vision_config,
        projector_config=projector_config,
        image_token_index=image_token_id,
        pad_id=0,
        num_image_tokens=5,
        vocab_size=text["vocab_size"],
        tile_tag="2D",
        global_view_pos="head",
    )
    mx.random.seed(0)
    model = deepseek_vl_v2.Model(config)
    tokens = [image_token_id] * 5 + [1, 2, 3]
    image_values = mx.arange(
        2 * vision["num_channels"] * vision["image_size"] * vision["image_size"],
        dtype=mx.float32,
    ).reshape(1, 2, vision["num_channels"], vision["image_size"], vision["image_size"])
    output = model(
        mx.array([tokens]),
        image_values / max(1, image_values.size),
        None,
        images_spatial_crop=np.array([[[1, 1], [0, 0]]]),
        images_seq_mask=mx.array([[True] * 5 + [False] * 3]),
    )
    return model, output.logits


ADAPTERS = {
    "bert": _bert,
    "deepseek_vl_v2": _deepseek_vl_v2,
    "granite_vision": _granite_vision,
    "internvl_chat": _internvl_chat,
    "qwen2_5_vl": _qwen2_5_vl,
    "qwen2_vl": _qwen2_vl,
}

if frozenset(ADAPTERS) != SYNTHETIC_ADAPTERS:
    raise RuntimeError("synthetic adapter registry is inconsistent")


def _profile(job: Mapping[str, Any], profiles: Mapping[str, Any]) -> Mapping[str, Any]:
    synthetic = job.get("synthetic")
    if not isinstance(synthetic, Mapping):
        raise ValueError("ModelPath work has no synthetic configuration")
    name = synthetic.get("profile")
    if not isinstance(name, str) or name not in profiles:
        raise ValueError("synthetic profile is not configured")
    value = profiles[name]
    if not isinstance(value, Mapping):
        raise ValueError("synthetic profile must be an object")
    if value.get("base"):
        base = profiles.get(value["base"])
        if not isinstance(base, Mapping):
            raise ValueError("synthetic base profile is not configured")
        merged = dict(base)
        merged.update(value)
        return merged
    return value


def run(job: Mapping[str, Any], profiles: Mapping[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    synthetic = job["synthetic"]
    adapter_name = str(synthetic["adapter"])
    adapter = ADAPTERS.get(adapter_name)
    if adapter is None:
        raise ValueError(f"synthetic adapter is not implemented: {adapter_name}")
    model, logits = adapter(_profile(job, profiles))
    mx.eval(logits, model.parameters())
    array = np.asarray(logits, dtype=np.float32)
    parameters = [
        (name, tuple(int(dimension) for dimension in value.shape))
        for name, value in tree_flatten(model.parameters())
    ]
    signature = hashlib.sha256(
        json.dumps(parameters, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "adapter": adapter_name,
        "profile": synthetic["profile"],
        "parameter_signature": signature,
        "output_hash": hashlib.sha256(array.tobytes()).hexdigest()[:16],
        "output_shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "output": array.reshape(-1).tolist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text())
    config = load_yaml_mapping(args.profiles)
    profiles = config.get("synthetic_profiles", {})
    result = run(job, profiles)
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
