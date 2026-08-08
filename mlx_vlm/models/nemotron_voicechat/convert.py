"""Prepare an upload-ready mlx-vlm artifact from VoiceChat safetensors.

The converter adds the runtime config and tokenizer files required by generic
``mlx_vlm.load`` dispatch. Existing indexed MLX shards are preserved without
repacking unless ``--copy-weights`` is requested.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

OFFICIAL_REVISION = "c5d3b70183b6bb9d7553590e111b05685049751c"
BASE_TOKENIZER_REPO = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BASE_TOKENIZER_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
RUNTIME_CONFIG_VERSION = 2


def _resolve(
    source: str | Path,
    revision: str | None,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    return Path(
        snapshot_download(
            str(source),
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )


def _text_config(base: dict) -> dict:
    pattern = base.get("hybrid_override_pattern")
    if pattern is None:
        raise ValueError("base tokenizer model config lacks hybrid_override_pattern")
    time_step_limit = base.get("time_step_limit")
    if time_step_limit is None:
        # ``time_step_min/max`` control parameter initialization in Nemotron-H;
        # they are not the runtime dt clamp. The reference runtime is unbounded
        # unless ``time_step_limit`` is explicitly configured.
        time_step_limit = [0.0, float("inf")]
    return {
        "model_type": "nemotron_h",
        "vocab_size": base["vocab_size"],
        "hidden_size": base["hidden_size"],
        "intermediate_size": base["intermediate_size"],
        "num_hidden_layers": base["num_hidden_layers"],
        "max_position_embeddings": base["max_position_embeddings"],
        "num_attention_heads": base["num_attention_heads"],
        "num_key_value_heads": base["num_key_value_heads"],
        "attention_bias": base.get("attention_bias", False),
        "mamba_num_heads": base["mamba_num_heads"],
        "mamba_head_dim": base["mamba_head_dim"],
        "mamba_proj_bias": base.get("mamba_proj_bias", False),
        "ssm_state_size": base.get("ssm_state_size", base["mamba_state_dim"]),
        "conv_kernel": base["conv_kernel"],
        "n_groups": base.get("n_groups", base["mamba_num_groups"]),
        "mlp_bias": base.get("mlp_bias", False),
        "layer_norm_epsilon": base.get(
            "layer_norm_epsilon", base.get("rms_norm_eps", 1e-5)
        ),
        "use_bias": base.get("use_bias", False),
        "use_conv_bias": base.get("use_conv_bias", True),
        "hybrid_override_pattern": pattern,
        "head_dim": base.get("head_dim"),
        "time_step_limit": time_step_limit,
    }


def build_runtime_config(source: dict, base: dict) -> dict:
    try:
        voicechat = source["model"]
        stt = voicechat["stt"]["model"]
        speech = voicechat["speech_generation"]["model"]
        perception = stt["perception"]
        rnnt = source["_rnnt_merge_info"]
    except KeyError as exc:
        raise ValueError("unrecognized VoiceChat config schema") from exc

    encoder = dict(perception["encoder"])
    encoder.pop("_target_", None)
    context = encoder.get("att_context_size", [70, 0])
    if context and isinstance(context[0], int):
        encoder["att_context_size"] = [context]

    preprocessor = dict(perception["preprocessor"])
    preprocessor.pop("_target_", None)
    decoder_source = rnnt["decoder_config"]
    joint_source = rnnt["joint_config"]
    decoder = {
        **decoder_source["prednet"],
        "vocab_size": decoder_source["vocab_size"],
        "blank_as_pad": decoder_source.get("blank_as_pad", True),
    }
    joint = {
        **joint_source["jointnet"],
        "num_classes": joint_source["num_classes"],
    }

    tts_source = speech["tts_config"]
    tts_backbone = tts_source["backbone_config"]
    character = tts_source["cas_config"]["backbone_config"]["encoder"]
    mog = tts_source["mog_head_config"]
    codec = speech["codec_config"]

    config = {
        "mlx_runtime_config_version": RUNTIME_CONFIG_VERSION,
        "model_type": "nemotron_voicechat",
        "architectures": ["NemotronVoiceChatForConditionalGeneration"],
        "text_config": _text_config(base),
        "audio_config": {
            "preprocessor": preprocessor,
            "encoder": encoder,
            "decoder": decoder,
            "joint": joint,
            "output_dim": perception["output_dim"],
            "max_symbols": 10,
        },
        "tts_config": {
            **{
                key: tts_backbone[key]
                for key in (
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "sliding_window",
                )
            },
            "latent_size": tts_source["latent_size"],
            "num_quantizers": tts_source["num_quantizers"],
            "codebook_size": tts_source["codebook_size"],
            "num_delay_speech_tokens": tts_source["num_delay_speech_tokens"],
            "exponent": tts_source["exponent"],
            "disable_eos_prediction": tts_source["disable_eos_prediction"],
            "use_gated_fusion_for_text_audio": tts_source[
                "use_gated_fusion_for_text_audio"
            ],
            "use_subword_flag_emb": tts_source["use_subword_flag_emb"],
            "use_bos_eos_emb": tts_source["use_bos_eos_emb"],
            "use_audio_prompt_frozen_projection": tts_source[
                "use_audio_prompt_frozen_projection"
            ],
            "character_encoder": character,
            "mog_head": mog,
            "guidance_scale": speech["inference_guidance_scale"],
            "top_p": speech["inference_top_p_or_k"],
            "noise_scale": speech["inference_noise_scale"],
            "audio_prompt_duration": voicechat["speech_generation"]["data"][
                "audio_prompt_duration"
            ],
        },
        "codec_config": {
            "sample_rate": voicechat["speech_generation"]["data"]["target_sample_rate"],
            "base_channels": codec["base_hidden_size"],
            "channel_multipliers": codec["channel_mult"],
            "downsample_rates": codec["rates"],
            "blocks_per_stage": codec["num_blocks"],
            "block_kernel_size": codec["kernel_size"],
            "latent_dim": codec["latent_size"],
            "n_fft": codec["n_fft"],
            "hop_length": codec["hop_length"],
            "num_quantizers": codec["num_quantizers"],
            "codebook_size": codec["codebook_size"],
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 12,
        "silence_token_id": 11,
        "rnnt_blank_id": 1024,
        "input_sample_rate": source["data"]["source_sample_rate"],
        "output_sample_rate": source["data"]["target_sample_rate"],
        "frame_duration": source["data"]["frame_length"],
        "function_channel_weight": stt["duplex_function_channel_weight"],
        "source_revision": OFFICIAL_REVISION,
        "base_tokenizer_revision": BASE_TOKENIZER_REVISION,
        "speaker": "Aria",
        "rnnt_vocabulary": joint_source.get("vocabulary", []),
    }

    quantization = source.get("mlx_conversion", {}).get("quantization")
    if quantization:
        runtime_quantization = {
            "group_size": quantization["group_size"],
            "bits": quantization["bits"],
            **quantization.get("modules", {}),
        }
        config["quantization"] = runtime_quantization
        config["quantization_config"] = runtime_quantization
    return config


def _link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def prepare_artifact(
    source_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    *,
    copy_weights: bool = False,
) -> Path:
    source_config = json.loads((source_path / "config.json").read_text())
    base_config = json.loads((tokenizer_path / "config.json").read_text())
    runtime_config = build_runtime_config(source_config, base_config)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "config.json").write_text(
        json.dumps(runtime_config, indent=2, ensure_ascii=False) + "\n"
    )

    index = source_path / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"missing {index}")
    _link_or_copy(index, output_path / index.name, copy_weights)
    shard_names = set(json.loads(index.read_text())["weight_map"].values())
    for shard_name in sorted(shard_names):
        _link_or_copy(
            source_path / shard_name,
            output_path / shard_name,
            copy_weights,
        )

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        candidate = tokenizer_path / name
        if candidate.exists():
            shutil.copy2(candidate, output_path / name)

    tokenizer_config_path = output_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config["fix_mistral_regex"] = True
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n"
        )

    readme = Path(__file__).with_name("README.md")
    if readme.exists():
        shutil.copy2(readme, output_path / "README.md")

    rnnt_source = source_path / "rnnt_tokenizer"
    if rnnt_source.exists():
        rnnt_output = output_path / "rnnt_tokenizer"
        rnnt_output.mkdir(exist_ok=True)
        for candidate in rnnt_source.iterdir():
            if candidate.is_file():
                shutil.copy2(candidate, rnnt_output / candidate.name)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tokenizer", default=BASE_TOKENIZER_REPO)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-revision", default=BASE_TOKENIZER_REVISION)
    parser.add_argument("--copy-weights", action="store_true")
    args = parser.parse_args()

    source = _resolve(args.source, args.revision)
    tokenizer = _resolve(
        args.tokenizer,
        args.tokenizer_revision,
        allow_patterns=TOKENIZER_FILES,
    )
    prepare_artifact(source, tokenizer, args.output, copy_weights=args.copy_weights)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
