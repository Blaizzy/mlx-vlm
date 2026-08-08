import json

from mlx_vlm.models.nemotron_voicechat import convert
from mlx_vlm.models.nemotron_voicechat.convert import build_runtime_config


def _source_config():
    return {
        "data": {
            "source_sample_rate": 16_000,
            "target_sample_rate": 22_050,
            "frame_length": 0.08,
        },
        "model": {
            "stt": {
                "model": {
                    "duplex_function_channel_weight": 2.0,
                    "perception": {
                        "output_dim": 4_480,
                        "preprocessor": {
                            "_target_": "Preprocessor",
                            "sample_rate": 16_000,
                            "features": 128,
                            "n_fft": 512,
                        },
                        "encoder": {
                            "_target_": "Conformer",
                            "feat_in": 128,
                            "n_layers": 24,
                            "d_model": 1_024,
                            "n_heads": 8,
                            "att_context_size": [70, 0],
                        },
                    },
                }
            },
            "speech_generation": {
                "data": {
                    "audio_prompt_duration": 3.0,
                    "target_sample_rate": 22_050,
                },
                "model": {
                    "inference_guidance_scale": 0.2,
                    "inference_top_p_or_k": 0.95,
                    "inference_noise_scale": 0.001,
                    "codec_config": {
                        "base_hidden_size": 384,
                        "channel_mult": [1, 2, 4],
                        "rates": [7, 7, 9],
                        "num_blocks": 3,
                        "kernel_size": 7,
                        "latent_size": 512,
                        "n_fft": 16,
                        "hop_length": 4,
                        "num_quantizers": 31,
                        "codebook_size": 1_024,
                    },
                    "tts_config": {
                        "hidden_size": 1_152,
                        "latent_size": 512,
                        "num_quantizers": 31,
                        "codebook_size": 1_024,
                        "num_delay_speech_tokens": 2,
                        "exponent": 3.0,
                        "disable_eos_prediction": True,
                        "use_gated_fusion_for_text_audio": True,
                        "use_subword_flag_emb": True,
                        "use_bos_eos_emb": True,
                        "use_audio_prompt_frozen_projection": True,
                        "backbone_config": {
                            "hidden_size": 1_152,
                            "intermediate_size": 4_608,
                            "num_hidden_layers": 28,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 16,
                            "head_dim": 72,
                            "sliding_window": 7_500,
                        },
                        "cas_config": {
                            "backbone_config": {
                                "encoder": {
                                    "hidden_size": 1_152,
                                    "intermediate_size": 4_608,
                                    "num_hidden_layers": 1,
                                    "num_attention_heads": 16,
                                    "num_key_value_heads": 16,
                                    "head_dim": 72,
                                }
                            }
                        },
                        "mog_head_config": {
                            "intermediate_size": 4_608,
                            "low_rank": 64,
                            "min_log_std": -4.0,
                            "num_layers": 3,
                            "num_predictions": 1_024,
                        },
                    },
                },
            },
        },
        "_rnnt_merge_info": {
            "decoder_config": {
                "vocab_size": 1_024,
                "blank_as_pad": True,
                "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
            },
            "joint_config": {
                "num_classes": 1_024,
                "vocabulary": ["<unk>", "▁hello"],
                "jointnet": {
                    "joint_hidden": 640,
                    "activation": "relu",
                    "encoder_hidden": 1_024,
                    "pred_hidden": 640,
                },
            },
        },
        "mlx_conversion": {
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "modules": {"stt_model.embed_tokens": {"group_size": 64, "bits": 4}},
            }
        },
    }


def _base_config():
    return {
        "vocab_size": 131_072,
        "hidden_size": 4_480,
        "intermediate_size": 15_680,
        "num_hidden_layers": 56,
        "max_position_embeddings": 131_072,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "mamba_num_heads": 128,
        "mamba_head_dim": 80,
        "mamba_state_dim": 128,
        "mamba_num_groups": 8,
        "conv_kernel": 4,
        "hybrid_override_pattern": "M" * 56,
    }


def test_build_runtime_config_normalizes_source_schema():
    config = build_runtime_config(_source_config(), _base_config())
    assert config["model_type"] == "nemotron_voicechat"
    assert config["mlx_runtime_config_version"] == convert.RUNTIME_CONFIG_VERSION
    assert config["text_config"]["time_step_limit"] == [0.0, float("inf")]
    assert config["eos_token_id"] == 2
    assert config["pad_token_id"] == 12
    assert config["audio_config"]["encoder"]["att_context_size"] == [[70, 0]]
    assert config["codec_config"]["base_channels"] == 384
    assert config["codec_config"]["channel_multipliers"] == [1, 2, 4]
    assert config["codec_config"]["downsample_rates"] == [7, 7, 9]
    assert config["rnnt_vocabulary"] == ["<unk>", "▁hello"]


def test_build_runtime_config_flattens_per_module_quantization():
    config = build_runtime_config(_source_config(), _base_config())
    quantization = config["quantization"]
    assert quantization["group_size"] == 64
    assert quantization["bits"] == 4
    assert quantization["stt_model.embed_tokens"] == {
        "group_size": 64,
        "bits": 4,
    }
    assert config["quantization_config"] == quantization


def test_build_runtime_config_keeps_bf16_unquantized():
    source = _source_config()
    source["mlx_conversion"]["quantization"] = None

    config = build_runtime_config(source, _base_config())

    assert "quantization" not in config
    assert "quantization_config" not in config


def _write_artifact_inputs(tmp_path):
    source = tmp_path / "source"
    tokenizer = tmp_path / "tokenizer"
    source.mkdir()
    tokenizer.mkdir()
    (source / "config.json").write_text(json.dumps(_source_config()))
    (tokenizer / "config.json").write_text(json.dumps(_base_config()))

    shard_name = "model-00001-of-00001.safetensors"
    (source / shard_name).write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"stt_model.weight": shard_name}})
    )
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizer"})
    )
    (tokenizer / "special_tokens_map.json").write_text("{}")
    return source, tokenizer, shard_name


def test_prepare_artifact_writes_generic_load_layout(tmp_path):
    source, tokenizer, shard_name = _write_artifact_inputs(tmp_path)
    output = tmp_path / "output"

    result = convert.prepare_artifact(
        source,
        tokenizer,
        output,
        copy_weights=True,
    )

    assert result == output
    assert json.loads((output / "config.json").read_text())["model_type"] == (
        "nemotron_voicechat"
    )
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    assert tokenizer_config["fix_mistral_regex"] is True
    assert (output / "tokenizer.json").exists()
    assert (output / "model.safetensors.index.json").exists()
    assert (output / shard_name).read_bytes() == b"weights"
    assert not (output / shard_name).is_symlink()
    assert (output / "README.md").exists()


def test_prepare_artifact_can_link_weight_shards(tmp_path):
    source, tokenizer, shard_name = _write_artifact_inputs(tmp_path)
    output = tmp_path / "linked-output"

    convert.prepare_artifact(source, tokenizer, output)

    assert (output / "model.safetensors.index.json").is_symlink()
    assert (output / shard_name).is_symlink()
