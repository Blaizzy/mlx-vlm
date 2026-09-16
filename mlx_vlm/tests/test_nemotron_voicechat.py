"""Nemotron VoiceChat runtime and checkpoint conversion."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import load
from mlx_vlm.models.gemma3.config import TextConfig as Gemma3TextConfig
from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.nemotron_voicechat import convert
from mlx_vlm.models.nemotron_voicechat.config import CharacterEncoderConfig, MoGConfig
from mlx_vlm.models.nemotron_voicechat.convert import build_runtime_config
from mlx_vlm.models.nemotron_voicechat.model import Model
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession
from mlx_vlm.models.nemotron_voicechat.streaming import (
    VoiceChatFrameTiming,
    VoiceChatProfile,
)
from mlx_vlm.models.nemotron_voicechat.tts import CharAwareSubwordEncoder, MoGHead

# Runtime and streaming


def test_streaming_profile_summarizes_synchronized_stage_timings():
    profile = VoiceChatProfile(
        frame_duration_ms=80.0,
        frames=[
            VoiceChatFrameTiming(0, 10, 2, 20, 30, 8, 72),
            VoiceChatFrameTiming(1, 8, 1, 16, 24, 6, 56),
            VoiceChatFrameTiming(2, 9, 1, 18, 27, 7, 63),
        ],
    )
    summary = profile.summary(drop_first=1)

    assert summary["frames"] == 2
    assert summary["dropped_cold_frames"] == 1
    assert summary["stages"]["total"]["mean_ms"] == 59.5
    assert summary["processing_frames_per_second"] == pytest.approx(1000 / 59.5)
    assert summary["realtime_factor"] == pytest.approx(59.5 / 80)


def test_character_aware_encoder_prepares_and_scatters_subwords():
    config = CharacterEncoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        char_vocab_size=4,
    )
    encoder = CharAwareSubwordEncoder(config, out_size=8, vocab_size=6)
    encoder.set_vocabulary({"a": 0, "b": 1, "c": 2, "ab": 3, "<s>": 4, "</s>": 5})
    ids = mx.array([[3, 4, 5]], dtype=mx.int32)
    mask = mx.array([[True, False, True]])
    output = encoder(ids, mask)
    mx.eval(output)
    assert output.shape == (1, 3, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_mog_head_inference_shapes_and_finite_values():
    config = MoGConfig(
        intermediate_size=16, low_rank=2, num_layers=1, num_predictions=4
    )
    head = MoGHead(hidden_size=8, out_size=4, config=config)
    inputs = mx.zeros((2, 1, 8))
    mean, logs = head.infer(inputs, guidance_scale=0.2, top_p=0.95)
    mx.eval(mean, logs)
    assert mean.shape == (1, 1, 4)
    assert logs.shape == (1, 1, 1)
    assert bool(mx.all(mx.isfinite(mean)))
    assert bool(mx.all(mx.isfinite(logs)))


def test_gemma3_can_preserve_caller_supplied_embedding_scale():
    config = Gemma3TextConfig(
        model_type="gemma3_text",
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=8,
        sliding_window_pattern=1,
    )

    class Capture:
        def __call__(self, inputs, mask=None, cache=None):
            del mask, cache
            self.inputs = inputs
            return inputs

    class Identity:
        def __call__(self, inputs):
            return inputs

    model = Gemma3Model(config, scale_inputs_embeds=False)
    capture = Capture()
    model.layers = [capture]
    model.norm = Identity()
    inputs = mx.ones((1, 1, config.hidden_size))
    output = model(None, inputs_embeds=inputs)

    assert mx.array_equal(capture.inputs, inputs)
    assert mx.array_equal(output, inputs)


def test_model_creates_session_from_wrapped_tokenizer():
    vocabulary = {"hello": 0}

    class Tokenizer:
        def encode(self, *_args, **_kwargs):
            return [0]

        def decode(self, *_args, **_kwargs):
            return "hello"

        def get_vocab(self):
            return vocabulary

    class TTS:
        def set_vocabulary(self, value):
            self.vocabulary = value

    tts = TTS()
    model = SimpleNamespace(tts_model=SimpleNamespace(tts_model=tts))
    processor = SimpleNamespace(tokenizer=Tokenizer())

    created = Model.create_session(model, processor)

    assert isinstance(created, VoiceChatSession)
    assert created.model is model
    assert created.tokenizer is processor.tokenizer
    assert tts.vocabulary == vocabulary


def test_model_create_session_requires_tokenizer_interface():
    with pytest.raises(TypeError, match="tokenizer-like processor"):
        Model.create_session(SimpleNamespace(), object())


@pytest.mark.skipif(
    not os.environ.get("VOICECHAT_MODEL_PATH"),
    reason="set VOICECHAT_MODEL_PATH to run the real-checkpoint smoke test",
)
def test_real_checkpoint_offline_smoke():
    model_path = Path(os.environ["VOICECHAT_MODEL_PATH"])
    audio_path = os.environ.get("VOICECHAT_AUDIO_PATH")
    if not audio_path:
        pytest.skip("set VOICECHAT_AUDIO_PATH to an input wav")
    model, processor = load(str(model_path), lazy=True)
    result = model.create_session(processor).generate(
        audio_path, system_prompt="Answer briefly.", max_frames=2
    )
    mx.eval(result.audio)
    assert result.audio.shape == (3528,)
    assert result.audio_codes.shape == (2, 31)
    assert result.sample_rate == 22_050
    assert bool(mx.all(mx.isfinite(result.audio)))


@pytest.mark.skipif(
    not os.environ.get("VOICECHAT_MODEL_PATH"),
    reason="set VOICECHAT_MODEL_PATH to run the real-checkpoint smoke test",
)
def test_real_checkpoint_streaming_first_frame_matches_offline():
    from mlx_audio.stt.utils import load_audio

    model_path = Path(os.environ["VOICECHAT_MODEL_PATH"])
    audio_path = os.environ.get("VOICECHAT_AUDIO_PATH")
    if not audio_path:
        pytest.skip("set VOICECHAT_AUDIO_PATH to an input wav")
    model, processor = load(str(model_path), lazy=True)
    session = model.create_session(processor)
    audio = load_audio(audio_path, sr=16_000).squeeze()[:1280]
    offline = session.generate(
        audio, system_prompt="Answer briefly.", max_frames=2, seed=0
    )
    stream = session.create_streaming_session(
        system_prompt="Answer briefly.", seed=0, max_streaming_seconds=1.0
    )
    events = stream.push_audio(audio, sample_rate=16_000)
    audio_event = next(event for event in events if event.kind == "audio")

    assert stream._text_tokens[-1] == int(offline.text_tokens[0])
    assert stream._function_tokens[-1] == int(offline.function_tokens[0])
    assert mx.array_equal(audio_event.audio_codes, offline.audio_codes[0])
    assert mx.allclose(audio_event.samples, offline.audio[:1764], atol=2e-4)


# Checkpoint conversion


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
                "data": {"audio_prompt_duration": 3.0, "target_sample_rate": 22_050},
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


def test_build_runtime_config_preserves_finite_time_step_limit():
    base = _base_config()
    base["time_step_limit"] = [0.001, 0.1]

    config = build_runtime_config(_source_config(), base)

    assert config["text_config"]["time_step_limit"] == [0.001, 0.1]


def test_build_runtime_config_rejects_non_finite_time_step_limit():
    base = _base_config()
    base["time_step_limit"] = [0.0, float("inf")]

    with pytest.raises(ValueError, match="two finite numbers"):
        build_runtime_config(_source_config(), base)


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

    result = convert.prepare_artifact(source, tokenizer, output, copy_weights=True)

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
