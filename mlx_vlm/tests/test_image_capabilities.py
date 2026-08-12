from __future__ import annotations

import json

import pytest

from mlx_vlm.generate.capabilities import model_capabilities
from mlx_vlm.generate.edit_image import is_image_edit_model
from mlx_vlm.generate.image import is_image_generation_model

FLUX2_IDS = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9b-kv",
    # Quantized re-releases use the same FLUX.2-klein weights; the registry
    # must resolve them via size-marker scan, not exact alias lookup.
    "Runpod/FLUX.2-klein-4B-mflux-4bit",
)


@pytest.mark.parametrize("model_id", FLUX2_IDS)
def test_model_capabilities_flux2_generation_and_editing(model_id: str) -> None:
    caps = model_capabilities(model_id)
    assert caps == ["image_generation", "image_editing"]
    assert is_image_generation_model(model_id)
    assert is_image_edit_model(model_id)


def test_model_capabilities_mage_flow_edit_only() -> None:
    # Edit variants generate nothing; edit capability only.
    assert model_capabilities("mage-flow-edit-base") == ["image_editing"]
    assert model_capabilities("microsoft/Mage-Flow-Edit-Turbo") == ["image_editing"]
    assert is_image_edit_model("mage-flow-edit-base")
    assert not is_image_generation_model("mage-flow-edit")


def test_model_capabilities_mage_flow_generation_only() -> None:
    assert model_capabilities("mage-flow-turbo") == ["image_generation"]
    assert is_image_generation_model("mage-flow-turbo")
    assert not is_image_edit_model("mage-flow-turbo")


@pytest.mark.parametrize(
    "model_id",
    ["unknown/not-a-model", "org/repo-with-lora", None],
)
def test_model_capabilities_unknown_models_report_empty(model_id: str | None) -> None:
    assert model_capabilities(model_id) == []


def test_model_capabilities_unknown_local_path() -> None:
    assert model_capabilities("/nonexistent/local/snapshot") == []


def _write_config(snapshot_path, config: dict) -> str:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    (snapshot_path / "config.json").write_text(json.dumps(config))
    return str(snapshot_path)


def _write_tokenizer_config(snapshot_path, chat_template) -> str:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    (snapshot_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": chat_template})
    )
    return str(snapshot_path)


def test_model_capabilities_text_only_from_config(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    assert model_capabilities("org/qwen2", snapshot_path=snapshot) == [
        "text_generation"
    ]


def test_model_capabilities_tools_from_template(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    _write_tokenizer_config(
        snapshot, "{% if tool_call %}<tool_call>\n<function={{ tool_call.name }}"
    )
    assert model_capabilities("org/qwen2", snapshot_path=snapshot) == [
        "text_generation",
        "tools",
    ]


def test_model_capabilities_tools_from_template_list(tmp_path) -> None:
    # Newer repos store chat_template as a list of {name, template} entries.
    snapshot = _write_config(tmp_path / "gemma", {"model_type": "gemma3"})
    _write_tokenizer_config(
        snapshot,
        [{"name": "default", "template": "<|tool_call|>{{ tool }}"}],
    )
    assert model_capabilities("org/gemma", snapshot_path=snapshot) == [
        "text_generation",
        "tools",
    ]


def test_model_capabilities_no_tools_without_markers(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "llama", {"model_type": "llama3.2"})
    _write_tokenizer_config(snapshot, "{{ message.content }}")
    assert model_capabilities("org/llama", snapshot_path=snapshot) == [
        "text_generation"
    ]


def test_model_capabilities_vlm_from_config(tmp_path) -> None:
    snapshot = _write_config(
        tmp_path / "llava",
        {"model_type": "llava", "vision_config": {"model_type": "clip_vit"}},
    )
    assert model_capabilities("org/llava", snapshot_path=snapshot) == [
        "text_generation",
        "vision",
    ]


def test_model_capabilities_audio_from_config(tmp_path) -> None:
    # Audio model types live in mlx_audio (no mlx_vlm.models module); the
    # audio_config alone is the capability signal.
    snapshot = _write_config(
        tmp_path / "bark", {"model_type": "bark", "audio_config": {}}
    )
    assert model_capabilities("org/bark", snapshot_path=snapshot) == ["audio"]


def test_model_capabilities_embeddings_from_config(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "roberta", {"model_type": "xlm-roberta"})
    assert model_capabilities("org/roberta", snapshot_path=snapshot) == ["embeddings"]


def test_model_capabilities_embedding_only_arch(tmp_path) -> None:
    # bert-class encoders (bge-micro etc.) embed but never claim chat text.
    snapshot = _write_config(tmp_path / "bge", {"model_type": "bert"})
    assert model_capabilities("org/bge-micro", snapshot_path=snapshot) == [
        "embeddings"
    ]


def test_model_capabilities_lm_with_embed_remap(tmp_path) -> None:
    # qwen3 is a chat LM the server can also serve embeddings from.
    snapshot = _write_config(tmp_path / "qwen3", {"model_type": "qwen3"})
    assert model_capabilities("org/qwen3", snapshot_path=snapshot) == [
        "embeddings",
        "text_generation",
    ]


def test_model_capabilities_image_flags_suppress_text(tmp_path) -> None:
    # A diffusion model config must not gain text_generation even though
    # get_model_and_args would resolve its type module; a tool-ish template
    # must not add tools either.
    snapshot = _write_config(tmp_path / "flux2", {"model_type": "flux2"})
    _write_tokenizer_config(snapshot, "<tool_call>\n<function=")
    caps = model_capabilities(
        "black-forest-labs/FLUX.2-klein-4B", snapshot_path=snapshot
    )
    assert caps == ["image_generation", "image_editing"]


def test_model_capabilities_reasoning_from_template(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    _write_tokenizer_config(
        snapshot,
        "{% if thought %} thinking{{ thought }} response{% endif %}",
    )
    assert model_capabilities("org/qwen2", snapshot_path=snapshot) == [
        "reasoning",
        "text_generation",
    ]


def test_model_capabilities_no_reasoning_without_markers(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    _write_tokenizer_config(snapshot, "{{ message.content }}")
    assert model_capabilities("org/qwen2", snapshot_path=snapshot) == [
        "text_generation"
    ]


def test_model_capabilities_drafter_from_namespace(tmp_path) -> None:
    # eagle3 resolves into mlx_vlm.speculative.drafters on get_model_and_args.
    snapshot = _write_config(tmp_path / "eagle3", {"model_type": "eagle3"})
    caps = model_capabilities("org/eagle3", snapshot_path=snapshot)
    assert "drafter" in caps


def test_model_capabilities_video_from_class_hook(tmp_path) -> None:
    # inkling's Model class declares prepare_video_frame_pairs.
    snapshot = _write_config(tmp_path / "inkling", {"model_type": "inkling"})
    caps = model_capabilities("org/inkling", snapshot_path=snapshot)
    assert "video" in caps


def test_model_capabilities_video_from_loaded_flag(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    caps = model_capabilities(
        "org/qwen2",
        snapshot_path=snapshot,
        supports_video_input=True,
    )
    assert caps == ["text_generation", "video"]


def test_model_capabilities_kind_hint_audio_tts() -> None:
    assert model_capabilities("any/model", kind_hint="audio_tts") == [
        "audio",
        "text_to_speech",
    ]


def test_model_capabilities_kind_hint_audio_stt() -> None:
    assert model_capabilities("any/model", kind_hint="audio_stt") == [
        "audio",
        "speech_to_text",
    ]


def test_model_capabilities_kind_hint_embedding() -> None:
    assert model_capabilities("any/model", kind_hint="embedding") == [
        "embeddings"
    ]
