from pathlib import Path

import pytest

import mlx_vlm.reranker_loader as reranker_loader


@pytest.mark.parametrize(
    "architecture",
    [
        "BertForSequenceClassification",
        "ModernBertForSequenceClassification",
        "XLMRobertaForSequenceClassification",
    ],
)
def test_identifies_sequence_classifier_architectures(architecture):
    assert reranker_loader.is_sequence_classifier_config(
        {"architectures": [architecture]}
    )


@pytest.mark.parametrize("model_type", ["bert", "modernbert", "xlm-roberta"])
def test_loads_supported_native_sequence_classifier(model_type, monkeypatch):
    config = {
        "architectures": ["ExampleForSequenceClassification"],
        "model_type": model_type,
        "num_labels": 1,
    }
    sentinel = object()
    captured = {}

    def fake_load_encoder_model(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(reranker_loader, "load_encoder_model", fake_load_encoder_model)

    model = reranker_loader.load_sequence_classification_model(
        Path("model"), config=config
    )

    assert model is sentinel
    assert captured["model_class_name"] == "SequenceClassificationModel"
    assert captured["config"] == config
    assert captured["config_overrides"] == {"num_labels": 1}


def test_rejects_multi_label_sequence_classifier():
    config = {
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "num_labels": 2,
    }

    with pytest.raises(ValueError, match="exactly one output label"):
        reranker_loader.load_sequence_classification_model(Path("model"), config=config)


def test_rejects_unsupported_sequence_classifier_family():
    config = {
        "architectures": ["DebertaV2ForSequenceClassification"],
        "model_type": "deberta-v2",
        "num_labels": 1,
    }

    with pytest.raises(ValueError, match="Unsupported reranker model type"):
        reranker_loader.load_sequence_classification_model(Path("model"), config=config)


def test_rejects_embedding_checkpoint_as_reranker(monkeypatch):
    monkeypatch.setattr(
        reranker_loader, "get_model_path", lambda *args, **kwargs: Path("model")
    )
    monkeypatch.setattr(
        reranker_loader,
        "load_config",
        lambda *args, **kwargs: {
            "architectures": ["BertModel"],
            "model_type": "bert",
        },
    )

    with pytest.raises(ValueError, match="sequence-classification checkpoints"):
        reranker_loader.load_reranker("embedding-model")
