import importlib
import json
import stat
from types import SimpleNamespace

import mlx.nn as nn


def test_convert_can_update_tokenizer_from_read_only_hub_cache(tmp_path, monkeypatch):
    converter = importlib.import_module("mlx_vlm.convert")
    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    tokenizer_file = source / "tokenizer.json"
    tokenizer_file.write_text('{"original": true}')
    tokenizer_file.chmod(0o444)
    (source / "config.json").write_text('{"model_type": "lfm2_moe"}')

    def save_processor(path):
        (path / "tokenizer.json").write_text('{"saved": true}')

    monkeypatch.setattr(converter, "get_model_path", lambda *a, **k: source)
    monkeypatch.setattr(
        converter,
        "fetch_from_hub",
        lambda *a, **k: (
            nn.Module(),
            {"model_type": "lfm2_moe"},
            SimpleNamespace(save_pretrained=save_processor),
        ),
    )
    monkeypatch.setattr(converter, "save_weights", lambda path, *a, **k: path.mkdir())
    monkeypatch.setattr(converter, "create_model_card", lambda *a, **k: None)

    converter.convert(str(source), target)

    assert json.loads((target / "tokenizer.json").read_text()) == {"saved": True}
    assert (target / "tokenizer.json").stat().st_mode & stat.S_IWUSR
    assert not tokenizer_file.stat().st_mode & stat.S_IWUSR
    assert json.loads(tokenizer_file.read_text()) == {"original": True}
