"""JSONL splits preserve asset paths for CLI and notebook preprocessing."""

import json
from unittest.mock import patch

import pytest

from mlx_vlm.trainer.datasets import RawSplit, load


def test_folder_and_single_file_keep_media_base_path(tmp_path):
    row = {"images": ["images/red.png"], "question": "Which color?", "answer": "red"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    (tmp_path / "valid.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    splits = load(tmp_path)
    assert set(splits) == {"train", "valid"}
    assert isinstance(splits["train"], RawSplit)
    assert splits["train"][0] == row
    assert splits["train"].base_path == tmp_path
    assert load(path).base_path == tmp_path
    assert load(tmp_path, split="train") == [row]
    assert load(tmp_path, split="validation") == []


def test_hub_jsonl_repository_uses_snapshot_asset_paths(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    with patch(
        "huggingface_hub.snapshot_download", return_value=str(tmp_path)
    ) as download:
        splits = load("owner/dataset")
    download.assert_called_once_with(
        repo_id="owner/dataset", repo_type="dataset", allow_patterns=None
    )
    assert splits["train"].base_path == tmp_path


def test_empty_folder_reports_expected_split_names(tmp_path):
    with pytest.raises(ValueError, match="No JSONL splits"):
        load(tmp_path)
