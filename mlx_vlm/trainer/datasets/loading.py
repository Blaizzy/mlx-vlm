"""Shared dataset loading and caching helpers.

Each modality's dataset module implements raw sample containers and
preprocessing on top of these helpers, so a local
folder and a Hugging Face repository look identical to the dataset classes.
The loading pattern mirrors the ``mlx-lm-lora`` ``trainer/datasets.py``
reference: datasets are folders holding one ``<split>.jsonl`` per split,
either on disk or inside a Hub snapshot.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

# Accept both common validation split names.
DEFAULT_SPLIT_NAMES = ("train", "validation", "valid", "test")


class RawSplit(list):
    """Raw JSONL rows for one split, carrying their source folder.

    ``base_path`` lets dataset classes resolve relative asset paths (for
    example ``audio/train/000001.wav``) against the folder the rows were
    loaded from. It behaves like a plain ``list`` everywhere else.
    """

    def __init__(self, rows, base_path: Optional[Path] = None):
        super().__init__(rows)
        self.base_path = Path(base_path) if base_path is not None else None


class CacheDataset:
    """Lazy preprocessing cache shared by every dataset implementation.

    The wrapped dataset must provide ``process(raw_sample)``; processed
    samples are cached in memory after their first access. ``max_size`` bounds
    the LRU cache (0 disables it, None keeps every processed sample).
    Set ``process=False`` when indexing already prepares the sample.
    ``itemlen`` returns
    the length used for length-grouped batching: ``token_ids`` when the
    processed sample carries them, otherwise the processed sample itself.
    """

    def __init__(self, data, max_size: int | None = None, *, process: bool = True):
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be nonnegative or None.")
        self._data = data
        self._process = data.process if process else lambda sample: sample
        self.max_size = max_size
        self._processed = OrderedDict()
        self._lengths: dict[int, int] = {}

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self._processed:
            self._processed.move_to_end(index)
            return self._processed[index]
        sample = self._process(self._data[index])
        if self.max_size != 0:
            self._processed[index] = sample
            if self.max_size is not None and len(self._processed) > self.max_size:
                self._processed.popitem(last=False)
        return sample

    def itemlen(self, index: int) -> int:
        if index not in self._lengths:
            self._lengths[index] = self._itemlen(index)
        return self._lengths[index]

    def _itemlen(self, index: int) -> int:
        source_itemlen = getattr(self._data, "itemlen", None)
        if source_itemlen is not None:
            return int(source_itemlen(index))

        processed = self[index]
        token_ids = getattr(processed, "token_ids", None)
        if token_ids is not None:
            return len(token_ids)
        if isinstance(processed, tuple) and processed:
            # Contrastive samples are ``(anchor, positive, negative)``.
            return len(processed[0])
        if isinstance(processed, dict):
            if "input_ids" in processed:
                return int(processed["input_ids"].shape[-1])
            if "chosen" in processed:
                return max(
                    int(processed[key]["input_ids"].shape[-1])
                    for key in ("chosen", "rejected")
                )
        return len(processed)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name):
        return getattr(self._data, name)


class ConcatenatedDataset:
    """Concatenate sources while leaving their stored rows unchanged."""

    def __init__(self, data: Sequence[Any]):
        self._data = list(data)
        self._offsets = []
        self._len = 0
        for dataset in self._data:
            self._offsets.append(self._len)
            self._len += len(dataset)

    def _resolve(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError(index)
        source = bisect_right(self._offsets, index) - 1
        return source, index - self._offsets[source]

    def __getitem__(self, index: int):
        source, local_index = self._resolve(index)
        datum = self._data[source][local_index]
        if isinstance(datum, Mapping):
            datum = dict(datum, _dataset=source)
        return datum

    def process(self, datum):
        if isinstance(datum, Mapping) and "_dataset" in datum:
            datum = dict(datum)
            source = int(datum.pop("_dataset"))
            return self._data[source].process(datum)
        return datum

    def __len__(self) -> int:
        return self._len


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of row dictionaries."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hf_repo_snapshot(
    data_id: str,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Path:
    """Download a Hugging Face dataset repository snapshot.

    Working from the canonical repository files (instead of the Hub's
    auto-generated Parquet representation) preserves relative asset paths,
    which audio, image, and diffusion datasets need.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Loading Hugging Face datasets requires `huggingface_hub`."
        ) from error

    try:
        snapshot_path = snapshot_download(
            repo_id=data_id,
            repo_type="dataset",
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except Exception as error:
        raise ValueError(
            f"Could not download Hugging Face dataset {data_id!r}."
        ) from error

    return Path(snapshot_path)


def load(
    source: Any,
    split: Optional[str] = None,
    split_names: Sequence[str] = DEFAULT_SPLIT_NAMES,
    allow_patterns: Optional[Sequence[str]] = None,
):
    """Load raw dataset rows from a local path or a Hugging Face repository.

    ``source`` is treated as a local file or folder when it exists and as a
    Hub dataset repository identifier otherwise:

    - a local ``.jsonl`` file returns a single :class:`RawSplit`;
    - a folder (or Hub repo) returns ``{split_name: RawSplit}`` built from
      its ``<split>.jsonl`` files, or just the requested ``split``.

    Every returned split keeps its source folder in ``base_path`` so dataset
    classes can resolve relative asset paths. For repositories that are not
    JSONL-shaped, load rows with the Hugging Face ``datasets`` package
    directly and pass them to the dataset class instead.
    """
    source_path = Path(source)

    if source_path.exists():
        base_path = source_path if source_path.is_dir() else source_path.parent
    else:
        print(f"Loading Hugging Face dataset {source}.")
        base_path = hf_repo_snapshot(
            str(source),
            allow_patterns=allow_patterns,
        )
        source_path = base_path

    if source_path.is_file():
        return RawSplit(load_jsonl(source_path), base_path)

    found: dict[str, RawSplit] = {}
    for name in split_names:
        jsonl_path = source_path / f"{name}.jsonl"
        if jsonl_path.exists():
            found[name] = RawSplit(load_jsonl(jsonl_path), base_path)

    if not found:
        raise ValueError(
            f"No JSONL splits ({', '.join(split_names)}) found in "
            f"{source_path}. Load this dataset with the Hugging Face "
            "`datasets` package and pass the rows to the dataset class."
        )

    if split is not None:
        return found.get(split, RawSplit([], base_path))

    return found
