"""Paired media preprocessing and batching for VLM preferences."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from mlx_vlm.trainer.common.utils import iterate_batch_indices
from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
from mlx_vlm.trainer.vlm.sft.runtime import (
    _image_token_id,
    _sample_signature,
    collate_vlm_batch,
)


def _decode_json_array(value: Any) -> Any:
    if not isinstance(value, str) or not value.lstrip().startswith("["):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    if isinstance(decoded, list) and all(
        isinstance(item, Mapping)
        or (isinstance(item, str) and item.lstrip().startswith("{"))
        for item in decoded
    ):
        return decoded
    return value


class PreferenceVisionDataset:
    """Prepare chosen/rejected VLM conversations with one shared media row.

    Accepted layouts include full ``chosen``/``rejected`` chat histories, or a
    shared ``prompt``/``messages`` field plus string completions. Images and
    other processor-supported media live on the top-level row and are attached
    to both candidates.
    """

    def __init__(
        self,
        data: Any,
        model_config: Any,
        processor: Any,
        config: Any | None = None,
        *,
        prompt_feature: str = "prompt",
        chosen_feature: str = "chosen",
        rejected_feature: str = "rejected",
        base_path: str | Path | None = None,
    ) -> None:
        if isinstance(data, Mapping):
            raise TypeError("PreferenceVisionDataset expects one dataset split.")
        if not hasattr(data, "__len__") or not hasattr(data, "__getitem__"):
            raise TypeError("PreferenceVisionDataset needs finite, indexable rows.")
        values = (
            {}
            if config is None
            else (dict(config) if isinstance(config, Mapping) else vars(config).copy())
        )
        values["train_on_completions"] = True
        self.dataset = data
        self.prompt_feature = prompt_feature
        self.chosen_feature = chosen_feature
        self.rejected_feature = rejected_feature
        self.preprocessor = VisionDataset(
            data,
            model_config,
            processor,
            config=values,
            base_path=base_path,
        )
        self.config = self.preprocessor.config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.process(self.dataset[index])

    def media_signature(self, index: int) -> tuple[str, ...]:
        return self.preprocessor.media_signature(index)

    def _prompt_messages(self, row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        value = row.get(self.prompt_feature)
        if value is None:
            value = row.get(self.preprocessor.message_feature)
        if value is None:
            value = row.get("messages", row.get("conversations"))
        if value is None:
            return None
        value = _decode_json_array(value)
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        return self.preprocessor._conversation({"messages": value})

    def _candidate_messages(
        self,
        value: Any,
        *,
        prompt: list[dict[str, Any]] | None,
        feature: str,
    ) -> list[dict[str, Any]]:
        if isinstance(value, str):
            if prompt is None:
                value = _decode_json_array(value)
                if not isinstance(value, str):
                    return self._candidate_messages(
                        value,
                        prompt=prompt,
                        feature=feature,
                    )
                raise ValueError(
                    f"Preference row has string {feature!r} but no shared prompt. "
                    "Provide `prompt`, `messages`, or full candidate conversations."
                )
            return [*prompt, {"role": "assistant", "content": value}]
        if isinstance(value, Mapping):
            if "messages" in value or "conversations" in value:
                value = value.get("messages", value.get("conversations"))
            elif "role" in value or "from" in value or "speaker" in value:
                value = [value]
            else:
                raise ValueError(
                    f"Preference feature {feature!r} must be a chat message or list."
                )
        if not isinstance(value, list) or not value:
            raise ValueError(f"Preference feature {feature!r} must be non-empty.")
        messages = self.preprocessor._conversation({"messages": value})
        if (
            prompt is not None
            and len(messages) == 1
            and messages[0]["role"] == "assistant"
        ):
            messages = [*prompt, *messages]
        elif len(messages) == 1 and messages[0]["role"] == "assistant":
            raise ValueError(
                f"Preference feature {feature!r} contains only a response; provide "
                "a shared prompt or the full conversation."
            )
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                f"Preference feature {feature!r} must end with an assistant response."
            )
        return messages

    def process(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError("Preference dataset rows must be mappings.")
        if self.chosen_feature not in row or self.rejected_feature not in row:
            raise ValueError(
                "Each preference row must contain configured chosen and rejected fields."
            )
        prompt = self._prompt_messages(row)
        keep_fields = {
            key
            for _, configured, aliases in self.preprocessor._media_fields
            for key in (configured, *aliases)
            if key
        }
        media_row = {key: row[key] for key in keep_fields if key in row}
        result = {}
        for name, feature in (
            ("chosen", self.chosen_feature),
            ("rejected", self.rejected_feature),
        ):
            messages = self._candidate_messages(
                row[feature], prompt=prompt, feature=feature
            )
            candidate_row = dict(media_row)
            candidate_row["messages"] = messages
            prepared = self.preprocessor.process(candidate_row)
            if (
                "completion_mask" not in prepared
                or int(prepared["completion_mask"].sum().item()) == 0
            ):
                raise ValueError(
                    f"Preference candidate {feature!r} has no response tokens."
                )
            result[name] = prepared
        return result


def iterate_vlm_preference_batches(
    dataset: PreferenceVisionDataset,
    batch_size: int,
    max_seq_length: int,
    *,
    train: bool = False,
    pad_to_multiple: int = 32,
    pad_token_id: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    seed: int | None = None,
) -> Iterable[dict[str, dict[str, Any]]]:
    """Use the same homogeneous, distributed batching as supervised VLMs."""
    if rank is None or world_size is None:
        world = mx.distributed.init()
        rank, world_size = world.rank(), world.size()
    length = getattr(dataset, "itemlen", None)
    if length is None:

        def length(index):
            pair = dataset[index]
            return max(
                pair[key]["input_ids"].shape[-1] for key in ("chosen", "rejected")
            )

    signature = getattr(dataset, "media_signature", None)
    if signature is None:

        def signature(index):
            return _sample_signature(dataset[index]["chosen"])

    batches = iterate_batch_indices(
        len(dataset),
        batch_size,
        length_key=length,
        group_key=signature,
        train=train,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    for indices in batches:
        rows = [dataset[index] for index in indices]
        yield {
            candidate: collate_vlm_batch(
                [row[candidate] for row in rows],
                max_seq_length,
                pad_to_multiple=pad_to_multiple,
                pad_token_id=pad_token_id,
                image_token_id=_image_token_id(dataset),
            )
            for candidate in ("chosen", "rejected")
        }
