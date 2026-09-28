"""Atomic image, audio and video prefixes for single-request Qwen3-Omni."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .apc import media_token_spans, semantic_extra_hash
from .apc_prefix import PrefixContext
from .models.qwen3_omni_moe.audio import _get_feat_extract_output_lengths


@dataclass
class OmniPrefixContext(PrefixContext):
    # (kind, source index, start patch/frame, end patch/frame), in token order.
    items: list[tuple[str, int, int, int]]
    pixel_values: Any
    kwargs: dict

    @classmethod
    def prepare(cls, model, processor, token_ids, pixel_values, kwargs, tenant):
        if model.config.model_type != "qwen3_omni_moe" or kwargs.get(
            "use_audio_in_video"
        ):
            return None
        config = model.config.thinker_config
        merge = config.vision_config.spatial_merge_size
        entries = []
        for kind, token, values, grid in (
            (
                "image",
                config.image_token_id,
                pixel_values,
                kwargs.get("image_grid_thw"),
            ),
            (
                "video",
                config.video_token_id,
                kwargs.get("pixel_values_videos"),
                kwargs.get("video_grid_thw"),
            ),
        ):
            spans = media_token_spans(token_ids, {token})
            if not spans:
                if values is not None or grid is not None:
                    return None
                continue
            if (
                values is None
                or grid is None
                or values.ndim != 2
                or grid.shape != (len(spans), 3)
            ):
                return None
            offset = 0
            for i, ((start, end), row) in enumerate(zip(spans, grid.tolist())):
                if any(type(n) is not int or n <= 0 for n in row):
                    return None
                count = row[0] * row[1] * row[2]
                if (
                    (kind == "image" and row[0] != 1)
                    or row[1] % merge
                    or row[2] % merge
                    or count // (merge * merge) != end - start
                ):
                    return None
                if offset + count > values.shape[0]:
                    return None
                payload = {
                    "kind": kind,
                    "pixels": values[offset : offset + count],
                    "grid": grid[i],
                    "span": mx.array([start, end]),
                }
                if kind == "video":
                    fps = kwargs.get("fps")
                    if isinstance(fps, (list, tuple)):
                        if len(fps) != len(spans):
                            return None
                        fps = fps[i]
                    payload["fps"] = fps
                    seconds = kwargs.get("video_second_per_grid")
                    if seconds is not None:
                        if len(seconds) != len(spans):
                            return None
                        payload["seconds"] = seconds[i]
                entries.append((start, end, (kind, i, offset, offset + count), payload))
                offset += count
            if offset != values.shape[0]:
                return None

        spans = media_token_spans(token_ids, {config.audio_token_id})
        features = kwargs.get("input_features")
        mask = kwargs.get("input_features_mask")
        if mask is None:
            mask = kwargs.get("feature_attention_mask")
        if spans:
            if (
                features is None
                or features.ndim != 3
                or features.shape[0] != len(spans)
            ):
                return None
            lengths = kwargs.get("audio_feature_lengths")
            if lengths is None:
                if mask is None or mask.ndim != 2 or mask.shape[0] != len(spans):
                    return None
                lengths = mask.sum(-1)
                if mask.shape[-1] > features.shape[-1]:
                    lengths = lengths // (mask.shape[-1] // features.shape[-1])
            if lengths.shape != (len(spans),):
                return None
            for i, ((start, end), length) in enumerate(zip(spans, lengths.tolist())):
                length = int(length)
                if (
                    length <= 0
                    or length > features.shape[-1]
                    or int(_get_feat_extract_output_lengths(length)) != end - start
                ):
                    return None
                payload = {
                    "kind": "audio",
                    "features": features[i, :, :length],
                    "length": str(length),
                    "span": mx.array([start, end]),
                }
                entries.append((start, end, ("audio", i, 0, length), payload))
        elif (
            features is not None
            or mask is not None
            or kwargs.get("audio_feature_lengths") is not None
        ):
            return None
        entries.sort(key=lambda x: x[0])
        if any(a[1] > b[0] for a, b in zip(entries, entries[1:])):
            return None
        hashes = [
            semantic_extra_hash(
                tenant=tenant,
                model=model,
                processor=processor,
                media={"omni_prefix_schema": "v1"},
            )
        ]
        for _, _, _, payload in entries:
            hashes.append(semantic_extra_hash(image_hash=hashes[-1], media=payload))
        return cls(
            list(token_ids),
            tuple((a, b) for a, b, _, _ in entries),
            hashes,
            [item for _, _, item, _ in entries],
            pixel_values,
            kwargs.copy(),
        )

    def suffix_inputs(self, prefix_len):
        self.prefix_hash(prefix_len)
        completed = {"image": 0, "video": 0, "audio": 0}
        for (_, end), (kind, _, _, _) in zip(self.spans, self.items):
            if end <= prefix_len:
                completed[kind] += 1
        out = {}
        pixels = None
        for kind, values, grid_key, pixel_key in (
            ("image", self.pixel_values, "image_grid_thw", None),
            (
                "video",
                self.kwargs.get("pixel_values_videos"),
                "video_grid_thw",
                "pixel_values_videos",
            ),
        ):
            items = [x for x in self.items if x[0] == kind]
            n = completed[kind]
            if n < len(items):
                part = values[items[n][2] :]
                out[grid_key] = self.kwargs[grid_key][n:]
                if pixel_key:
                    out[pixel_key] = part
                else:
                    pixels = part
                if (
                    kind == "video"
                    and self.kwargs.get("video_second_per_grid") is not None
                ):
                    out["video_second_per_grid"] = self.kwargs["video_second_per_grid"][
                        n:
                    ]
        audio_items = [x for x in self.items if x[0] == "audio"]
        n = completed["audio"]
        if n < len(audio_items):
            for key in (
                "input_features",
                "feature_attention_mask",
                "input_features_mask",
                "audio_feature_lengths",
            ):
                if self.kwargs.get(key) is not None:
                    out[key] = self.kwargs[key][n:]
        return pixels, out
