import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.apc import APCManager
from mlx_vlm.apc_omni import OmniPrefixContext
from mlx_vlm.models.qwen3_omni_moe import Model, ModelConfig


def tiny_model():
    cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
        "cases"
    ]
    cfg = copy.deepcopy(
        next(x["config"] for x in cases if x["module"] == "qwen3_omni_moe")
    )
    cfg["thinker_config"]["audio_config"].update(
        num_mel_bins=8, downsample_hidden_size=8
    )
    return Model(ModelConfig.from_dict(cfg))


def media(kind, seed):
    mx.random.seed(seed)
    if kind == "image":
        return (
            [29] + [31] * 4 + [28],
            mx.random.normal((16, 24)),
            {"image_grid_thw": mx.array([[1, 4, 4]])},
        )
    if kind == "video":
        return (
            [29] + [30] * 8 + [28],
            None,
            {
                "pixel_values_videos": mx.random.normal((32, 24)),
                "video_grid_thw": mx.array([[2, 4, 4]]),
                "video_second_per_grid": [2.0],
                "fps": 1.0,
            },
        )
    length = 50 if seed == 1 else 150
    count = 7 if length == 50 else 20
    return (
        [27] * count,
        None,
        {
            "input_features": mx.random.normal((1, 8, length)),
            "feature_attention_mask": mx.ones((1, length), dtype=mx.int32),
        },
    )


def join(a, b):
    pixels = [x for x in (a[1], b[1]) if x is not None]
    out = {}
    for key in a[2].keys() | b[2].keys():
        values = [p[2][key] for p in (a, b) if key in p[2]]
        if key == "fps":
            out[key] = values[0]
        elif isinstance(values[0], list):
            out[key] = sum(values, [])
        else:
            if key in {"input_features", "feature_attention_mask"}:
                longest = max(v.shape[-1] for v in values)
                values = [
                    mx.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, longest - v.shape[-1])])
                    for v in values
                ]
            out[key] = mx.concatenate(values, axis=0)
    return mx.concatenate(pixels, axis=0) if pixels else None, out


@pytest.mark.parametrize(
    "before,added",
    [
        ("image", "image"),
        ("image", "video"),
        ("video", "image"),
        ("audio", "audio"),
        ("image", "audio"),
        ("audio", "video"),
    ],
)
def test_restored_media_suffix_matches_full_float32_logits(before, added):
    with mx.stream(mx.cpu):
        mx.random.seed(0)
        model = tiny_model()
        a, b = media(before, 1), media(added, 2)
        old = mx.array([[1] + a[0] + [2, 3]])
        full = mx.array([[*old[0].tolist(), 4, *b[0], 5, 6]])
        pixels, kwargs = join(a, b)
        old_ctx = OmniPrefixContext.prepare(
            model, None, old[0].tolist(), a[1], a[2], "test"
        )
        full_ctx = OmniPrefixContext.prepare(
            model, None, full[0].tolist(), pixels, kwargs, "test"
        )
        assert old_ctx is not None and full_ctx is not None
        complete = model.get_input_embeddings(full, pixels, **kwargs)
        reference = model.language_model(
            full, cache=model.language_model.make_cache(), **complete.to_dict()
        ).logits
        mx.eval(reference)
        old_embeds = model.get_input_embeddings(old, a[1], **a[2])
        cache = model.language_model.make_cache()
        model.language_model(old, cache=cache, **old_embeds.to_dict())
        mx.eval([c.state for c in cache])
        manager = APCManager(overrides={"memory_max_gb": 0.1})
        manager.exact_cache_min_tokens = 1
        assert manager.store_exact_cache(
            old[0].tolist(), cache, extra_hash=old_ctx.prefix_hash(old.shape[1])
        )
        for chunks in (False, True):
            hit = full_ctx.lookup(manager)
            assert hit is not None and hit["prefix_len"] == old.shape[1]
            suffix_pixels, suffix_kwargs = full_ctx.suffix_inputs(old.shape[1])
            positions, delta = model.language_model.get_rope_index(
                full, kwargs.get("image_grid_thw"), kwargs.get("video_grid_thw")
            )
            suffix = full[:, old.shape[1] :]
            embeds = model.get_input_embeddings(
                suffix,
                suffix_pixels,
                position_ids=positions,
                rope_deltas=delta,
                **suffix_kwargs,
            )
            data = embeds.to_dict()
            step = 3 if chunks else suffix.shape[1]
            for start in range(0, suffix.shape[1], step):
                end = min(start + step, suffix.shape[1])
                output = model.language_model(
                    suffix[:, start:end],
                    cache=hit["warm_cache"],
                    **{**data, "inputs_embeds": embeds.inputs_embeds[:, start:end]},
                ).logits
            np.testing.assert_allclose(
                np.array(output[:, -1]),
                np.array(reference[:, -1]),
                atol=2e-3,
                rtol=2e-3,
            )
        manager.clear()


def test_audio_features_do_not_depend_on_other_clip_padding():
    with mx.stream(mx.cpu):
        model = tiny_model()
        a, b = media("audio", 1), media("audio", 2)
        _, kwargs = join(a, b)
        first = model.thinker.get_audio_features(**a[2])
        together = model.thinker.get_audio_features(**kwargs)
        np.testing.assert_allclose(
            np.array(first), np.array(together[: first.shape[0]]), atol=1e-5, rtol=1e-5
        )
        old = OmniPrefixContext.prepare(
            model, None, [1] + a[0] + [2], None, a[2], "tenant"
        )
        new = OmniPrefixContext.prepare(
            model, None, [1] + a[0] + [2] + b[0] + [3], None, kwargs, "tenant"
        )
        assert old.prefix_hash(len(old.token_ids)) == new.prefix_hash(
            len(old.token_ids)
        )
        mutated = kwargs.copy()
        mutated["input_features"] = kwargs["input_features"].at[0, :, :50].add(1)
        changed = OmniPrefixContext.prepare(
            model, None, new.token_ids, None, mutated, "tenant"
        )
        assert changed.prefix_hash(len(old.token_ids)) != old.prefix_hash(
            len(old.token_ids)
        )


def test_video_fps_is_local_to_each_clip_and_interleaving_falls_back():
    with mx.stream(mx.cpu):
        model = tiny_model()
        a, b = media("video", 1), media("video", 2)
        pixels, kwargs = join(a, b)
        kwargs["fps"] = [1, 2]
        old = OmniPrefixContext.prepare(
            model, None, [1] + a[0] + [2], None, {**a[2], "fps": [1]}, "test"
        )
        new = OmniPrefixContext.prepare(
            model, None, [1] + a[0] + [2] + b[0] + [3], pixels, kwargs, "test"
        )
        assert old.prefix_hash(len(old.token_ids)) == new.prefix_hash(
            len(old.token_ids)
        )
        assert (
            OmniPrefixContext.prepare(
                model,
                None,
                new.token_ids,
                pixels,
                {**kwargs, "use_audio_in_video": True},
                "test",
            )
            is None
        )


@pytest.mark.parametrize("frames", [50, 100, 150, 200, 250])
def test_audio_tower_chunk_output_lengths(frames):
    with mx.stream(mx.cpu):
        model = tiny_model()
        features = mx.zeros((8, frames + 50))
        output = model.thinker.audio_tower(
            features, feature_lens=mx.array([frames, 50])
        )
        from mlx_vlm.models.qwen3_omni_moe.audio import _get_feat_extract_output_lengths

        assert output.shape[0] == int(_get_feat_extract_output_lengths(frames)) + 7


def test_audio_length_math_agrees_for_numpy_and_mlx():
    from mlx_vlm.models.qwen3_omni_moe.audio import (
        _get_feat_extract_output_lengths as encoder,
    )
    from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        _get_feat_extract_output_lengths as processor,
    )

    lengths = np.arange(401, dtype=np.int32)
    with mx.stream(mx.cpu):
        np.testing.assert_array_equal(
            np.array(encoder(mx.array(lengths))), processor(lengths)
        )
        assert encoder(mx.array([0, 100, 200, 300])).tolist() == [0, 13, 26, 39]


def test_audio_mask_alias_precedence_matches_model():
    with mx.stream(mx.cpu):
        model = tiny_model()
        kwargs = {
            "input_features": mx.ones((1, 8, 100)),
            "input_features_mask": mx.ones((1, 100)),
            "feature_attention_mask": mx.zeros((1, 100)),
        }
        ids = [1] + [27] * 13 + [2]
        a = OmniPrefixContext.prepare(model, None, ids, None, kwargs, "x")
        b = OmniPrefixContext.prepare(
            model, None, ids, None, {**kwargs, "feature_attention_mask": None}, "x"
        )
        assert a is not None and b is not None
        assert a.hashes == b.hashes


def test_independent_audio_preprocessing_handles_odd_samples_and_flat_lists():
    from transformers import WhisperFeatureExtractor

    from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    p = object.__new__(Qwen3OmniMoeProcessor)
    p.feature_extractor = WhisperFeatureExtractor(
        feature_size=8, return_attention_mask=True
    )
    p.image_processor = SimpleNamespace(merge_size=2)
    p.video_processor = SimpleNamespace(merge_size=2)
    p.tokenizer = lambda text, **kwargs: {
        "input_ids": np.ones((len(text), 1), dtype=np.int32)
    }
    p.audio_token = "<audio>"
    p.image_token = "<image>"
    p.video_token = "<video>"
    a = np.sin(np.arange(16159, dtype=np.float32) * 0.05)
    b = np.sin(np.arange(32017, dtype=np.float32) * 0.08)
    with mx.stream(mx.cpu):
        single = p(
            text="<audio>",
            audio=a.tolist(),
            padding=True,
            sampling_rate=16000,
            return_attention_mask=True,
        )
        multiple = p(
            text="<audio><audio>",
            audio=[a, b],
            padding=True,
            sampling_rate=16000,
            return_attention_mask=True,
        )
        n = int(single["feature_attention_mask"].sum().item())
        assert n == 100
        assert multiple["feature_attention_mask"][0].sum().item() == n
        np.testing.assert_array_equal(
            np.array(single["input_features"][0, :, :n]),
            np.array(multiple["input_features"][0, :, :n]),
        )
