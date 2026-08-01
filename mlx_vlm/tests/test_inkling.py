import mlx.core as mx
import numpy as np

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import (
    LanguageModel,
    _restore_cache_state,
    _snapshot_cache_state,
    banded_additive_mask,
)
from mlx_vlm.speculative.drafters.inkling_mtp import InklingMTPDraftModel
from mlx_vlm.speculative.drafters.inkling_mtp.config import InklingMTPConfig


def _tiny_text_config(num_hidden_layers=2):
    return TextConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=8,
        layer_types=["hybrid_sliding", "hybrid"][:num_hidden_layers],
        d_rel=4,
        rel_extent=16,
        sconv_kernel_size=4,
        mlp_layer_types=["dense"] * num_hidden_layers,
        intermediate_size=32,
        dense_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        logits_mup_width_multiplier=1.0,
    )


def test_banded_mask_excludes_left_padding():
    rel = mx.zeros((2, 3, 1, 1))
    proj = mx.zeros((1, 4))
    mask = banded_additive_mask(
        rel,
        proj,
        mx.array(0),
        mx.array(3),
        0,
        4,
        left_padding=mx.array([0, 2]),
    )
    mx.eval(mask)

    assert mx.all(mask[1, :, :, :2] < -1e29).item()
    assert mask[1, 0, 2, 2].item() == 0.0


def test_server_batch_matches_independent_rows():
    mx.random.seed(7)
    model = LanguageModel(_tiny_text_config())
    model.eval()
    mx.eval(model.parameters())

    long_prompt = mx.random.normal((1, 3, 32))
    short_prompt = mx.random.normal((1, 1, 32))
    batch_prompt = mx.concatenate(
        [
            long_prompt,
            mx.concatenate([mx.zeros((1, 2, 32)), short_prompt], axis=1),
        ],
        axis=0,
    )

    batch_cache = _make_cache(model, [0, 2])
    long_cache = model.make_cache()
    short_cache = model.make_cache()

    batch_logits = model(inputs_embeds=batch_prompt, cache=batch_cache).logits
    long_logits = model(inputs_embeds=long_prompt, cache=long_cache).logits
    short_logits = model(inputs_embeds=short_prompt, cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, -1], long_logits[0, -1], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, -1], short_logits[0, -1], atol=3e-3).item()

    next_embeddings = mx.random.normal((2, 1, 32))
    batch_logits = model(inputs_embeds=next_embeddings, cache=batch_cache).logits
    long_logits = model(inputs_embeds=next_embeddings[:1], cache=long_cache).logits
    short_logits = model(inputs_embeds=next_embeddings[1:], cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 0], long_logits[0, 0], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()


def test_server_right_padded_prefill_preserves_conv_state():
    mx.random.seed(11)
    model = LanguageModel(_tiny_text_config())
    model.eval()
    mx.eval(model.parameters())

    long_prompt = mx.random.normal((1, 3, 32))
    short_prompt = mx.random.normal((1, 1, 32))
    batch_prompt = mx.concatenate(
        [
            long_prompt,
            mx.concatenate([short_prompt, mx.zeros((1, 2, 32))], axis=1),
        ],
        axis=0,
    )

    batch_cache = _make_cache(model, [0, 0])
    for cache in batch_cache:
        cache.prepare(right_padding=[0, 2], lengths=[3, 1])
    long_cache = model.make_cache()
    short_cache = model.make_cache()

    batch_logits = model(inputs_embeds=batch_prompt, cache=batch_cache).logits
    long_logits = model(inputs_embeds=long_prompt, cache=long_cache).logits
    short_logits = model(inputs_embeds=short_prompt, cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 2], long_logits[0, 2], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()

    for cache in batch_cache:
        cache.finalize()
    next_embeddings = mx.random.normal((2, 1, 32))
    batch_logits = model(inputs_embeds=next_embeddings, cache=batch_cache).logits
    long_logits = model(inputs_embeds=next_embeddings[:1], cache=long_cache).logits
    short_logits = model(inputs_embeds=next_embeddings[1:], cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 0], long_logits[0, 0], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()


def test_empty_batch_cache_snapshot_restores_metadata():
    cache = CacheList(BatchKVCache([0, 2]), ArraysCache(4, left_padding=[0, 2]))
    snapshot = _snapshot_cache_state([cache])

    keys = mx.ones((2, 1, 1, 4))
    cache[0].update_and_fetch(keys, keys)
    cache[1][0] = mx.ones((2, 3, 4))
    cache[1].advance(1)
    _restore_cache_state([cache], snapshot)

    assert cache[0].keys is None
    assert cache[0].offset.tolist() == [0, -2]
    assert cache[0].left_padding.tolist() == [0, 2]
    assert cache[0]._idx == 0
    assert cache[1].cache == [None] * 4
    assert cache[1].left_padding.tolist() == [0, 2]


def test_mtp_eval_state_skips_empty_kv_caches():
    config = InklingMTPConfig(
        text_config=_tiny_text_config(1),
        num_mtp_layers=2,
        mtp_local_layer_ids=[0],
    )
    drafter = InklingMTPDraftModel(config)
    drafter._cache = drafter.make_cache()

    state = drafter.draft_eval_state()

    assert len(state) == 4


def test_audio_features_match_numpy_stft():
    from mlx_vlm.models.inkling.audio_feature_extractor import (
        InklingAudioFeatureExtractor,
    )

    extractor = InklingAudioFeatureExtractor()
    rng = np.random.default_rng(5)
    waveform = rng.normal(0.0, 0.1, 2401).astype(np.float32)
    actual = np.asarray(extractor(waveform)["input_features"])[0]

    right_pad = (-len(waveform)) % extractor.hop_length
    padded = np.pad(
        waveform,
        (extractor.n_fft - extractor.hop_length, right_pad),
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, extractor.n_fft)[
        :: extractor.hop_length
    ]
    spectrum = np.fft.rfft(frames * np.asarray(extractor.window), axis=-1)
    magnitudes = np.maximum(np.abs(spectrum), 1e-10)
    mel = magnitudes @ np.asarray(extractor.mel_filters).T
    expected = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)

    assert actual.shape == (4, 80)
    assert np.allclose(actual, expected, atol=2e-6)


def test_audio_features_pad_frames_and_mask():
    from mlx_vlm.models.inkling.audio_feature_extractor import (
        InklingAudioFeatureExtractor,
    )

    extractor = InklingAudioFeatureExtractor()
    result = extractor(
        [np.ones(799, dtype=np.float32), np.ones(1600, dtype=np.float32)]
    )

    assert result["input_features"].shape == (2, 2, 80)
    assert result["input_features_mask"].tolist() == [
        [True, False],
        [True, True],
    ]
    assert mx.all(result["input_features"][0, 1] == 0).item()


def test_audio_feature_chunks_preserve_stft_frames(monkeypatch):
    from mlx_vlm.models.inkling import audio_feature_extractor as audio_module
    from mlx_vlm.models.inkling.processing_inkling import extract_dmel_bins

    rng = np.random.default_rng(19)
    waveform = rng.normal(0.0, 0.1, 8000).astype(np.float32)
    unchunked = audio_module.InklingAudioFeatureExtractor(max_frames_per_chunk=1024)(
        waveform
    )["input_features"]

    calls = []
    native_stft = audio_module.stft

    def recording_stft(segment, *args, **kwargs):
        calls.append(segment.shape[0])
        return native_stft(segment, *args, **kwargs)

    monkeypatch.setattr(audio_module, "stft", recording_stft)
    chunked = audio_module.InklingAudioFeatureExtractor(max_frames_per_chunk=3)(
        waveform
    )["input_features"]

    assert calls == [3200, 3200, 3200, 1600]
    assert mx.allclose(chunked, unchunked, atol=3e-7).item()
    assert mx.array_equal(
        extract_dmel_bins(chunked, max_frames_per_chunk=3),
        extract_dmel_bins(unchunked),
    ).item()


def test_audio_tower_chunks_embedding_gather():
    from mlx_vlm.models.inkling.audio import AudioModel
    from mlx_vlm.models.inkling.config import AudioConfig

    model = AudioModel(
        AudioConfig(
            n_mel_bins=8,
            mel_vocab_size=4,
            text_hidden_size=32,
            max_frames_per_chunk=2,
        )
    )
    mx.eval(model.parameters())
    audio_input_ids = mx.arange(64).reshape(2, 4, 8) % 4

    model.max_frames_per_chunk = 1024
    unchunked = model(audio_input_ids)
    mx.eval(unchunked)
    model.max_frames_per_chunk = 2
    chunked = model(audio_input_ids)
    mx.eval(chunked)

    assert chunked.shape == (2, 4, 32)
    assert mx.allclose(chunked, unchunked, atol=1e-6).item()


def test_dmel_quantization_boundaries():
    from mlx_vlm.models.inkling.processing_inkling import (
        DMEL_MAX_VALUE,
        DMEL_MIN_VALUE,
        dmel_bin_boundaries,
        dmel_bin_centers,
        extract_dmel_bins,
    )

    centers = dmel_bin_centers()
    midpoints = (centers[:-1] + centers[1:]) / 2
    rounded_midpoints = midpoints.astype(np.float32)
    values = np.concatenate(
        [
            [DMEL_MIN_VALUE - 1],
            centers.astype(np.float32),
            np.nextafter(rounded_midpoints, np.float32(-np.inf)),
            rounded_midpoints,
            np.nextafter(rounded_midpoints, np.float32(np.inf)),
            [DMEL_MAX_VALUE + 1],
        ]
    ).astype(np.float32)
    clipped = np.clip(values, DMEL_MIN_VALUE, DMEL_MAX_VALUE)
    expected = np.abs(clipped.astype(np.float64)[:, None] - centers).argmin(axis=1)
    actual = np.asarray(extract_dmel_bins(mx.array(values), dmel_bin_boundaries()))

    assert actual.dtype == np.int32
    assert np.array_equal(actual, expected)


def test_processor_expands_audio_tokens_from_frame_mask():
    from mlx_vlm.models.inkling.audio_feature_extractor import (
        InklingAudioFeatureExtractor,
    )
    from mlx_vlm.models.inkling.processing_inkling import (
        AUDIO_TOKEN,
        DMEL_MAX_VALUE,
        DMEL_MIN_VALUE,
        InklingProcessor,
        dmel_bin_boundaries,
    )

    class Tokenizer:
        pad_token_id = 0
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self):
            self.encoded = []

        def encode(self, text):
            self.encoded.append(text)
            return [1] * (text.count(AUDIO_TOKEN) + 1)

    class ImageProcessor:
        model_input_names = ["pixel_values"]

    processor = InklingProcessor.__new__(InklingProcessor)
    processor.feature_extractor = InklingAudioFeatureExtractor()
    processor.tokenizer = Tokenizer()
    processor.image_processor = ImageProcessor()
    processor.image_token = "<image>"
    processor.audio_token = AUDIO_TOKEN
    processor.dmel_min_value = DMEL_MIN_VALUE
    processor.dmel_max_value = DMEL_MAX_VALUE
    processor.bin_boundaries = dmel_bin_boundaries()

    result = InklingProcessor.__call__(
        processor,
        text=[f"first {AUDIO_TOKEN}", f"second {AUDIO_TOKEN}"],
        audio=[
            np.zeros(799, dtype=np.float32),
            np.zeros(1600, dtype=np.float32),
        ],
    )

    assert [text.count(AUDIO_TOKEN) for text in processor.tokenizer.encoded] == [1, 2]
    assert result["audio_input_ids"].shape == (2, 2, 80)
    assert result["audio_input_ids"].dtype == mx.int32
    assert result["audio_input_ids_mask"].tolist() == [
        [True, False],
        [True, True],
    ]
    assert result["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]


def test_from_pretrained_builds_native_audio_extractor(tmp_path, monkeypatch):
    import json

    from mlx_vlm.models.inkling import processing_inkling
    from mlx_vlm.models.inkling.audio_feature_extractor import (
        InklingAudioFeatureExtractor,
    )

    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor": {
                    "feature_extractor_type": "InklingFeatureExtractor",
                    "feature_size": 12,
                    "sampling_rate": 8000,
                    "audio_token_duration_s": 0.05,
                    "window_size_multiplier": 2.0,
                    "hop_length": 400,
                    "window_size": 800,
                    "n_fft": 800,
                }
            }
        )
    )

    class Tokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        chat_template = None
        model_input_names = ["input_ids", "attention_mask"]

    monkeypatch.setattr(
        processing_inkling.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        processing_inkling.InklingProcessor,
        "check_argument_for_proper_class",
        lambda self, name, value: type(value),
    )

    processor = processing_inkling.InklingProcessor.from_pretrained(tmp_path)

    assert isinstance(processor.feature_extractor, InklingAudioFeatureExtractor)
    assert processor.feature_extractor.feature_size == 12
    assert processor.feature_extractor.sampling_rate == 8000
    assert processor.feature_extractor.hop_length == 400
