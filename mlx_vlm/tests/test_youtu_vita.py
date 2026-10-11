"""Youtu-Parsing-Omni (``youtu_vita``): conversion, multimodal embeddings, processor.

The shared ``forward``/``language``/``input_embeddings`` contracts live in
``model_cases.json``; this module covers what is specific to the port. Nothing here
downloads weights or tokenizers: models use a tiny random config and the processor
a character-level stand-in tokenizer.
"""

from __future__ import annotations

import io
import sys
import wave

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.models.youtu_vita import Model, ModelConfig
from mlx_vlm.models.youtu_vita import processing_youtu_vita as proc
from mlx_vlm.models.youtu_vita.omni import audio_output_length
from mlx_vlm.tests.test_models import DATA, ModelChecks, build_config
from mlx_vlm.utils import get_model_and_args

CASE = next(case for case in DATA["cases"] if case["id"] == "youtu_vita")
IMAGE_ID, AUDIO_ID = CASE["config"]["image_token_id"], CASE["config"]["audio_token_id"]


def tiny_model(**overrides):
    import importlib

    module = importlib.import_module("mlx_vlm.models.youtu_vita")
    config = build_config(module, {**CASE["config"], **overrides})
    mx.random.seed(3)
    model = Model(config)
    model.eval()
    return model


def randomize(model):
    """Nonuniform weights, so layout / value mistakes cannot cancel out."""
    mx.random.seed(11)
    params = tree_flatten(model.parameters())
    model.load_weights(
        [(k, mx.random.normal(v.shape).astype(v.dtype) * 0.2) for k, v in params],
        strict=False,
    )
    return model


def to_checkpoint(model):
    """Invert ``Model.sanitize``: tiny MLX parameters -> the PyTorch/HF key layout."""
    t = model.config.text_config
    params = flat(model.parameters())
    weights = {}
    for key, value in params.items():
        if key.startswith("omni_model."):
            if "audio_embeddings.conv2d" in key and key.endswith(".weight"):
                value = value.transpose(
                    0, 3, 1, 2
                )  # MLX [O,kH,kW,I] -> torch [O,I,kH,kW]
            weights["model." + key] = value
        elif key.startswith("language_model.lm_head."):
            weights[key[len("language_model.") :]] = value
        elif key.endswith((".embed_q.weight", ".unembed_out.weight")):
            continue  # rebuilt below as one kv_b_proj
        else:
            weights["model.language_model." + key[len("language_model.model.") :]] = (
                value
            )
    for i in range(t.num_hidden_layers):
        p = f"language_model.model.layers.{i}.self_attn"
        k_part = params[f"{p}.embed_q.weight"].swapaxes(-1, -2)  # [H, nope, r]
        v_part = params[f"{p}.unembed_out.weight"]  # [H, v, r]
        heads = t.num_attention_heads
        weights[f"model.language_model.layers.{i}.self_attn.kv_b_proj.weight"] = (
            mx.concatenate([k_part, v_part], axis=1).reshape(
                heads * (t.qk_nope_head_dim + t.v_head_dim), -1
            )
        )
    return weights


def flat(params):
    return {k: v for k, v in tree_flatten(params)}


def assert_same_weights(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        assert mx.array_equal(actual[key], expected[key]).item(), key


# --------------------------------------------------------------------- registry
def test_registry_and_config():
    module, _ = get_model_and_args({"model_type": "youtu_vita"})
    assert module.__name__.endswith("youtu_vita")
    config = ModelConfig.from_dict(
        {
            "model_type": "youtu_vita",
            "text_config": {"tie_word_embeddings": True, "hidden_size": 32},
            "omni_config": {"hidden_size": 32},
        }
    )
    # the checkpoint has a separate lm_head even though text_config says "tied"
    assert config.text_config.tie_word_embeddings is False
    assert config.omni_config.hidden_size == 32
    assert config.encoder_dtype == "float32"
    assert config.eos_token_id == [128001, 128257]
    # utils.update_module_configs re-parses the raw text_config: still untied
    raw = type(config.text_config).from_dict({"tie_word_embeddings": True})
    assert raw.tie_word_embeddings is False


# --------------------------------------------------------------------- sanitize
def test_sanitize_converts_checkpoint_layout_and_is_idempotent():
    model = randomize(tiny_model())
    expected = flat(model.parameters())
    checkpoint = to_checkpoint(model)
    assert any(k.endswith("kv_b_proj.weight") for k in checkpoint)
    assert "lm_head.weight" in checkpoint
    # keys the reference carries but MLX does not use
    checkpoint["model.vision_model.junk"] = mx.zeros((1,))
    checkpoint["model.audio_model.junk"] = mx.zeros((1,))
    checkpoint["model.language_model.layers.0.self_attn.rotary_emb.inv_freq"] = (
        mx.zeros((2,))
    )

    first = model.sanitize(dict(checkpoint))
    assert_same_weights(first, expected)
    conv = first["omni_model.audio_embeddings.conv2d2.weight"]
    assert conv.shape[1:3] == (3, 3)  # transposed to MLX NHWC kernels

    second = model.sanitize(
        dict(first)
    )  # load_model re-runs sanitize on converted ckpts
    assert_same_weights(second, expected)

    model.load_weights(list(first.items()), strict=True)


def test_sanitize_encoder_dtype():
    model = randomize(tiny_model())
    half = {
        k: v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
        for k, v in to_checkpoint(model).items()
    }
    out = model.sanitize(dict(half))
    assert all(
        v.dtype == mx.float32 for k, v in out.items() if k.startswith("omni_model.")
    )
    assert out["language_model.lm_head.weight"].dtype == mx.bfloat16

    same = tiny_model(encoder_dtype="same")
    out = same.sanitize(dict(half))
    assert (
        out["omni_model.vision_embeddings.patch_embedding.weight"].dtype == mx.bfloat16
    )


def test_quant_predicate_skips_omni_encoder():
    predicate = tiny_model().quant_predicate
    assert not predicate("omni_model.encoder.layers.0.self_attn.q_proj", None)
    assert predicate("language_model.model.layers.0.mlp.gate_proj", None)
    assert predicate("language_model.lm_head", None)


# ------------------------------------------------------------- language model
def test_mla_cache_geometry_and_logits_to_keep():
    model = tiny_model()
    lm = model.language_model
    assert lm.n_kv_heads == 1  # MLA caches one shared latent head
    assert lm.head_dim == 12  # qk_nope + qk_rope
    assert "lm_head" in dict(lm.named_modules()) or hasattr(lm, "lm_head")
    ids = mx.array([[1, 5, 9, 13, 2, 7]])
    full = model(ids).logits
    last = lm(ids, logits_to_keep=1).logits
    assert last.shape == (1, 1, 64)
    assert mx.allclose(full[:, -1:], last, atol=1e-5).item()


def test_chunked_prefill_matches_full_prefill():
    model = randomize(tiny_model())
    ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
    ModelChecks().prefill_parity(model, ids, {}, (3,))


def test_text_only_generation_cache_is_used():
    model = randomize(tiny_model())
    cache = make_prompt_cache(model.language_model)
    ids = mx.array([[1, 5, 9, 13]])
    model(ids, cache=cache)
    assert cache[0].offset == 4


# ------------------------------------------------------------ multimodal embeds
def image_inputs(grid=(1, 4, 4), patch=4, channels=3):
    t, gh, gw = grid
    pixels = mx.random.normal((t * gh * gw, channels * patch * patch))
    tokens = t * gh * gw // 4  # 2x2 spatial merge
    ids = [1, 2] + [IMAGE_ID] * tokens + [3, 4]
    return mx.array([ids]), pixels, np.array([grid]), tokens


def test_image_features_are_scattered_at_image_tokens():
    model = tiny_model()
    ids, pixels, grid, n = image_inputs()
    out = model.get_input_embeddings(ids, pixel_values=pixels, image_grid_thw=grid)
    base = model.language_model.model.embed_tokens(ids)
    feats = model.omni_model.forward_vision(pixels, grid)
    assert feats.shape == (n, 32) and feats.dtype == mx.float32
    assert out.inputs_embeds.shape == base.shape
    assert mx.array_equal(out.inputs_embeds[0, 2 : 2 + n], feats.astype(base.dtype))
    assert mx.array_equal(out.inputs_embeds[0, :2], base[0, :2]).item()
    assert mx.array_equal(out.inputs_embeds[0, 2 + n :], base[0, 2 + n :]).item()
    # explicit processor indices (batch row, absolute position) agree with token search
    indices = np.stack([np.zeros(n, np.int64), np.arange(2, 2 + n)])
    again = model.get_input_embeddings(
        ids, pixel_values=pixels, image_grid_thw=grid, image_indices=indices
    )
    assert mx.array_equal(out.inputs_embeds, again.inputs_embeds).item()


def test_image_requires_grid_and_matching_token_count():
    model = tiny_model()
    ids, pixels, grid, n = image_inputs()
    with pytest.raises(ValueError, match="image_grid_thw"):
        model.get_input_embeddings(ids, pixel_values=pixels)
    short = mx.array([[1, 2] + [IMAGE_ID] * (n - 1) + [3]])
    with pytest.raises(ValueError, match="does not match"):
        model.get_input_embeddings(short, pixel_values=pixels, image_grid_thw=grid)


def test_two_images_use_separate_attention_windows():
    model = randomize(tiny_model())
    _, p1, g1, _ = image_inputs()
    _, p2, g2, _ = image_inputs()
    joint = model.omni_model.forward_vision(
        mx.concatenate([p1, p2]), np.concatenate([g1, g2])
    )
    a = model.omni_model.forward_vision(p1, g1)
    b = model.omni_model.forward_vision(p2, g2)
    np.testing.assert_allclose(
        np.array(joint), np.array(mx.concatenate([a, b])), atol=1e-4, rtol=1e-4
    )


def test_audio_features_are_scattered_and_lengths_match_processor():
    model = tiny_model()
    mels = [mx.random.normal((120, 16)), mx.random.normal((100, 16))]
    lens = [audio_output_length(m.shape[0]) for m in mels]
    assert lens == [proc.audio_token_length(120), proc.audio_token_length(100)]
    assert lens[1] == 13
    feats = model.omni_model.forward_audio(mels)
    assert [f.shape for f in feats] == [(n, 32) for n in lens]

    positions, ids = [], [1, 2]
    for n in lens:
        positions.append(np.arange(len(ids), len(ids) + n))
        ids += [AUDIO_ID] * n + [3]
    idx = [np.stack([np.zeros((1, len(p)), np.int64), p[None]]) for p in positions]
    out = model.get_input_embeddings(
        mx.array([ids]), audios=mels, audio_indices=idx
    ).inputs_embeds
    mx.eval(out)
    base = model.language_model.model.embed_tokens(mx.array([ids]))
    for p, f in zip(positions, feats):
        assert mx.array_equal(out[0, int(p[0]) : int(p[-1]) + 1], f).item()
    assert mx.array_equal(out[0, :2], base[0, :2]).item()


def test_video_input_is_rejected_without_frames_path():
    model = tiny_model()
    with pytest.raises(NotImplementedError):
        model.get_input_embeddings(
            mx.array([[1, 2]]), pixel_values_videos=mx.zeros((1, 48))
        )


# -------------------------------------------------------------------- processor
class CharTokenizer:
    """Stand-in for the fast tokenizer: specials map to fixed ids, text is per-char."""

    SPECIAL = {
        "<|image|>": 100,
        "<|image_pad|>": IMAGE_ID,
        "<|vision_start|>": 102,
        "<|vision_end|>": 103,
        "<|audio|>": 104,
        "<|audio_pad|>": AUDIO_ID,
        "<|audio_start|>": 106,
        "<|audio_end|>": 107,
        "<|video|>": 108,
        "<|video_start|>": 109,
        "<|video_end|>": 110,
    }

    chat_template = None
    pad_token = None
    eos_token = "<eos>"

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL.get(token, -1)

    def __call__(self, text, add_special_tokens=True):
        ids, i = [], 0
        while i < len(text):
            for tag, tid in self.SPECIAL.items():
                if text.startswith(tag, i):
                    ids.append(tid)
                    i += len(tag)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


def make_processor():
    return proc.YoutuVitaProcessor(tokenizer=CharTokenizer())


def test_image_processor_resize_pad_and_patch_order():
    ip = proc.YoutuVitaImageProcessor()
    # inside [min, max]: not resized, padded bottom/right to a multiple of 32 with 127
    canvas, ph, pw, rh, rw = ip.resize_and_pad(
        proc.Image.new("RGB", (40, 70), (255, 0, 0))
    )
    assert (rh, rw) == (70, 40) and (ph, pw) == (96, 64)
    arr = np.array(canvas)
    assert tuple(arr[0, 0]) == (255, 0, 0) and tuple(arr[-1, -1]) == (127, 127, 127)
    patches, grid, _ = ip.process_image(np.full((40, 70, 3), 128, np.uint8))
    assert grid == [1, 3, 4] or grid == [1, 4, 6]  # (padded h, w) / 16
    assert patches.shape == (grid[1] * grid[2], 3 * 16 * 16)
    assert patches.dtype == np.float32
    # downscale when above the pixel budget keeps the aspect ratio
    small = proc.YoutuVitaImageProcessor(image_max_num_tokens=4)
    _, ph, pw, rh, rw = small.resize_and_pad(proc.Image.new("RGB", (512, 256)))
    assert ph * pw <= small.max_pixels and abs(rw / rh - 2) < 0.1
    # 2x2 merge order: the first four rows are one 2x2 neighbourhood
    x = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    p = ip.patchify(x)
    assert np.array_equal(p[0].reshape(3, 16, 16)[:, 0, 0], x[0, 0])
    assert np.array_equal(p[1].reshape(3, 16, 16)[:, 0, 0], x[0, 16])
    assert np.array_equal(p[2].reshape(3, 16, 16)[:, 0, 0], x[16, 0])


def test_processor_expands_image_tag():
    processor = make_processor()
    image = np.full((64, 96, 3), 200, np.uint8)
    out = processor(text="a<|image|>b", images=[image])
    ids = out["input_ids"][0]
    t, gh, gw = out["image_grid_thw"][0]
    assert (t, gh, gw) == (1, 4, 6)
    assert ids[0] == ord("a") and ids[-1] == ord("b")
    assert (ids == IMAGE_ID).sum() == (gh // 2) * (gw // 2)
    assert out["images"].shape == (gh * gw, 3 * 16 * 16)
    # <|vision_start|> "64*96" \n ... <|vision_end|>
    assert ids[1] == 102 and ids[-2] == 103
    assert ids[2 : 2 + 5].tolist() == [ord(c) for c in "64*96"]
    batch, pos = out["image_indices"]
    assert (batch == 0).all()
    assert (ids[pos] == IMAGE_ID).all() and len(pos) == (ids == IMAGE_ID).sum()
    assert out["attention_mask"].shape == out["input_ids"].shape

    with pytest.raises(ValueError, match="tag"):
        processor(text="no tag", images=[image])


def test_processor_expands_audio_tag_and_indices():
    processor = make_processor()
    wave_ = (0.1 * np.sin(np.linspace(0, 400 * np.pi, 16000 * 2))).astype(np.float32)
    out = processor(text="x<|audio|>y", audio=[wave_])
    ids = out["input_ids"][0]
    (clip,) = out["audios"]
    assert clip.shape[1] == 128 and clip.dtype == np.float32
    n = proc.audio_token_length(clip.shape[0])
    assert (ids == AUDIO_ID).sum() == n
    (index,) = out["audio_indices"]
    assert index.shape == (2, 1, n)
    assert (ids[index[1][0]] == AUDIO_ID).all()
    assert ids[index[1][0][0] - 1] == 106 and ids[index[1][0][-1] + 1] == 107
    with pytest.raises(ValueError, match="tag"):
        processor(text="no tag", audio=[wave_])


def test_audio_helpers():
    assert proc.audio_token_length(100) == 13
    assert proc.audio_token_length(120) == audio_output_length(120)
    mel = proc.WhisperLogMel()(np.zeros(16000, np.float32))
    assert mel.shape == (100, 128) and np.isfinite(mel).all()
    # short clip: one chunk; long clip: split under the 30 s limit
    assert len(proc.split_audio_chunks(300, 3.0)) == 1
    spans = proc.split_audio_chunks(7000, 70.0)
    assert spans[0][0] == 0 and spans[-1][1] == 7000 and len(spans) >= 3


def test_load_audio_accepts_arrays_and_wav_file_objects():
    samples = (np.sin(np.linspace(0, 100, 8000)) * 0.5).astype(np.float32)
    assert proc.load_audio(samples, 16000).shape == (8000,)
    assert proc.load_audio((samples, 8000), 16000).shape == (16000,)  # resampled
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    buf.seek(0)
    loaded = proc.load_audio(buf, 16000)  # the server hands over BytesIO
    assert loaded.shape == (8000,)
    np.testing.assert_allclose(loaded, samples, atol=1e-3)
    assert make_processor().load_audio(samples).shape == (8000,)


def test_video_sampling_policy():
    idx, ts, fps = proc.sample_frame_indices(300, 30.0, 1, 128)
    assert fps == 30 and idx[:3] == [0, 30, 60] and len(idx) == 10
    assert ts[1] == pytest.approx(1.0)
    idx, _, _ = proc.sample_frame_indices(5000, 25.0, 1, 128)
    assert len(idx) == 128 and idx[-1] < 5000


def test_video_decoding_does_not_require_av(tmp_path, monkeypatch):
    """``av`` is optional: frames fall back to OpenCV, and importing needs neither."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setitem(sys.modules, "av", None)  # `import av` -> ImportError
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write test video here")
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    frames, stamps, fps, duration = proc.decode_video_frames(path, 1, 128)
    assert len(frames) == len(stamps) >= 3
    assert frames[0].shape == (48, 64, 3) and fps == 4
    assert duration == pytest.approx(3.0, abs=0.5)
    assert proc.YoutuVitaProcessor.decodes_video_files is True


def test_prompt_utils_registers_youtu_vita():
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

    assert MODEL_CONFIG["youtu_vita"] == MessageFormat.LIST_WITH_IMAGE_FIRST
