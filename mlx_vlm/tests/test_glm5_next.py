from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.fp8 import transform_fp8_weights
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchPoolingCache,
    HierarchyCache,
    KVCache,
    PoolingCache,
)
from mlx_vlm.models.glm5_next.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.glm5_next.glm5_next import Model
from mlx_vlm.models.glm5_next.language import (
    Glm5NextLinearAttention,
    LanguageModel,
    _exact_pool_select,
    _hisa_pool_select,
    _score_index_keys,
    _sparse_prefill_attention,
)
from mlx_vlm.models.glm5_next.processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    _resize_geometry,
)
from mlx_vlm.models.sparse_attention import indexed_sparse_attention
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs


def _text_config():
    return TextConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        indexer_types=["full", "full"],
        index_topk=4,
        index_kpool=2,
        index_head_dim=4,
        index_n_heads=2,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        max_position_embeddings=64,
    )


def _vision_config(depth=0):
    return VisionConfig(
        depth=depth,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        projection_intermediate_size=24,
    )


def _image_processor():
    return Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=4,
        max_image_tokens=64,
        do_normalize=False,
    )


def _video_processor():
    return Glm5NextVideoProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=4,
        max_image_tokens=64,
        do_normalize=False,
    )


def test_glm5_next_speculative_rejection_restores_recurrent_and_sparse_caches():
    language_model = LanguageModel(_text_config())
    rejected_cache = language_model.make_cache()
    reference_cache = language_model.make_cache()
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5]], dtype=mx.int32)

    language_model(prompt, cache=rejected_cache)
    language_model(prompt, cache=reference_cache)
    _, _, rollback_state = language_model.speculative_verify_hidden(
        verify, rejected_cache
    )
    language_model.rollback_speculative_cache(
        rejected_cache, rollback_state, accepted=0, block_size=2
    )
    language_model(verify[:, :1], cache=reference_cache)

    next_token = mx.array([[6]], dtype=mx.int32)
    rejected = language_model(next_token, cache=rejected_cache).logits
    reference = language_model(next_token, cache=reference_cache).logits
    mx.eval(rejected, reference)

    assert mx.allclose(rejected, reference, atol=1e-5, rtol=1e-5).item()


def test_glm5_next_speculative_verify_matches_sequential_decode_arithmetic():
    language_model = LanguageModel(_text_config())
    verify_cache = language_model.make_cache()
    sequential_cache = language_model.make_cache()
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify_tokens = mx.array([[4, 5]], dtype=mx.int32)

    language_model(prompt, cache=verify_cache)
    language_model(prompt, cache=sequential_cache)
    verify_hidden, _, _ = language_model.speculative_verify_hidden(
        verify_tokens, verify_cache
    )
    sequential_hidden = []
    for position in range(verify_tokens.shape[1]):
        output = language_model(
            verify_tokens[:, position : position + 1],
            cache=sequential_cache,
            return_hidden=True,
        )
        sequential_hidden.append(output.hidden_states[-1])
    sequential_hidden = mx.concatenate(sequential_hidden, axis=1)
    mx.eval(verify_hidden, sequential_hidden)

    assert mx.allclose(verify_hidden, sequential_hidden, atol=1e-5, rtol=1e-5).item()


def test_glm5_next_chunked_prefill_policy_supports_mtp_capture():
    language_model = LanguageModel(_text_config())
    draft_model = object()

    assert language_model.chunked_prefill_policy()
    assert language_model.chunked_prefill_policy(
        draft_model=draft_model,
        draft_kind="mtp",
        prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
    )
    assert not language_model.chunked_prefill_policy(
        draft_model=draft_model,
        draft_kind="mtp",
        prefill_kwargs={"return_hidden": True},
    )
    assert not language_model.chunked_prefill_policy(
        draft_model=draft_model,
        draft_kind="dflash",
    )


def test_glm5_next_speculative_argmax_uses_exact_head(monkeypatch):
    import mlx_vlm.models.glm5_next.speculative_verifier as verifier

    language_model = LanguageModel(_text_config())
    hidden = mx.zeros((1, 3, language_model.args.hidden_size))
    expected_logits = mx.zeros((1, 3, language_model.args.vocab_size))
    expected_logits[:, :, 7] = 1
    calls = []

    def exact_head(weight, inputs):
        calls.append((weight, inputs))
        return expected_logits

    monkeypatch.setattr(verifier, "exact_speculative_verify_weight", exact_head)
    tokens = language_model.speculative_argmax_from_hidden(hidden)
    mx.eval(tokens)

    assert len(calls) == 1
    assert calls[0][1] is hidden
    assert tokens.tolist() == [[7, 7, 7]]


class _Tokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    image_token = "<|image|>"
    image_token_id = 31
    video_token = "<|video|>"
    video_token_id = 30
    video_start_token = "<|begin_of_video|>"
    video_start_token_id = 28
    video_end_token = "<|end_of_video|>"
    video_end_token_id = 29
    pad_token = "<|pad|>"
    eos_token = "<|end|>"

    def __init__(self):
        self.last_text = None

    def convert_tokens_to_ids(self, token):
        return {
            self.image_token: self.image_token_id,
            self.video_token: self.video_token_id,
            self.video_start_token: self.video_start_token_id,
            self.video_end_token: self.video_end_token_id,
        }.get(token, 1)

    def __call__(self, text, **kwargs):
        del kwargs
        self.last_text = text
        rows = []
        for value in text:
            ids = []
            index = 0
            while index < len(value):
                if value.startswith(self.image_token, index):
                    ids.append(self.image_token_id)
                    index += len(self.image_token)
                elif value.startswith(self.video_token, index):
                    ids.append(self.video_token_id)
                    index += len(self.video_token)
                elif value.startswith(self.video_start_token, index):
                    ids.append(self.video_start_token_id)
                    index += len(self.video_start_token)
                elif value.startswith(self.video_end_token, index):
                    ids.append(self.video_end_token_id)
                    index += len(self.video_end_token)
                else:
                    ids.append(1)
                    index += 1
            rows.append(ids)
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def batch_decode(self, *args, **kwargs):
        del args, kwargs
        return []

    def decode(self, *args, **kwargs):
        del args, kwargs
        return ""

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        return ""


def test_glm5_next_config_builds_checkpoint_schedules():
    config = TextConfig(num_hidden_layers=5, index_topk=8, index_kpool=4)

    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ]
    assert config.mlp_layer_types == ["dense", "dense", "dense", "sparse", "sparse"]
    assert config.linear_num_heads == 64
    assert config.linear_lower_bound == -5.0
    assert config.index_hisa_block == 0
    assert config.index_hisa_keep == 0


def test_glm5_next_config_matches_transformers_indexer_schedule_and_validation():
    config = TextConfig(
        num_hidden_layers=5,
        index_topk=8,
        index_kpool=4,
        index_topk_freq=2,
        index_skip_topk_offset=2,
    )
    assert config.indexer_types == ["full", "full", "shared", "full", "shared"]
    assert config.head_dim == config.qk_rope_head_dim
    assert config.qk_head_dim == config.qk_nope_head_dim

    pattern = TextConfig(
        num_hidden_layers=5,
        index_topk=8,
        index_kpool=4,
        index_topk_pattern="FSSFS",
    )
    assert pattern.indexer_types == ["full", "shared", "shared", "full", "shared"]

    with pytest.raises(ValueError, match="num_attention_heads"):
        TextConfig(num_attention_heads=2, num_key_value_heads=1)


def test_glm5_next_preprocessing_preserves_aspect_and_pads_temporally():
    settings = {
        "patch_size": 2,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "patch_expand_factor": 1,
        "min_image_tokens": 4,
        "max_image_tokens": 64,
    }
    assert _resize_geometry(2, 5, 9, **settings) == (8, 12, 6, 12)

    image = np.full((5, 9, 3), 255, dtype=np.uint8)
    image_inputs = _image_processor()(image)
    assert image_inputs["pixel_values"].shape == (24, 24)
    np.testing.assert_array_equal(image_inputs["image_grid_thw"], [[1, 4, 6]])

    video = np.full((3, 5, 9, 3), 255, dtype=np.uint8)
    video_inputs = _video_processor()(video)
    assert video_inputs["pixel_values_videos"].shape == (48, 24)
    np.testing.assert_array_equal(video_inputs["video_grid_thw"], [[2, 4, 6]])


def test_glm5_next_processor_expands_video_frames_with_timestamps():
    tokenizer = _Tokenizer()
    processor = Glm5NextProcessor(
        image_processor=_image_processor(),
        tokenizer=tokenizer,
        video_processor=_video_processor(),
    )
    video = np.full((3, 5, 9, 3), 255, dtype=np.uint8)
    metadata = SimpleNamespace(timestamps=[0.0, 0.5, 1.0, 1.5])

    output = processor(
        text="<|begin_of_video|><|video|><|end_of_video|>",
        videos=video,
        video_metadata=[metadata],
    )

    rendered = tokenizer.last_text[0]
    assert rendered.count("<|begin_of_image|>") == 2
    assert rendered.count("<|image|>") == 12
    assert "0.0 seconds" in rendered
    assert "1.0 seconds" in rendered
    token_types = np.asarray(output["mm_token_type_ids"])
    assert int((token_types == 2).sum()) == 12
    assert int((token_types == 1).sum()) == 0


def test_glm5_next_processor_handles_ragged_multimodal_token_types():
    processor = Glm5NextProcessor(
        image_processor=_image_processor(),
        tokenizer=_Tokenizer(),
        video_processor=_video_processor(),
    )
    assert processor.create_mm_token_type_ids(
        [
            [31, 1],
            [28, 31, 31, 29, 31],
        ]
    ) == [
        [1, 0],
        [0, 2, 2, 0, 1],
    ]


def test_glm5_next_tied_embeddings_match_transformers_behavior():
    config = _text_config()
    config.tie_word_embeddings = True
    model = LanguageModel(config)
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    hidden = model.model(tokens)
    expected = model.model.embed_tokens.as_linear(hidden)
    actual = model(tokens).logits
    mx.eval(expected, actual)

    assert not hasattr(model, "lm_head")
    assert float(mx.max(mx.abs(actual - expected)).item()) == 0.0


def test_glm5_next_prepare_inputs_propagates_video_metadata():
    tokenizer = _Tokenizer()
    processor = Glm5NextProcessor(
        image_processor=_image_processor(),
        tokenizer=tokenizer,
        video_processor=_video_processor(),
    )
    video = np.full((3, 3, 5, 9), 255, dtype=np.uint8)

    output = prepare_inputs(
        processor,
        videos=video,
        prompts="<|video|>",
        fps=2.0,
    )

    assert "pixel_values_videos" in output
    assert "0.0 seconds" in tokenizer.last_text[0]
    assert "1.0 seconds" in tokenizer.last_text[0]


def test_glm5_next_prepare_inputs_samples_file_and_preserves_timestamps(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
    writer.release()

    tokenizer = _Tokenizer()
    processor = Glm5NextProcessor(
        image_processor=_image_processor(),
        tokenizer=tokenizer,
        video_processor=_video_processor(),
    )
    output = prepare_inputs(
        processor,
        videos=video_path,
        prompts="<|video|>",
        fps=2.0,
        max_frames=4,
    )

    assert output["pixel_values_videos"].shape[0] > 0
    assert "0.0 seconds" in tokenizer.last_text[0]
    assert "1.9 seconds" in tokenizer.last_text[0]


def test_glm5_next_frame_sampling_matches_transformers_capped_case():
    processor = _video_processor()
    metadata = SimpleNamespace(total_num_frames=300, fps=30.0, duration=10.0)

    indices = processor.sample_frames(metadata, fps=2.0, max_frames=4)

    assert indices.tolist() == [0, 99, 199, 299]


def test_glm5_next_prompt_utils_formats_image_and_video_content():
    image_messages = apply_chat_template(
        None,
        {"model_type": "glm5_next"},
        "Describe the image.",
        return_messages=True,
        num_images=1,
    )
    assert image_messages[0]["content"][0] == {"type": "image"}

    video_messages = apply_chat_template(
        None,
        {"model_type": "glm5_next"},
        "Describe the video.",
        return_messages=True,
        video="clip.mp4",
        fps=2,
    )
    assert video_messages[0]["content"][0] == {
        "type": "video",
        "video": "clip.mp4",
        "max_pixels": 224 * 224,
        "fps": 2,
    }


def test_glm5_next_hybrid_decoder_cache_matches_full_forward():
    mx.random.seed(0)
    model = LanguageModel(_text_config())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    full = model(tokens).logits
    cache = model.make_cache()
    decoded = []
    for index in range(tokens.shape[1]):
        logits = model(tokens[:, index : index + 1], cache=cache).logits
        mx.eval(logits)
        decoded.append(logits)
    decoded = mx.concatenate(decoded, axis=1)
    mx.eval(full, decoded)

    assert full.shape == (1, 5, 40)
    assert bool(mx.all(mx.isfinite(full)).item())
    assert float(mx.max(mx.abs(full - decoded)).item()) < 1e-5

    chunk_cache = model.make_cache()
    first = model(tokens[:, :2], cache=chunk_cache).logits
    second = model(tokens[:, 2:], cache=chunk_cache).logits
    chunked = mx.concatenate([first, second], axis=1)
    mx.eval(chunked)
    assert float(mx.max(mx.abs(full - chunked)).item()) < 1e-5


def test_glm5_next_kda_masks_all_padding_dependent_projections():
    mx.random.seed(16)
    config = _text_config()
    attention = Glm5NextLinearAttention(config, 0)
    inputs = mx.random.normal((2, 5, config.hidden_size))
    mask = mx.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=mx.bool_)
    masked_inputs = mx.where(mask[..., None], inputs, 0)

    actual = attention(inputs, mask)
    expected = attention(masked_inputs, mask)
    mx.eval(actual, expected)

    assert float(mx.max(mx.abs(actual - expected)).item()) == 0.0


def test_glm5_next_sparse_prefill_uses_fused_sdpa_equivalent_math():
    mx.random.seed(1)
    q = mx.random.normal((1, 2, 3, 4))
    k = mx.random.normal((1, 2, 5, 4))
    v = mx.random.normal((1, 2, 5, 3))
    indices = mx.array([[[0, -1, -1], [0, 1, 2], [1, 3, 4]]], dtype=mx.int32)
    scale = 0.5

    actual = _sparse_prefill_attention(q, k, v, indices, scale, chunk_size=2)

    safe = mx.clip(indices, 0, k.shape[2] - 1)
    gather_idx = safe[:, None, :, :, None]
    selected_k = mx.take_along_axis(
        mx.broadcast_to(k[:, :, None], (1, 2, 3, 5, 4)),
        gather_idx,
        axis=3,
    )
    selected_v = mx.take_along_axis(
        mx.broadcast_to(v[:, :, None], (1, 2, 3, 5, 3)),
        gather_idx,
        axis=3,
    )
    scores = (q[..., None, :].astype(mx.float32) * selected_k.astype(mx.float32)).sum(
        axis=-1
    ) * scale
    scores = mx.where(indices[:, None] >= 0, scores, mx.finfo(mx.float32).min)
    probs = mx.softmax(scores, axis=-1, precise=True).astype(v.dtype)
    expected = (probs[..., None] * selected_v).sum(axis=-2)
    mx.eval(actual, expected)

    assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-5


def test_indexed_sparse_attention_metal_matches_fallback():
    mx.random.seed(11)
    q = mx.random.normal((2, 4, 5, 32)).astype(mx.bfloat16)
    k = mx.random.normal((2, 4, 64, 32)).astype(mx.bfloat16)
    v = mx.random.normal((2, 4, 64, 64)).astype(mx.bfloat16)
    indices = mx.random.randint(0, 64, (2, 5, 9)).astype(mx.int32)
    valid_width = mx.array([7, 5], dtype=mx.int32)[:, None, None]
    indices = mx.where(mx.arange(9)[None, None] < valid_width, indices, -1)
    scale = 32**-0.5

    actual = indexed_sparse_attention(q, k, v, indices, scale, min_sparse_ratio=0)
    if actual is None:
        pytest.skip("The fused indexed-attention kernel requires Metal")
    expected = _sparse_prefill_attention(
        q, k, v, indices, scale, chunk_size=2, use_kernel=False
    )
    mx.eval(actual, expected)

    error = mx.abs(actual - expected)
    assert float(mx.max(error).item()) <= 1e-2
    assert float(mx.mean(error).item()) <= 2e-3


def test_glm5_next_hisa_keep_all_matches_flat_indexer():
    mx.random.seed(12)
    batch, query_length, heads, dim = 2, 7, 3, 8
    pool_count, block_size, select_k = 24, 4, 6
    q = mx.random.normal((batch, query_length, heads, dim)).astype(mx.bfloat16)
    pool_keys = mx.random.normal((batch, pool_count, dim)).astype(mx.bfloat16)
    weights = mx.random.normal((batch, query_length, heads))
    candidates = mx.arange(pool_count)[None, None] <= (
        mx.arange(query_length)[None, :, None] + 17
    )
    candidates = mx.broadcast_to(candidates, (batch, query_length, pool_count))
    representatives = pool_keys.reshape(
        batch, pool_count // block_size, block_size, dim
    ).mean(axis=2)

    hisa = _hisa_pool_select(
        q,
        pool_keys,
        weights,
        candidates,
        representatives,
        [pool_count // block_size] * batch,
        select_k,
        dim**-0.5,
        block_size,
        pool_count // block_size,
        chunk_size=3,
    )
    flat_scores = _score_index_keys(q, pool_keys, weights, dim**-0.5)
    flat_scores = mx.where(candidates, flat_scores, mx.finfo(mx.float32).min)
    flat = mx.argpartition(-flat_scores, kth=select_k - 1, axis=-1)[..., :select_k]
    mx.eval(hisa, flat)

    assert bool(mx.all(mx.sort(hisa, axis=-1) == mx.sort(flat, axis=-1)).item())


def test_glm5_next_exact_pool_select_matches_full_pool_scoring():
    mx.random.seed(15)
    batch, query_length, heads, dim = 2, 7, 3, 8
    pool_count, select_k = 11, 3
    q = mx.random.normal((batch, query_length, heads, dim)).astype(mx.bfloat16)
    pool_keys = mx.random.normal((batch, pool_count, dim)).astype(mx.bfloat16)
    weights = mx.random.normal((batch, query_length, heads)).astype(mx.float32)
    pool_ends = mx.broadcast_to(
        mx.arange(pool_count, dtype=mx.int32)[None], (batch, pool_count)
    )
    pool_valid = mx.ones((batch, pool_count), dtype=mx.bool_)
    pool_valid[1, -1] = False
    query_positions = mx.arange(4, 4 + query_length, dtype=mx.int32)

    actual, actual_valid = _exact_pool_select(
        q,
        pool_keys,
        weights,
        pool_ends,
        pool_valid,
        query_positions,
        select_k,
        dim**-0.5,
        chunk_size=3,
    )
    candidates = pool_valid[:, None] & (
        pool_ends[:, None] <= query_positions[None, :, None]
    )
    scores = _score_index_keys(q, pool_keys, weights, dim**-0.5)
    scores = mx.where(candidates, scores, mx.finfo(mx.float32).min)
    expected = mx.argpartition(-scores, kth=select_k - 1, axis=-1)[..., :select_k]
    expected_valid = mx.take_along_axis(candidates, expected, axis=-1)
    mx.eval(actual, actual_valid, expected, expected_valid)

    assert bool(mx.all(mx.sort(actual, axis=-1) == mx.sort(expected, axis=-1)).item())
    assert bool(mx.all(actual_valid == expected_valid).item())


def test_hierarchy_cache_incremental_means_and_state_roundtrip():
    mx.random.seed(13)
    values = mx.random.normal((1, 17, 8)).astype(mx.bfloat16)
    cache = HierarchyCache(block_size=4)
    cache.update_and_fetch(values[:, :3])
    cache.update_and_fetch(values[:, 3:8])
    representatives, lengths = cache.update_and_fetch(values[:, 8:])
    expected = values[:, :16].reshape(1, 4, 4, 8).mean(axis=2)
    mx.eval(representatives, expected)

    assert lengths == [4]
    assert cache.remainders == [1]
    assert float(mx.max(mx.abs(representatives - expected)).item()) == 0.0

    restored = HierarchyCache.from_state(cache.state, cache.meta_state)
    mx.eval(restored.representatives)
    assert restored.block_size == 4
    assert restored.remainders == [1]
    assert restored.representative_lengths == [4]
    assert (
        float(mx.max(mx.abs(restored.representatives - representatives)).item()) == 0.0
    )


def test_hierarchy_cache_merge_preserves_variable_length_rows():
    first = mx.arange(20).reshape(1, 10, 2).astype(mx.bfloat16)
    second = (100 + mx.arange(10)).reshape(1, 5, 2).astype(mx.bfloat16)
    first_cache = HierarchyCache(block_size=4)
    second_cache = HierarchyCache(block_size=4)
    first_cache.update_and_fetch(first)
    second_cache.update_and_fetch(second)

    merged = HierarchyCache.merge([first_cache, second_cache])
    additions = mx.array(
        [
            [[20, 21], [22, 23], [0, 0]],
            [[110, 111], [112, 113], [114, 115]],
        ],
        dtype=mx.bfloat16,
    )
    representatives, lengths = merged.update_and_fetch(additions, new_counts=[2, 3])
    expected_first = (
        mx.concatenate([first, additions[:1, :2]], axis=1)
        .reshape(1, 3, 4, 2)
        .mean(axis=2)
    )
    expected_second = (
        mx.concatenate([second, additions[1:, :3]], axis=1)
        .reshape(1, 2, 4, 2)
        .mean(axis=2)
    )
    mx.eval(representatives, expected_first, expected_second)

    assert lengths == [3, 2]
    assert merged.remainders == [0, 0]
    assert float(mx.max(mx.abs(representatives[:1, :3] - expected_first)).item()) == 0
    assert float(mx.max(mx.abs(representatives[1:, :2] - expected_second)).item()) == 0
    assert merged.extract(0).representative_lengths == [3]
    merged.filter(mx.array([1], dtype=mx.int32))
    assert merged.representative_lengths == [2]


def test_glm5_next_chunked_prefill_projects_only_new_tokens():
    mx.random.seed(2)
    model = LanguageModel(_text_config())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    full = model(tokens).logits

    attention = model.layers[1].self_attn
    original_embed_q = attention.embed_q
    projected_lengths = []

    class RecordingMultiLinear:
        def __call__(self, x, transpose=True):
            projected_lengths.append(x.shape[2])
            return original_embed_q(x, transpose=transpose)

    attention.embed_q = RecordingMultiLinear()
    cache = model.make_cache()
    first = model(tokens[:, :2], cache=cache).logits
    second = model(tokens[:, 2:], cache=cache).logits
    chunked = mx.concatenate([first, second], axis=1)
    mx.eval(full, chunked)

    sparse_cache = cache[1]
    assert projected_lengths == [2, 3]
    assert sparse_cache[1].keys.shape[-1] == 1
    assert isinstance(sparse_cache[2], PoolingCache)
    assert sparse_cache[2].pooled.shape[1] == 2
    assert sparse_cache[2].remainder == 1
    assert len(sparse_cache.caches) == 4
    assert isinstance(sparse_cache[3], KVCache)
    assert sparse_cache[3].size() == 5
    assert float(mx.max(mx.abs(full - chunked)).item()) < 1e-5


def test_glm5_next_chunked_hisa_matches_full_forward():
    mx.random.seed(14)
    config = _text_config()
    config.index_hisa_block = 2
    config.index_hisa_keep = 2
    config.index_hisa_min_pools = 0
    model = LanguageModel(config)
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

    full = model(tokens).logits
    cache = model.make_cache()
    chunks = [
        model(tokens[:, start : start + 4], cache=cache).logits
        for start in range(0, 12, 4)
    ]
    chunked = mx.concatenate(chunks, axis=1)
    mx.eval(full, chunked)

    hierarchy = cache[1][3]
    assert hierarchy.representative_lengths == [3]
    assert hierarchy.remainders == [0]
    assert float(mx.max(mx.abs(full - chunked)).item()) < 1e-5


def test_glm5_next_incremental_pooling_matches_left_padded_batch():
    mx.random.seed(3)
    model = LanguageModel(_text_config())
    model.eval()
    tokens = mx.array([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    attention_mask = mx.array([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=mx.bool_)
    full = model(tokens, attention_mask=attention_mask).logits

    cache = _make_cache(model, left_padding=[2, 0])
    first = model(
        tokens[:, :3], cache=cache, attention_mask=attention_mask[:, :3]
    ).logits
    second = model(
        tokens[:, 3:], cache=cache, attention_mask=attention_mask[:, 3:]
    ).logits
    chunked = mx.concatenate([first, second], axis=1)
    mx.eval(full, chunked)

    valid = mx.broadcast_to(attention_mask[..., None], full.shape)
    error = mx.where(valid, mx.abs(full - chunked), 0)
    assert isinstance(cache[1][2], BatchPoolingCache)
    assert cache[1][2]._pool_lengths == [2, 3]
    assert len(cache[1].caches) == 4
    assert isinstance(cache[1][3], BatchKVCache)
    assert float(mx.max(error).item()) < 1e-5


def test_glm5_next_multimodal_forward_and_sanitize():
    config = ModelConfig(
        text_config=_text_config(),
        vision_config=_vision_config(depth=1),
        image_token_id=31,
        video_token_id=30,
        image_start_token_id=32,
        image_end_token_id=33,
        video_start_token_id=34,
        video_end_token_id=35,
    )
    model = Model(config)
    model.eval()
    image_inputs = _image_processor()(np.full((4, 4, 3), 255, dtype=np.uint8))
    input_ids = mx.array([[31] * 4], dtype=mx.int32)
    output = model(
        input_ids,
        pixel_values=mx.array(image_inputs["pixel_values"]),
        image_grid_thw=mx.array(image_inputs["image_grid_thw"]),
    )
    mx.eval(output.logits)
    assert output.logits.shape == (1, 4, 40)
    assert bool(mx.all(mx.isfinite(output.logits)).item())

    raw = {
        "model.visual.patch_embed.proj.weight": mx.zeros((8, 3, 2, 2, 2)),
        "model.visual.downsample.weight": mx.zeros((16, 8, 2, 2)),
        "model.language_model.layers.0.hc_attn_fn": mx.zeros((24, 64)),
        "model.language_model.layers.0.self_attn.q_conv1d.weight": mx.zeros((8, 1, 2)),
        "model.language_model.layers.0.self_attn.k_conv1d.weight": mx.zeros((8, 1, 2)),
        "model.language_model.layers.0.self_attn.v_conv1d.weight": mx.zeros((8, 1, 2)),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((8, 16)),
        "model.language_model.layers.0.self_attn.k_proj.weight": mx.zeros((8, 16)),
        "model.language_model.layers.0.self_attn.v_proj.weight": mx.zeros((8, 16)),
        "model.language_model.layers.0.self_attn.f_a_proj.weight": mx.zeros((4, 16)),
        "model.language_model.layers.0.self_attn.b_proj.weight": mx.zeros((2, 16)),
        "model.language_model.layers.0.self_attn.g_a_proj.weight": mx.zeros((4, 16)),
        "model.language_model.layers.0.mlp.gate_proj.weight": mx.zeros((32, 16)),
        "model.language_model.layers.0.mlp.up_proj.weight": mx.zeros((32, 16)),
        "model.language_model.layers.1.self_attn.q_a_proj.weight": mx.zeros((8, 16)),
        "model.language_model.layers.1.self_attn.kv_a_proj_with_mqa.weight": mx.zeros(
            (4, 16)
        ),
        "model.language_model.layers.1.self_attn.kv_b_proj.weight": mx.zeros((16, 4)),
        "model.language_model.layers.1.mlp.shared_experts.gate_proj.weight": mx.zeros(
            (8, 16)
        ),
        "model.language_model.layers.1.mlp.shared_experts.up_proj.weight": mx.zeros(
            (8, 16)
        ),
        "model.language_model.layers.2.input_layernorm.weight": mx.zeros((16,)),
        "model.language_model.layers.0.fp8_proj.weight": mx.zeros(
            (32, 32), dtype=mx.uint8
        ),
        "model.language_model.layers.0.fp8_proj.weight_scale_inv": mx.ones(
            (1, 1), dtype=mx.bfloat16
        ),
    }
    for expert in range(2):
        for name, shape in (
            ("gate_proj", (8, 16)),
            ("up_proj", (8, 16)),
            ("down_proj", (16, 8)),
        ):
            raw[f"model.language_model.layers.1.mlp.experts.{expert}.{name}.weight"] = (
                mx.zeros(shape)
            )

    raw, _ = transform_fp8_weights(
        raw,
        {
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        },
    )
    sanitized = model.sanitize(raw)
    assert sanitized["vision_tower.patch_embed.proj.weight"].shape == (8, 2, 2, 2, 3)
    assert sanitized["vision_tower.downsample.weight"].shape == (16, 2, 2, 8)
    assert sanitized[
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"
    ].shape == (2, 8, 16)
    assert sanitized[
        "language_model.model.layers.0.self_attn.qkv_proj.weight"
    ].shape == (24, 16)
    assert sanitized[
        "language_model.model.layers.0.self_attn.qkv_conv.conv.weight"
    ].shape == (24, 2, 1)
    assert sanitized[
        "language_model.model.layers.0.self_attn.fbg_a_proj.weight"
    ].shape == (10, 16)
    assert sanitized["language_model.model.layers.0.mlp.gate_up_proj.weight"].shape == (
        64,
        16,
    )
    assert sanitized[
        "language_model.model.layers.1.self_attn.qkv_a_proj.weight"
    ].shape == (12, 16)
    assert sanitized[
        "language_model.model.layers.1.mlp.shared_experts.gate_up_proj.weight"
    ].shape == (16, 16)
    assert "language_model.model.layers.2.input_layernorm.weight" not in sanitized
    assert sanitized["language_model.model.layers.0.fp8_proj.weight"].dtype == mx.uint32
    assert sanitized["language_model.model.layers.0.fp8_proj.scales"].dtype == mx.uint8
    assert set(model.sanitize(sanitized)) == set(sanitized)
