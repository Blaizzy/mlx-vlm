"""Cancellation must preserve in-flight prefill state for surviving rows."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager
from mlx_vlm.generate import BatchGenerator, PromptProcessingBatch
from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache
from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3BatchKVCache
from mlx_vlm.models.qwen3_5 import LanguageModel, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache
from mlx_vlm.speculative.utils import SpeculativePrefill
from mlx_vlm.turboquant import BatchTurboQuantKVCache
from mlx_vlm.utils import StoppingCriteria


@pytest.fixture
def tiny_model():
    mx.random.seed(17)
    config = TextConfig(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=3,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 1, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    model = LanguageModel(
        config,
        ModelConfig(
            model_type="qwen3_5", text_config=config, vision_config=VisionConfig()
        ),
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def insert_prompts(gen, model):
    prompts = [[1, 2, 3, 4], list(range(1, 14))]
    uids = gen.insert(
        prompts,
        prompt_kwargs=[
            {
                "inputs_embeds": model.model.embed_tokens(mx.array([ids])),
                "position_ids": mx.broadcast_to(
                    mx.arange(len(ids))[None, None], (3, 1, len(ids))
                ),
                "rope_deltas": mx.zeros((1, 1), dtype=mx.int32),
            }
            for ids in prompts
        ],
    )
    return prompts, uids


@pytest.mark.parametrize("cancelled_row", [0, 1])
@pytest.mark.parametrize("step_size,chunks", [(1, 1), (3, 1), (3, 3), (3, 4), (5, 1)])
def test_remove_prefill_row_matches_standalone(
    tiny_model, cancelled_row, step_size, chunks
):
    processor = SimpleNamespace(stopping_criteria=StoppingCriteria([-1]))
    gen = BatchGenerator(
        tiny_model,
        processor,
        prefill_batch_size=2,
        prefill_step_size=step_size,
        max_tokens=5,
    )
    try:
        prompts, uids = insert_prompts(gen, tiny_model)
        for _ in range(chunks):
            gen.next()
        batch = gen._prompt_batch
        processed = batch._processed_prompt_columns
        remaining = batch._input_ids.shape[1]
        survivor = 1 - cancelled_row

        assert gen.remove(uids[cancelled_row]) is True
        assert gen._prompt_batch is batch
        assert batch.uids == [uids[survivor]]
        assert batch._processed_prompt_columns == processed
        assert batch._input_ids.shape == (1, remaining)
        assert not gen.unprocessed_prompts
        assert gen.remove(uids[cancelled_row]) is False

        tokens = []
        for _ in range(30):
            if not gen.has_work:
                break
            progress, responses = gen.next()
            assert all(p.uid == uids[survivor] for p in progress)
            assert all(r.uid == uids[survivor] for r in responses)
            tokens.extend(r.token for r in responses)
        assert not gen.has_work

        expected = []
        cache = tiny_model.make_cache()
        inputs = mx.array([prompts[survivor]])
        # Remove model-global MRoPE state left by the batched run.
        tiny_model._rope_deltas = None
        for _ in range(5):
            output = tiny_model(inputs, cache=cache)
            inputs = mx.argmax(output.logits[:, -1], axis=-1)[:, None]
            expected.append(inputs.item())
        assert tokens == expected
    finally:
        gen.close()


def test_remove_all_prefill_rows_releases_apc(tiny_model):
    processor = SimpleNamespace(stopping_criteria=StoppingCriteria([-1]))
    gen = BatchGenerator(
        tiny_model, processor, prefill_batch_size=2, prefill_step_size=3
    )
    try:
        _, uids = insert_prompts(gen, tiny_model)
        gen.next()
        batch = gen._prompt_batch
        # Track borrowed references independently of the model's APC policy.
        manager = APCManager(num_blocks=2, block_size=2)
        keys = mx.ones((1, 1, 4, 4))
        blocks = manager.store_kv_blocks([1, 2, 3, 4], [keys], [keys])
        assert len(blocks) == 2
        batch._apc_manager = manager
        batch._apc_meta = [{"apc_blocks": [block]} for block in blocks]

        assert gen.remove(uids[0]) is True
        assert [b.ref_cnt for b in blocks] == [0, 1]
        assert gen.remove(uids[1]) is True
        assert [b.ref_cnt for b in blocks] == [0, 0]
        assert gen._prompt_batch is None
        assert batch.prompt_cache == []
        assert batch._apc_meta == []
        assert not gen.has_work
        assert gen.next() == ([], [])
    finally:
        gen.close()


@pytest.mark.parametrize("kind", ["float", "quantized", "turboquant", "minimax", "qsa"])
@pytest.mark.parametrize("processed", [0, 3, 12])
def test_cache_filter_only_trims_processed_padding(kind, processed):
    if kind == "float":
        cache = BatchKVCache([9, 0])
    elif kind == "quantized":
        cache = BatchQuantizedKVCache([9, 0], group_size=32, bits=4)
    elif kind == "turboquant":
        cache = BatchTurboQuantKVCache([9, 0], bits=4)
    elif kind == "minimax":
        cache = MiniMaxM3BatchKVCache([9, 0])
    else:
        cache = BatchQSAKVCache([9, 0])
    if processed:
        keys = mx.random.normal((2, 2, processed, 64))
        cache.update_and_fetch(keys, keys)
        if kind == "minimax":
            cache.update_index_and_fetch(keys)
        elif kind == "qsa":
            cache.update_indexer(
                keys[:, 0], mx.broadcast_to(mx.arange(processed)[None], (2, processed))
            )
        mx.eval(cache.state)
    cache.filter(mx.array([0], dtype=mx.int32))

    trimmed = min(9, processed)
    assert cache._idx == processed - trimmed
    assert cache.left_padding.tolist() == [9 - trimmed]
    assert cache.offset.tolist() == [processed - 9]
    if kind in ("minimax", "qsa"):
        assert cache.index_offset == processed - trimmed


def test_remove_right_padded_prefill_preserves_finished_rows():
    class EchoModel:
        def __call__(self, inputs, **kwargs):
            return SimpleNamespace(logits=mx.eye(16)[inputs])

    batch = PromptProcessingBatch(
        model=EchoModel(),
        uids=[10, 11, 12],
        input_ids=[[1, 2, 3, 4, 5, 6, 7], [5, 6], [6, 7, 8]],
        max_tokens=[1, 1, 1],
        inputs_embeds=mx.zeros((3, 7, 4)),
        prompt_kwargs={
            "position_ids": mx.broadcast_to(mx.arange(7)[None, None], (3, 3, 7)),
            "rope_deltas": mx.array([[10], [11], [12]]),
        },
        prefill_step_size=2,
        warm_cache=[],
        right_pad_per_row=[0, 5, 4],
    )
    assert batch.prompt_step() == 2
    # Row 11's final logits have already been saved before removing row 10.
    batch.remove(10)
    assert batch._prompt_kwargs["position_ids"].shape == (3, 2, 5)
    assert batch._prompt_kwargs["rope_deltas"].tolist() == [[11], [12]]
    while batch.needs_processing():
        batch.prompt_step()
    decoded = batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    responses = decoded.next()
    assert [(r.uid, r.token) for r in responses] == [(11, 6), (12, 8)]


def test_speculative_prefill_filter_keeps_hidden_chunks_aligned():
    drafter = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[0]))
    prefill = SpeculativePrefill("dflash", drafter)
    prefill.append(SimpleNamespace(hidden_states=[mx.array([[[1]], [[2]]])]))
    prefill.filter([1])
    output = prefill.finish(SimpleNamespace(hidden_states=[mx.array([[[3]]])]))
    assert output.hidden_states[0].tolist() == [[[2], [3]]]
    prefill.append(output)
    prefill.filter([])
    assert prefill.chunks == []
