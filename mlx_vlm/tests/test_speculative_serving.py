"""Serving contracts shared by every native MTP adapter."""

from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.generate.ar import BatchGenerator, generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.speculative.prefix_cache import SpeculativePrefixCache
from mlx_vlm.tests.test_speculative import models as glm_models


@pytest.fixture(autouse=True)
def small_checkpoints(monkeypatch):
    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")


def qwen_models(moe=False):
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import Model, ModelConfig

    mx.random.seed(42)
    config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=96,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=3,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=128,
        head_dim=32,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    if moe:
        from dataclasses import asdict

        from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoEConfig
        from mlx_vlm.models.qwen3_5_moe.language import (
            LanguageModel as MoELanguageModel,
        )

        args = asdict(config)
        args.update(
            model_type="qwen3_5_moe_text",
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=32,
            moe_intermediate_size=32,
        )
        config = MoEConfig.from_dict(args)
        LanguageModel = MoELanguageModel
    target = LanguageModel(
        config,
        SimpleNamespace(
            text_config=config,
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=30,
            video_token_id=31,
            vision_start_token_id=29,
        ),
    )
    draft = Model(ModelConfig(text_config=config))
    target.eval()
    draft.eval()
    return target, draft


@pytest.fixture(
    params=[glm_models, qwen_models, partial(qwen_models, True)],
    ids=["glm", "qwen", "qwen_moe"],
)
def pair(request):
    return request.param()


def wrapper(target):
    return SimpleNamespace(
        language_model=target,
        config=target.config,
        get_input_embeddings=lambda inputs, *args, **kwargs: InputEmbeddingsFeatures(
            inputs_embeds=target.model.embed_tokens(inputs)
        ),
    )


def generate(target, draft, prompt, *, state=None, prefix=None, **kwargs):
    full = mx.array([prompt])
    suffix = full if state is None else full[:, int(state.position.item()) :]
    holder = []
    kwargs.setdefault("max_tokens", 12)
    kwargs.setdefault("draft_block_size", 4)
    kwargs.setdefault("prefill_step_size", 2)
    tokens = [
        token
        for token, _ in generate_step(
            suffix,
            wrapper(target),
            None,
            None,
            draft_model=draft,
            full_prompt_tokens=full,
            speculative_cache=state,
            speculative_cache_callback=holder.append,
            speculative_checkpoint=(
                (lambda state: prefix.store(prompt, state)) if prefix else None
            ),
            **kwargs,
        )
    ]
    return tokens, holder[0] if holder else None


@pytest.mark.parametrize("temperature", [0, 0.8])
@pytest.mark.parametrize("chunk", [1, 3, 32])
def test_processors_see_full_committed_history(pair, temperature, chunk):
    target, draft = pair
    prompt = [1, 2, 1, 3, 4, 2]
    seen = []

    def processor(history, scores):
        seen.append(history.tolist())
        # Force a different token at every position, exercising draft rejection.
        wanted = (sum(history.tolist()) + len(history)) % scores.shape[-1]
        return mx.where(mx.arange(scores.shape[-1])[None] == wanted, 0, -mx.inf)

    kwargs = dict(
        temperature=temperature,
        seed=33,
        top_p=0.9,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.2,
        logit_bias={5: 0.5},
        prefill_step_size=chunk,
        logits_processors=[processor],
    )
    expected, _ = generate(target, None, prompt, **kwargs)
    ar_seen = seen[:]
    seen.clear()
    actual, _ = generate(target, draft, prompt, **kwargs)
    assert actual == expected
    assert seen == ar_seen[: len(actual)]
    assert seen == [prompt + actual[:n] for n in range(len(actual))]


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_prefix_reuse_restores_both_caches_and_seed(pair, temperature):
    target, draft = pair
    manager = APCManager(num_blocks=1, block_size=4)
    prefix = SpeculativePrefixCache(manager, target, draft)
    prompt = [1, 2, 3, 4, 5, 6]
    kwargs = dict(temperature=temperature, seed=17, repetition_penalty=1.2)
    original, state = generate(target, draft, prompt, prefix=prefix, **kwargs)
    warm, position = prefix.lookup(prompt)
    assert position == len(prompt) - 1
    assert warm.position.item() == position
    assert warm.bonus.item() == prompt[-1]
    if hasattr(target, "_rope_deltas"):
        target._rope_deltas = mx.array([[99]])
    assert generate(target, draft, prompt, state=warm, **kwargs)[0] == original
    history = prompt + original
    assert prefix.store(history, state)
    warm, position = prefix.lookup(history + [7, 8, 9])
    assert position == len(history) - 1
    expected, _ = generate(target, None, history + [7, 8, 9], **kwargs)
    actual, _ = generate(target, draft, history + [7, 8, 9], state=warm, **kwargs)
    assert actual == expected
    manager.close()


def test_checkpoint_key_includes_pending_token_and_draft_identity(pair):
    target, draft = pair
    manager = APCManager(num_blocks=1)
    prefix = SpeculativePrefixCache(manager, target, draft)
    prompt = [1, 2, 3]
    generated, state = generate(target, draft, prompt, max_tokens=1)
    key = prompt + generated
    assert prefix.store(key, state)
    assert prefix.lookup(key)[1] == len(prompt)
    assert prefix.lookup(key[:-1] + [(key[-1] + 1) % 32]) == (None, 0)
    assert prefix.lookup(key, extra_hash=99) == (None, 0)
    assert SpeculativePrefixCache(manager, target, object()).lookup(key) == (None, 0)
    manager.close()


def test_speculative_checkpoint_disk_roundtrip(pair, tmp_path, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    target, draft = pair
    prompt = [1, 2, 3]
    generated, state = generate(target, draft, prompt, max_tokens=5)
    history = prompt + generated
    disk = DiskBlockStore(tmp_path, namespace="mtp")
    manager = APCManager(num_blocks=1, disk=disk)
    prefix = SpeculativePrefixCache(manager, target, draft)
    assert prefix.store(history, state)
    disk._q.join()
    manager.close()
    manager = APCManager(num_blocks=1, disk=DiskBlockStore(tmp_path, namespace="mtp"))
    prefix = SpeculativePrefixCache(manager, target, draft)
    warm, position = prefix.lookup(history + [5, 6])
    assert position == len(history) - 1
    assert mx.array_equal(warm.seed.hidden, state.seed.hidden).item()
    assert (
        generate(target, draft, history + [5, 6], state=warm)[0]
        == generate(target, None, history + [5, 6])[0]
    )
    manager.close()


class NeverStop:
    def add_eos_token_ids(self, _):
        pass

    def __call__(self, _):
        return False


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_server_mixed_warm_cold_rows_share_decode(pair, temperature):
    from mlx_vlm.generate.ar import SpeculativePromptBatch, _PositionedTargetSampler
    from mlx_vlm.sample_utils import make_logits_processors

    target, draft = pair
    manager = APCManager(num_blocks=1, block_size=4)
    kwargs = dict(seed=23, temperature=temperature, repetition_penalty=1.2)
    prompt = [1, 2, 3, 4, 5]
    _, state = generate(target, draft, prompt, max_tokens=1, **kwargs)
    # Use a prompt checkpoint whose shifted draft cache depends on the next input.
    prefix = SpeculativePrefixCache(manager, target, draft)
    key = prompt + [int(state.bonus.item())]
    assert prefix.store(key, state)
    prompts = [[8, 9, 10], key + [6, 7]]
    sampler = (
        None
        if temperature == 0
        else _PositionedTargetSampler(temperature=temperature, top_p=1.0, seed=23)
    )
    generator = BatchGenerator(
        target,
        SimpleNamespace(stopping_criteria=NeverStop()),
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=4,
        apc_manager=manager,
        prefill_batch_size=2,
        completion_batch_size=2,
        prefill_step_size=2,
        sampler=sampler,
    )
    processor = make_logits_processors(repetition_penalty=1.2)
    prompt_kwargs = [
        {
            "inputs_embeds": target.model.embed_tokens(mx.array([ids])),
            "_apc_semantic_hash": 0,
        }
        for ids in prompts
    ]
    ids = generator.insert(
        prompts,
        max_tokens=[5, 9],
        prompt_kwargs=prompt_kwargs,
        logits_processors=[processor, processor],
    )
    actual = {uid: [] for uid in ids}
    cached = {}
    saw_restored = False
    while generator.has_work:
        progress, responses = generator.next()
        saw_restored |= isinstance(generator._prompt_batch, SpeculativePromptBatch)
        for row in progress:
            cached[row.uid] = row.cached_tokens
        for response in responses:
            if response.token is not None:
                actual[response.uid].append(response.token)
    for uid, prompt, limit in zip(ids, prompts, [5, 9]):
        assert (
            actual[uid] == generate(target, None, prompt, max_tokens=limit, **kwargs)[0]
        )
    assert saw_restored
    assert cached[ids[1]] == len(key) - 1
    for uid, prompt in zip(ids, prompts):
        history = prompt + actual[uid]
        warm, position = prefix.lookup(history)
        assert position == len(history) - 1
        assert warm.bonus.item() == history[-1]
    generator.close()
    manager.close()


def test_stream_cancel_saves_only_delivered_prefix(pair):
    from mlx_vlm.generate.common import PromptCacheState
    from mlx_vlm.generate.dispatch import stream_generate
    from mlx_vlm.tests.test_generate import MockDetokenizer, MockTokenizer

    target, draft = pair
    target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
    model = wrapper(target)
    tokenizer = MockTokenizer()
    tokenizer.stopping_criteria = NeverStop()
    processor = SimpleNamespace(tokenizer=tokenizer, detokenizer=MockDetokenizer())
    request_cache = PromptCacheState()
    prompt = [1, 2, 3, 4, 5]
    stream = stream_generate(
        model,
        processor,
        prompt="",
        input_ids=mx.array([prompt]),
        pixel_values=None,
        mask=None,
        draft_model=draft,
        draft_block_size=4,
        prefill_step_size=2,
        prompt_cache_state=request_cache,
        max_tokens=20,
    )
    delivered = [next(stream).token, next(stream).token]
    stream.close()
    prefix = SpeculativePrefixCache(request_cache.speculative_manager, model, draft)
    # Request hashes also include processor/media semantics; the mock has none.
    history = prompt + delivered
    warm, position = prefix.lookup(history)
    assert position == len(history) - 1
    assert warm.bonus.item() == delivered[-1]
    assert not any(getattr(c, "is_speculating", False) for c in warm.target)
    results = list(
        stream_generate(
            model,
            processor,
            prompt="",
            input_ids=mx.array([history + [6]]),
            pixel_values=None,
            mask=None,
            draft_model=draft,
            draft_block_size=4,
            prefill_step_size=2,
            prompt_cache_state=request_cache,
            max_tokens=5,
        )
    )
    assert results[-1].cached_tokens == position
    assert results[-1].token_ids == [0] * 5
    request_cache.speculative_manager.close()


def test_processor_requiring_external_updates_yields_each_token(pair):
    from mlx_vlm.generate.ar import PromptProcessingBatch

    class Processor:
        requires_immediate_decode_yield = True
        position = 0

        def __call__(self, context, scores):
            return mx.where(
                mx.arange(scores.shape[-1])[None] == self.position, 0, -mx.inf
            )

    target, draft = pair
    processor = Processor()
    prompt = [1, 2, 3]
    processing = PromptProcessingBatch(
        target,
        [0],
        [prompt],
        [8],
        target.model.embed_tokens(mx.array([prompt])),
        {},
        logits_processors=[[processor]],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=4,
        greedy_sampling=True,
    )
    while processing.needs_processing():
        processing.prompt_step()
    batch = processing.generate(lambda scores: mx.argmax(scores, axis=-1), NeverStop())
    actual = []
    while len(batch):
        responses = batch.next()
        assert len(responses) == 1
        actual.append(responses[0].token)
        processor.position += 1
    assert actual == list(range(8))


def test_transaction_detects_replaced_caches():
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.speculative.cache_state import CacheTransaction

    caches = [KVCache()]
    with CacheTransaction(caches, 1) as transaction:
        caches[0] = KVCache()
        with pytest.raises(RuntimeError, match="replaced cache objects"):
            transaction.commit([1])


@pytest.mark.parametrize("bits", [None, 4])
def test_ragged_commit_invalidates_attention_padding_metadata(bits):
    from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache
    from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask
    from mlx_vlm.speculative.cache_state import CacheTransaction

    cache = (
        BatchKVCache([0, 1])
        if bits is None
        else BatchQuantizedKVCache([0, 1], bits=bits)
    )
    keys = mx.zeros((2, 1, 3, 64))
    cache.update_and_fetch(keys, keys)
    query = mx.zeros((2, 1, 64))
    _create_qwen3_5_attention_mask(query, cache)
    assert cache._qwen3_5_decode_left_padding == [0, 1]
    old_padding = cache.left_padding
    with CacheTransaction([cache], 2) as transaction:
        cache.update_and_fetch(keys[:, :, :2], keys[:, :, :2])
        transaction.commit([1, 2])
    _create_qwen3_5_attention_mask(query, cache)
    assert cache._qwen3_5_decode_left_padding == [1, 1]
    assert old_padding.tolist() == [0, 1]


@pytest.mark.parametrize("moe", [False, True])
def test_decode_positions_belong_to_request_cache(moe):
    target, draft = qwen_models(moe)
    prompt = mx.array([[1, 2, 3, 4, 5]])
    expected, _ = generate(target, None, prompt[0].tolist(), max_tokens=12)
    stream = generate_step(
        prompt,
        wrapper(target),
        None,
        None,
        draft_model=draft,
        max_tokens=12,
        draft_block_size=4,
        prefill_step_size=2,
    )
    actual = [next(stream)[0]]
    # Another request can overwrite the model's legacy positional scratch data.
    target._rope_deltas = mx.array([[99]])
    target._position_ids = None
    actual.extend(token for token, _ in stream)
    assert actual == expected
