import json
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.generate.ar import _make_cache, _PositionedTargetSampler
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.gated_delta import gated_delta_update
from mlx_vlm.models.glm5_next import language as glm5_next_language
from mlx_vlm.speculative.cache_state import (
    CacheTransaction,
    SpeculativeCache,
    SpeculativePrefill,
    iter_leaf_caches,
)
from mlx_vlm.speculative.drafters.glm5_next_mtp import Glm5NextMTPDraftModel
from mlx_vlm.speculative.drafters.glm5_next_mtp import ModelConfig as Glm5NextMTPConfig
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import split_glm5_next_mtp
from mlx_vlm.speculative.mtp import mtp_rounds
from mlx_vlm.split_mtp import split_mtp


def models():
    mx.random.seed(35)
    config = _tiny_glm5_next_text_config()
    target = glm5_next_language.LanguageModel(config)
    draft = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=config))
    target.eval()
    draft.eval()
    return target, draft


@pytest.mark.parametrize("retained", [[0, 0], [1, 3], [3, 1], [2, 2], [3, 3]])
def test_cache_commit_matches_independent_prefixes(retained):
    target, _ = models()
    prefix = mx.array([[0, 1, 2], [3, 4, 5]])
    block = mx.array([[6, 7, 8], [9, 10, 11]])
    caches = _make_cache(target, [1, 0])
    mx.eval(target(prefix, cache=caches, attention_mask=prefix != 0).logits)
    with CacheTransaction(caches, 3) as transaction:
        mx.eval(target(block, cache=caches).logits)
        transaction.commit(retained)
    probe = mx.array([[12], [13]])
    actual = target(probe, cache=caches).logits
    for row, keep in enumerate(retained):
        reference = target.make_cache()
        start = 1 if row == 0 else 0
        target(prefix[row : row + 1, start:], cache=reference)
        if keep:
            target(block[row : row + 1, :keep], cache=reference)
        expected = target(probe[row : row + 1], cache=reference).logits
        assert mx.allclose(actual[row : row + 1], expected, atol=2e-5).item()
    assert all(
        not getattr(c, "is_speculating", False) for c in iter_leaf_caches(caches)
    )


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("all_accepted", [False, True])
def test_mtp_generation_matches_ar(depth, batch, all_accepted):
    target, draft = models()
    if all_accepted:
        target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
    prompt = mx.array([[1, 2, 3], [0, 4, 5]][:batch])
    pads = [0, 1][:batch]
    caches = _make_cache(target, pads)
    out = target(prompt, cache=caches, return_hidden=True, attention_mask=prompt != 0)
    first = mx.argmax(out.logits[:, -1], axis=-1)
    actual = [[t] for t in first.tolist()]
    for tokens, _ in mtp_rounds(
        target,
        draft,
        caches,
        out.hidden_states[-1],
        prompt_tokens=prompt,
        first_bonus=first,
        max_tokens=10,
        sampler=None,
        greedy_sampling=True,
        draft_block_size=depth + 1,
    ):
        for row, token in enumerate(tokens):
            if token is not None:
                actual[row].append(token)
    for row in range(batch):
        cache = target.make_cache()
        out = target(prompt[row : row + 1, pads[row] :], cache=cache)
        expected = []
        for _ in range(10):
            token = mx.argmax(out.logits[:, -1], axis=-1)
            expected.append(token.item())
            out = target(token[:, None], cache=cache)
        assert actual[row] == expected
    assert not hasattr(draft, "_cache")
    assert not hasattr(draft, "_seed_hidden")


def test_cancel_commits_only_delivered_prefix():
    target, draft = models()
    target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
    caches = target.make_cache()
    prompt = mx.array([[1, 2, 3]])
    out = target(prompt, cache=caches, return_hidden=True, attention_mask=prompt != 0)
    iterator = mtp_rounds(
        target,
        draft,
        caches,
        out.hidden_states[-1],
        prompt_tokens=prompt,
        first_bonus=mx.array([0]),
        max_tokens=10,
        sampler=None,
        greedy_sampling=True,
        draft_block_size=4,
    )
    assert next(iterator)[0] == [0]
    iterator.close()
    assert caches[1][0].offset == 4
    assert all(
        not getattr(c, "is_speculating", False) for c in iter_leaf_caches(caches)
    )


def test_cache_aborts_failed_verification():
    target, draft = models()
    caches = target.make_cache()
    prompt = mx.array([[1, 2, 3]])
    out = target(prompt, cache=caches, return_hidden=True, attention_mask=prompt != 0)
    forward = partial(draft, target_model=target)
    state = SpeculativeCache(caches, draft.make_cache(), [0], mx.array([4]))
    state.prefill(prompt, out.hidden_states[-1], forward)
    proposals = state.propose(3, forward)
    target(state.verify_inputs(proposals), cache=caches)
    state.abort()
    assert caches[1][0].offset == 3
    assert state.draft[0][0].offset == 3
    assert state.position.tolist() == [3]


def test_target_sampling_preserves_positioned_draws():
    target, draft = models()
    prompt = mx.array([[1, 2, 3]])
    sampler = _PositionedTargetSampler(temperature=0.8, top_p=0.9, seed=123)
    caches = target.make_cache()
    out = target(prompt, cache=caches, return_hidden=True, attention_mask=prompt != 0)
    scores = out.logits[:, -1]
    first = sampler.sample_target(
        scores - mx.logsumexp(scores, axis=-1, keepdims=True),
        row_ids=[0],
        positions=[0],
    )
    actual = [first.item()]
    for tokens, _ in mtp_rounds(
        target,
        draft,
        caches,
        out.hidden_states[-1],
        prompt_tokens=prompt,
        first_bonus=first,
        max_tokens=12,
        sampler=sampler,
        draft_block_size=4,
    ):
        actual.append(tokens[0])
    cache = target.make_cache()
    out = target(prompt, cache=cache)
    expected = []
    for position in range(12):
        scores = out.logits[:, -1]
        token = sampler.sample_target(
            scores - mx.logsumexp(scores, axis=-1, keepdims=True),
            row_ids=[0],
            positions=[position],
        )
        expected.append(token.item())
        out = target(token[:, None], cache=cache)
    assert actual == expected


def test_chunked_prefill_retains_all_target_features():
    prefill = SpeculativePrefill("mtp", object())
    prefill.append(LanguageModelOutput(logits=None, hidden_states=[mx.ones((1, 3, 4))]))
    output = prefill.finish(
        LanguageModelOutput(logits=None, hidden_states=[mx.zeros((1, 2, 4))])
    )
    assert output.hidden_states[-1].shape == (1, 5, 4)
    assert not prefill.chunks


@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_public_generate_step_with_chunked_mtp(chunk):
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.models.base import InputEmbeddingsFeatures

    target, draft = models()
    model = SimpleNamespace(
        language_model=target,
        config=target.config,
        get_input_embeddings=lambda inputs, *args, **kwargs: InputEmbeddingsFeatures(
            inputs_embeds=target.model.embed_tokens(inputs)
        ),
    )
    prompt = mx.array([[1, 2, 3, 4, 5]])

    def generate(drafter):
        return [
            token
            for token, _ in generate_step(
                prompt,
                model,
                None,
                None,
                max_tokens=8,
                temperature=0,
                prefill_step_size=chunk,
                draft_model=drafter,
                draft_kind="mtp",
            )
        ]

    assert generate(draft) == generate(None)


def test_server_chunked_mtp_respects_individual_limits_and_commits_rounds():
    from mlx_vlm.generate.ar import PromptProcessingBatch

    target, draft = models()
    target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
    prompt = mx.array([[1, 2, 3, 4, 5], [0, 0, 0, 6, 7]])
    processing = PromptProcessingBatch(
        model=target,
        uids=[10, 20],
        input_ids=[[1, 2, 3, 4, 5], [6, 7]],
        max_tokens=[3, 8],
        inputs_embeds=target.model.embed_tokens(prompt),
        prompt_kwargs={"attention_mask": prompt != 0},
        prefill_step_size=2,
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=4,
        greedy_sampling=True,
    )
    while processing.needs_processing():
        processing.prompt_step()
    batch = processing.generate(
        lambda scores: mx.argmax(scores, axis=-1), lambda _: False
    )
    counts = {10: 0, 20: 0}
    while len(batch):
        for response in batch.next():
            if response.token is not None:
                counts[response.uid] += 1
        assert all(
            not getattr(c, "is_speculating", False)
            for c in iter_leaf_caches(batch.prompt_cache)
        )
    assert counts == {10: 3, 20: 8}
    assert batch.prompt_cache[1][0].offset.tolist() == [7, 9]


def test_two_requests_share_draft_weights_without_sharing_state():
    target, draft = models()

    def request(prompt):
        caches = target.make_cache()
        out = target(prompt, cache=caches, return_hidden=True)
        first = mx.argmax(out.logits[:, -1], axis=-1)
        return mtp_rounds(
            target,
            draft,
            caches,
            out.hidden_states[-1],
            prompt_tokens=prompt,
            first_bonus=first,
            max_tokens=8,
            sampler=None,
            greedy_sampling=True,
            draft_block_size=3,
        )

    prompts = [mx.array([[1, 2, 3]]), mx.array([[4, 5]])]
    expected = [[tokens[0] for tokens, _ in request(p)] for p in prompts]
    iterators = [request(p) for p in prompts]
    actual = [[], []]
    for _ in range(7):
        for row, iterator in enumerate(iterators):
            actual[row].append(next(iterator)[0][0])
    for iterator in iterators:
        iterator.close()
    assert actual == expected


def test_failed_draft_replay_restores_both_caches():
    target, draft = models()
    prompt = mx.array([[1, 2, 3]])
    caches = target.make_cache()
    out = target(prompt, cache=caches, return_hidden=True)
    state = SpeculativeCache(caches, draft.make_cache(), [0], mx.array([4]))
    forward = partial(draft, target_model=target)
    state.prefill(prompt, out.hidden_states[-1], forward)
    proposals = state.propose(3, forward)
    out = target(state.verify_inputs(proposals), cache=caches, return_hidden=True)
    state.record_verification(out.hidden_states[-1])

    def fail(*args, **kwargs):
        forward(*args, **kwargs)
        raise RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        state.commit([[5, 6]], fail)
    assert caches[1][0].offset == 3
    assert state.draft[0][0].offset == 3


def test_eos_inside_accepted_block_stops_at_committed_prefix():
    target, draft = models()
    target.lm_head.weight = mx.zeros_like(target.lm_head.weight)
    caches = target.make_cache()
    prompt = mx.array([[1, 2, 3]])
    output = target(prompt, cache=caches, return_hidden=True)
    rounds = mtp_rounds(
        target,
        draft,
        caches,
        output.hidden_states[-1],
        prompt_tokens=prompt,
        first_bonus=mx.array([1]),
        max_tokens=10,
        sampler=None,
        greedy_sampling=True,
        draft_block_size=4,
        eos_token_ids={0},
    )
    assert [tokens for tokens, _ in rounds] == [[0]]
    assert caches[1][0].offset == 4


@pytest.mark.parametrize("batched", [False, True])
def test_empty_optional_kv_cache_state_roundtrips(batched):
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    cache = BatchKVCache([0, 2]) if batched else KVCache()
    restored = type(cache).from_state(cache.state, cache.meta_state)
    assert restored.empty()
    assert restored.size() == 0
    assert restored.state[:2] == (None, None)


def test_unsupported_cache_fails_before_mutating_other_caches():
    from mlx_vlm.models.cache import ArraysCache, RotatingKVCache

    recurrent = ArraysCache(2)
    with pytest.raises(ValueError, match="RotatingKVCache"):
        CacheTransaction([recurrent, RotatingKVCache(8)], 2)
    assert not recurrent.is_speculating


@pytest.mark.parametrize("bits", [4, 8])
def test_native_quantized_kv_transaction_retains_each_row_prefix(bits):
    from mlx_vlm.models.cache import BatchQuantizedKVCache

    cache = BatchQuantizedKVCache([1, 0], bits=bits, group_size=64)
    prefix = mx.random.normal((2, 1, 3, 64)).astype(mx.bfloat16)
    block = mx.random.normal((2, 1, 4, 64)).astype(mx.bfloat16)
    cache.update_and_fetch(prefix, prefix)
    with CacheTransaction([cache], 4) as transaction:
        cache.update_and_fetch(block, block)
        transaction.commit([1, 3])
    assert cache.offset.tolist() == [3, 6]
    for row, (pad, keep) in enumerate([(1, 1), (0, 3)]):
        expected = mx.quantize(
            mx.concatenate(
                [prefix[row : row + 1, :, pad:], block[row : row + 1, :, :keep]], axis=2
            ),
            group_size=64,
            bits=bits,
        )
        start = cache.left_padding[row].item()
        for actual, reference in zip(cache.state[0], expected):
            assert mx.array_equal(actual[row : row + 1, :, start:], reference).item()


def test_mtp_statistics_count_partial_rounds_and_snapshots():
    from mlx_vlm.speculative.stats import (
        record_round,
        speculative_stats_since,
        speculative_stats_snapshot,
    )

    draft = SimpleNamespace()
    snapshot = speculative_stats_snapshot(draft)
    record_round(draft, mx.array([[1, 2, 3], [4, 5, 6]]), [[1, 9], []])
    assert speculative_stats_since(draft, snapshot) == (1, 1, 3)
    snapshot = speculative_stats_snapshot(draft)
    record_round(draft, mx.array([[1, 2]]), [[1, 2, 3]])
    assert speculative_stats_since(draft, snapshot) == (1, 2, 2)


def _tiny_glm5_next_text_config():
    from mlx_vlm.models.glm5_next.config import TextConfig

    return TextConfig(
        vocab_size=32,
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
        hc_mult=2,
        max_position_embeddings=64,
    )


@pytest.mark.parametrize("batch", [1, 4])
def test_glm5_next_cached_indexer_block_matches_stepwise(batch):
    mx.random.seed(4700 + batch)
    text_config = _tiny_glm5_next_text_config()
    language = glm5_next_language.LanguageModel(text_config)
    language.eval()
    indexer = language.model.layers[1].self_attn.indexer
    step_cache = _make_cache(language, left_padding=[0] * batch)[1]
    block_cache = _make_cache(language, left_padding=[0] * batch)[1]

    prefix_length = 5
    prefix = mx.random.normal((batch, prefix_length, text_config.hidden_size)).astype(
        mx.bfloat16
    )
    prefix_q = mx.random.normal((batch, prefix_length, text_config.q_lora_rank)).astype(
        mx.bfloat16
    )
    indexer(
        prefix,
        prefix_q,
        cache=step_cache[1],
        pool_cache=step_cache[2],
        offset=0,
    )
    indexer(
        prefix,
        prefix_q,
        cache=block_cache[1],
        pool_cache=block_cache[2],
        offset=0,
    )

    inputs = mx.random.normal((batch, 2, text_config.hidden_size)).astype(mx.bfloat16)
    q_resid = mx.random.normal((batch, 2, text_config.q_lora_rank)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [
            indexer(
                inputs[:, position : position + 1],
                q_resid[:, position : position + 1],
                cache=step_cache[1],
                pool_cache=step_cache[2],
                offset=prefix_length + position,
            )
            for position in range(2)
        ],
        axis=1,
    )
    actual = indexer(
        inputs,
        q_resid,
        cache=block_cache[1],
        pool_cache=block_cache[2],
        offset=prefix_length,
    )
    mx.eval(expected, actual, step_cache[1].state, block_cache[1].state)

    assert mx.array_equal(actual, expected).item()
    assert step_cache[2].remainder == block_cache[2].remainder
    assert step_cache[2]._pool_lengths == block_cache[2]._pool_lengths
    for cache_index in (1, 2):
        for (_, step_value), (_, block_value) in zip(
            tree_flatten(step_cache[cache_index].state),
            tree_flatten(block_cache[cache_index].state),
            strict=True,
        ):
            if step_value is None or block_value is None:
                assert step_value is block_value
            else:
                assert mx.array_equal(step_value, block_value).item(), (
                    cache_index,
                    mx.max(mx.abs(step_value - block_value)).item(),
                )


@pytest.mark.parametrize("batch", [1, 8])
def test_glm5_next_gated_delta_captured_states_match_stepwise(batch):
    mx.random.seed(800 + batch)
    length, heads, width, value_width = 3, 2, 64, 8
    q = mx.random.normal((batch, length, heads, width)).astype(mx.bfloat16)
    k = mx.random.normal((batch, length, heads, width)).astype(mx.bfloat16)
    v = mx.random.normal((batch, length, heads, value_width)).astype(mx.bfloat16)
    a = mx.random.normal((batch, length, heads, width)).astype(mx.bfloat16)
    b = mx.random.normal((batch, length, heads)).astype(mx.bfloat16)
    A_log = mx.random.normal((heads, 1)).astype(mx.float32)
    dt_bias = mx.random.normal((heads, width)).astype(mx.float32)
    initial = mx.zeros((batch, heads, value_width, width), dtype=mx.float32)

    outputs = []
    states = []
    state = initial
    for position in range(length):
        output, state = gated_delta_update(
            q[:, position : position + 1],
            k[:, position : position + 1],
            v[:, position : position + 1],
            a[:, position : position + 1],
            b[:, position : position + 1],
            A_log,
            dt_bias,
            state=state,
            lower_bound=-5.0,
        )
        outputs.append(output)
        states.append(state)

    expected_output = mx.concatenate(outputs, axis=1)
    expected_states = mx.stack(states[:-1], axis=1)
    output, final_state, captured_states = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state=initial,
        lower_bound=-5.0,
        state_steps=length - 1,
    )
    mx.eval(
        expected_output,
        state,
        expected_states,
        output,
        final_state,
        captured_states,
    )

    assert mx.array_equal(output, expected_output).item()
    assert mx.array_equal(final_state, state).item()
    assert mx.array_equal(captured_states, expected_states).item()


def test_glm5_next_mtp_sanitize_fuses_native_layer_weights():
    config = _tiny_glm5_next_text_config()
    context = SimpleNamespace(args=config)
    weights = {
        "mtp_block.mlp.shared_experts.gate_proj.weight": mx.zeros((8, 16)),
        "mtp_block.mlp.shared_experts.up_proj.weight": mx.zeros((8, 16)),
        "mtp_block.self_attn.q_a_proj.weight": mx.zeros((8, 16)),
        "mtp_block.self_attn.kv_a_proj_with_mqa.weight": mx.zeros((4, 16)),
        "mtp_block.self_attn.kv_b_proj.weight": mx.zeros((16, 4)),
    }
    for expert in range(config.n_routed_experts):
        weights[f"mtp_block.mlp.experts.{expert}.gate_proj.weight"] = mx.zeros((8, 16))
        weights[f"mtp_block.mlp.experts.{expert}.up_proj.weight"] = mx.zeros((8, 16))
        weights[f"mtp_block.mlp.experts.{expert}.down_proj.weight"] = mx.zeros((16, 8))

    out = Glm5NextMTPDraftModel.sanitize(context, weights)

    assert out["mtp_block.mlp.shared_experts.gate_up_proj.weight"].shape == (
        16,
        16,
    )
    assert out["mtp_block.mlp.switch_mlp.gate_proj.weight"].shape == (2, 8, 16)
    assert out["mtp_block.self_attn.qkv_a_proj.weight"].shape == (12, 16)
    assert out["mtp_block.self_attn.embed_q.weight"].shape == (2, 4, 4)
    assert out["mtp_block.self_attn.unembed_out.weight"].shape == (2, 4, 4)


def test_split_glm5_next_mtp_extracts_layer_after_target_stack(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_glm5_next_text_config()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": text_config.to_dict(),
            }
        )
    )
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            f"{prefix}.enorm.weight": mx.ones((16,)),
            f"{prefix}.hnorm.weight": mx.ones((16,)),
            f"{prefix}.eh_proj.weight": mx.ones((16, 32)),
            f"{prefix}.shared_head.norm.weight": mx.ones((16,)),
        },
    )

    split_glm5_next_mtp(str(source), str(output))

    config = json.loads((output / "config.json").read_text())
    weights = mx.load(str(output / "model.safetensors"))
    assert config["model_type"] == "glm5_next_mtp"
    assert config["block_size"] == 2
    assert set(weights) == {
        "eh_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "shared_head_norm.weight",
    }


def test_split_glm5_next_mtp_honors_requested_quantization_for_fp8(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_glm5_next_text_config()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": text_config.to_dict(),
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            f"{prefix}.eh_proj.weight": mx.to_fp8(
                mx.ones((128, 128), dtype=mx.bfloat16)
            ),
            f"{prefix}.eh_proj.weight_scale_inv": mx.full(
                (1, 1), 0.125, dtype=mx.bfloat16
            ),
        },
    )

    split_mtp(str(source), str(output), q_bits=4, q_group_size=64)

    config = json.loads((output / "config.json").read_text())
    weights = mx.load(str(output / "model.safetensors"))
    expected = {"group_size": 64, "bits": 4, "mode": "affine"}
    assert config["quantization"] == expected
    assert config["quantization_config"] == expected
    assert weights["eh_proj.weight"].dtype == mx.uint32
    assert weights["eh_proj.scales"].dtype == mx.bfloat16
    assert weights["eh_proj.biases"].dtype == mx.bfloat16
    assert not any(key.endswith("weight_scale_inv") for key in weights)


def test_split_glm5_next_mtp_supports_independent_mxfp8_quantization(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_glm5_next_text_config()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": text_config.to_dict(),
            }
        )
    )
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            f"{prefix}.eh_proj.weight": mx.ones((128, 128), dtype=mx.bfloat16),
        },
    )

    split_mtp(str(source), str(output), q_mode="mxfp8")

    config = json.loads((output / "config.json").read_text())
    weights = mx.load(str(output / "model.safetensors"))
    expected = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    assert config["quantization"] == expected
    assert config["quantization_config"] == expected
    assert weights["eh_proj.weight"].dtype == mx.uint32
    assert weights["eh_proj.scales"].dtype == mx.uint8
    assert "eh_proj.biases" not in weights
