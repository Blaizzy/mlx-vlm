"""Shared speculative generation, drafter, verification, and cache contracts."""

import json
from contextlib import nullcontext
from copy import deepcopy
from itertools import product
from types import SimpleNamespace as NS
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import mlx_vlm.speculative.utils as speculative
from mlx_vlm.generate.ar import _make_cache, generate_step
from mlx_vlm.models import fast_ops
from mlx_vlm.models import quantized_verifier as quantized
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.models.linear import native_batch_linear
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_vlm.speculative import common, mtp
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.ops import linear as verifier_linear
from mlx_vlm.speculative.utils import _mtp_verify_target, _speculative_walk_batch
from mlx_vlm.split_mtp import split_mtp
from mlx_vlm.tests import test_models as models

parametrize = pytest.mark.parametrize
module = models.module


def equal(actual, expected, **tolerance):
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b, **tolerance)
    elif actual is None or expected is None:
        assert actual is expected
    else:
        compare = mx.allclose if tolerance else mx.array_equal
        assert compare(actual, expected, **tolerance).item()


def greedy(logits):
    return mx.argmax(logits, axis=-1)


def generated(target, drafter=None, *, seed=41, temperature=0):
    if seed is None:
        mx.random.seed(47)
    options = {} if drafter is None else dict(draft_model=drafter, draft_kind="dflash")
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            mx.array([[1, 2, 3, 4]]),
            target,
            None,
            None,
            max_tokens=24 if seed is None else 12,
            temperature=temperature,
            seed=seed,
            prefill_step_size=None,
            **options,
        )
    ]


@parametrize(
    "family", ["dflash2", "glimmer", "dspark-lfm2", "dspark-lfm2-moe", "dspark-qwen"]
)
@parametrize("temperature,seed", [(0, 41), (0.5, 41), (1.0, 41), (0.7, None)])
def test_generation_and_request_reset(family, temperature, seed):
    mx.random.seed(37)
    target, drafter = models.dflash_target(family), models.dflash_drafter(family)
    mx.eval(target.language_model.parameters(), drafter.parameters())
    validate_drafter_compatibility(target, drafter, "dflash")
    expected = generated(target, temperature=temperature, seed=seed)
    for _ in range(2):
        assert (
            generated(target, drafter, temperature=temperature, seed=seed) == expected
        )
        assert drafter.draft_lens
        assert 999 not in drafter.accept_lens + drafter.draft_lens
        drafter.accept_lens.append(999)
        drafter.draft_lens.append(999)


@parametrize(
    "family,failure",
    [("qwen", False), ("glm", False), ("deepseek", False), ("glm", True)],
)
def test_mtp_generation(family, failure, monkeypatch):
    mx.random.seed(2127)
    model, config = models.language(family, inference=True)
    drafter = models.mtp_drafter(family, config)
    model.eval()
    drafter.eval()
    prompt = mx.array([[1, 2, 3]])
    ordinary, speculative = model.make_cache(), model.make_cache()
    expected, inputs = [], prompt
    for _ in range(8):
        token = greedy(model(inputs, cache=ordinary).logits[:, -1])
        expected.append(token.item())
        inputs = token[:, None]
    out = model(prompt, cache=speculative, return_hidden=True, return_shared_kv=True)
    first = greedy(out.logits[:, -1]).item()
    if failure:

        def fail(*args, **kwargs):
            assert drafter._round_appended == 2 and drafter._round_transaction.active
            raise RuntimeError("injected target failure")

        monkeypatch.setattr(module("speculative.mtp"), "_mtp_verify_target", fail)
    rounds = mtp._mtp_rounds(
        model,
        drafter,
        speculative,
        out.hidden_states[-1],
        {},
        prompt_tokens=prompt,
        first_bonus=first,
        max_tokens=8,
        sampler=greedy,
        draft_block_size=4,
        greedy_sampling=True,
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected target failure"):
            next(rounds)
        assert drafter._round_transaction is None and drafter._next_position == 3
        assert (
            drafter._seed_token is not None and not drafter._cache[0][2].is_speculating
        )
        assert module("speculative.mtp")._mtp_cache_offset_max(speculative) == 3
        return
    actual = [first] + [token for token, _ in rounds]
    assert actual == expected
    assert drafter._round_appended == 0


@parametrize("family", list(models.TEXT))
@parametrize("batch", [1, 2, 4])
@parametrize("dtype", [mx.float32, mx.bfloat16])
def test_verify_commit_matches_decode(family, batch, dtype):
    mx.random.seed(2127)
    model, _ = models.language(family, inference=True)
    model.load_weights(
        [
            (key, value.astype(dtype) if model.cast_predicate(key) else value)
            for key, value in tree_flatten(model.parameters())
        ]
    )
    model.eval()
    reference, actual = [_make_cache(model, [0] * batch) for _ in range(2)]
    prefix = mx.broadcast_to((mx.arange(19)[None] % 30) + 1, (batch, 19))
    for cache in (reference, actual):
        mx.eval(model(prefix, cache=cache).logits)

    def verify(tokens, cache):
        result = _mtp_verify_target(
            model, tokens, cache, None, sample_target_tokens=False
        )
        return (
            result.hidden,
            mtp._mtp_logits_from_hidden(model, result.hidden),
            result.rollback_state,
        )

    def compare(actual, expected):
        # Float32 recurrent reductions allow rounding; emitted tokens remain exact.
        if family == "qwen" and dtype == mx.float32:
            assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6).item()
        else:
            assert mx.array_equal(actual, expected).item()

    # Each model owns its verifier; compare blocks with one-token verifier calls.
    for retained in (1, 4, 3):
        tokens = mx.broadcast_to(mx.array([[4, 5, 6, 7]]), (batch, 4))
        oracle = deepcopy(reference)
        hidden_steps, logit_steps = [], []
        for i in range(4):
            hidden, logits, transaction = verify(tokens[:, i : i + 1], oracle)
            hidden_steps.append(hidden)
            logit_steps.append(logits)
            transaction.commit([1] * batch)
        hidden, logits, transaction = verify(tokens, actual)
        compare(hidden, mx.concatenate(hidden_steps, 1))
        compare(logits, mx.concatenate(logit_steps, 1))
        equal(greedy(logits), greedy(mx.concatenate(logit_steps, 1)))
        transaction.commit([retained] * batch)
        for i in range(retained):
            _, _, transaction = verify(tokens[:, i : i + 1], reference)
            transaction.commit([1] * batch)
        probe = mx.full((batch, 1), 20)
        compare(model(probe, cache=reference).logits, model(probe, cache=actual).logits)


def formats(groups=(64,)):
    return [
        ("affine", bits, size) for size in groups for bits in (2, 3, 4, 5, 6, 8)
    ] + [("mxfp4", 4, 32), ("mxfp8", 8, 32), ("nvfp4", 4, 16)]


def bf16_parameters(linear):
    linear.scales = linear.scales.astype(mx.bfloat16)
    if linear.biases is not None:
        linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


@parametrize("mode,bits,size", formats())
@parametrize("batch", [1, 2, 4, 5, 8, 9, 16, 32, 64, 127])
def test_quantized_linear_and_argmax(mode, bits, size, batch):
    mx.random.seed(100 + bits + batch)
    dense = nn.Linear(512, 16, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense, mode=mode, bits=bits, group_size=size
    )
    inputs = mx.random.normal((batch, 3, 512)).astype(mx.bfloat16)
    native = mx.concatenate(
        [linear(mx.contiguous(inputs[:, i : i + 1])) for i in range(3)], 1
    )
    expected = (
        verifier_linear._target_verify_singletons(linear, inputs)
        if mode == "nvfp4"
        else native
    )
    actual = quantized.decode_quantized_linear(linear, inputs)
    equal(native_batch_linear(linear, inputs), native)
    equal(actual, expected)
    equal(quantized.decode_quantized_argmax(linear, inputs), greedy(expected))
    allowed = mx.arange(batch * 3, dtype=mx.int32).reshape(batch, 3) % 16
    mask = (mx.array(1, dtype=mx.int32) << allowed).reshape(-1, 1)
    equal(quantized.decode_quantized_argmax(linear, inputs, token_mask=mask), allowed)


@parametrize("mode,bits,size", formats((32, 64, 128)))
@parametrize("batch", [1, 4, 8, 64, 127])
def test_quantized_moe_hyperconnection(mode, bits, size, batch, monkeypatch):
    mx.random.seed(600 + bits + batch)
    linear = QuantizedSwitchLinear(512, 16, 4, False, size, bits, mode=mode)
    if mode == "affine":
        bf16_parameters(linear)
    inputs = mx.random.normal((batch, 2, 2, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 4, dtype=mx.int32).reshape(batch, 2, 2) % 4
    weights = mx.softmax(mx.random.normal((batch, 2, 2)), axis=-1)
    shared = mx.random.normal((batch, 2, 16)).astype(mx.bfloat16)
    residual = mx.random.normal((batch, 2, 4, 16)).astype(mx.bfloat16)
    post, comb = mx.random.normal((batch, 2, 4)), mx.random.normal((batch, 2, 4, 4))
    routed = quantized.exact_quantized_selected_linear(linear, inputs, indices)
    expected = fast_ops.exact_hc_expand(
        SwitchGLU._combine(routed, weights, shared), residual, post, comb
    )

    def fail(*args, **kwargs):
        raise AssertionError("supported formats must use the fused backend")

    monkeypatch.setattr(
        "mlx_vlm.models.quantized_verifier.exact_quantized_selected_linear", fail
    )
    actual = quantized.exact_quantized_moe_hc_expand(
        linear, inputs, indices, weights, shared, residual, post, comb
    )
    equal(actual, expected)


@parametrize("bits", [4, 5, 8])
@parametrize("widths", [(16, 24), (16, 24, 32), (8, 16, 24, 32)])
@parametrize("length", [2, 3, 6, 8])
def test_fused_projection_parity(bits, widths, length):
    mx.random.seed(51 + bits + len(widths) + length)
    linears = [
        bf16_parameters(
            nn.QuantizedLinear(512, width, bias=False, group_size=64, bits=bits)
        )
        for width in widths
    ]
    inputs = mx.random.normal((1, length, 512)).astype(mx.bfloat16)
    expected = [
        verifier_linear._target_verify_timewise(linear, inputs) for linear in linears
    ]
    actual = verifier_linear._target_verify_linears(linears, inputs)
    equal(actual, expected)


@parametrize("kind", ["mtp", "dflash", "eagle3"])
@parametrize("batch", [1, 2])
@parametrize("token,failure", [(4, False), (7, False), (4, True)])
def test_round_commit_close_and_abort(kind, batch, token, failure):
    target = models.TransactionTarget(token)
    caches = [ArraysCache(1), BatchKVCache([0] * batch) if batch > 1 else KVCache()]
    caches[0][0] = mx.zeros((batch, 1))
    initial = mx.zeros((batch, 1, 2, 1))
    caches[1].update_and_fetch(initial, initial)

    def sample(logits):
        if failure:
            raise RuntimeError("injected sampler failure")
        return greedy(logits)

    suffix = "_batch" if batch > 1 else ""
    rounds = getattr(speculative, f"_{kind}_rounds{suffix}")
    if batch > 1:
        assert speculative.get_speculative_rounds_batch(kind) is rounds
    options = dict(
        first_bonus=1 if batch == 1 else mx.ones((batch,), dtype=mx.int32),
        max_tokens=4,
        sampler=sample,
        draft_block_size=2,
    )
    if kind == "mtp":
        options["shared_kv_states"] = {}
    generator = rounds(
        target, models.TransactionDrafter(), caches, mx.zeros((batch, 1, 4)), **options
    )
    error = pytest.raises(RuntimeError, match="injected sampler failure")
    with error if failure else nullcontext():
        next(generator)
    retained = 0 if failure else 2 if token == 4 else 1
    assert not target.transaction.active and not caches[0].is_speculating
    assert caches[0][0].tolist() == [[float(retained)]] * batch
    offset = caches[1].offset
    equal(mx.array(offset).reshape(-1), mx.array([2 + retained] * batch))
    generator.close()


def test_acceptance_walk_and_budgets():
    drafts = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    for retained, budget in product(range(4), (0, 1, 4)):
        targets = mx.concatenate(
            [drafts[:, :retained], mx.full((2, 4 - retained), 9)], 1
        )
        expected = [(row[:retained] + [9])[:budget] for row in drafts.tolist()]
        assert _speculative_walk_batch(drafts, targets, [budget] * 2) == (
            [retained] * 2,
            expected,
        )
    for budgets in ([1], [1, 1, 1], [1, -1]):
        with pytest.raises(ValueError):
            _speculative_walk_batch(mx.ones((2, 1)), mx.ones((2, 2)), budgets)
    drafts = mx.array([[10, 11, 12], [20, 21, 22]])
    targets = mx.array([[10, 99, 98, 97], [20, 21, 77, 76]])
    assert common._speculative_walk_batch_uniform_acceptance(
        drafts, targets, accepted_list=[1, 2], budgets=[4, 4]
    ) == ([1, 1], [[10, 99], [20, 21]])
    assert _speculative_walk_batch(
        mx.zeros((0, 2), dtype=mx.int32), mx.zeros((0, 3), dtype=mx.int32), budgets=[]
    ) == ([], [])


def test_sampler_rng_isolation():
    logits = mx.zeros((1, 8))
    sample = lambda: mx.random.categorical(logits)
    mx.random.seed(123)
    expected = [sample(), sample()]
    mx.eval(*expected)
    mx.random.seed(123)
    drafter = NS(_seed_token=None)
    rng = common._SpeculativeSamplerRNG(drafter, enabled=True)
    actual = [sample()]
    mx.eval(*actual)
    rng.target_sampled()
    rng.draft_call(lambda: setattr(drafter, "_seed_token", sample()))
    actual.append(sample())
    mx.eval(*actual)
    equal(actual, expected)
    assert drafter._seed_token is not None
    sampler = module("server.generation")._PositionedTargetSampler(
        temperature=1.0, top_p=0.95, top_k=20, seed=7
    )
    first = sampler.sample_proposal(logits, row_ids=[0], positions=[3])
    second = sampler.sample_proposal(logits, row_ids=[0], positions=[3])
    assert first.shape == (1,) and mx.array_equal(first, second).item()


@parametrize("family,quant", [("qwen", "mxfp8"), ("glm", "affine"), ("glm", "mxfp8")])
def test_split_and_requantize_checkpoint(tmp_path, family, quant):
    name, factory = models.TEXT[family]
    text = factory()
    text.mtp_num_hidden_layers = 1
    prefix = (
        "mtp.layers.0.mlp.down_proj"
        if family == "qwen"
        else f"model.language_model.layers.{text.num_hidden_layers}.eh_proj"
    )
    key = "layers.0.mlp.down_proj" if family == "qwen" else "eh_proj"
    fp8 = family == "qwen" or quant == "affine"
    config = dict(model_type=name, text_config=text.to_dict())
    weights = {prefix + ".weight": mx.ones((128, 128), dtype=mx.bfloat16)}
    if fp8:
        config["quantization_config"] = dict(
            quant_method="fp8", fmt="e4m3", weight_block_size=[128, 128]
        )
        weights[prefix + ".weight"] = mx.to_fp8(weights[prefix + ".weight"])
        weights[prefix + ".weight_scale_inv"] = mx.full(
            (1, 1), 0.125, dtype=mx.bfloat16
        )
    kwargs = (
        dict(q_bits=4, q_group_size=64) if quant == "affine" else dict(q_mode="mxfp8")
    )
    result, written, _ = split_checkpoint(tmp_path, config, weights, **kwargs)
    expected = (
        dict(group_size=64, bits=4, mode="affine")
        if quant == "affine"
        else dict(group_size=32, bits=8, mode="mxfp8")
    )
    assert result["quantization"] == result["quantization_config"] == expected
    assert written[key + ".weight"].dtype == mx.uint32
    assert written[key + ".scales"].dtype == (
        mx.bfloat16 if quant == "affine" else mx.uint8
    )
    assert not any(key.endswith("weight_scale_inv") for key in written)
    assert (key + ".biases" in written) == (quant == "affine")


@parametrize("family", ["dflash2", "glimmer", "dspark-lfm2", "dspark-qwen"])
def test_checkpoint_routing_and_validation(tmp_path, family):
    settings = models.dflash_config(family)
    (tmp_path / "config.json").write_text(json.dumps(settings))
    assert resolve_drafter_kind(tmp_path) == "dflash"
    drafter = models.dflash_drafter(family)
    target = models.dflash_target(family)
    if family == "glimmer":
        target.language_model.config.model_type = "other"
    else:
        config = target.language_model.config
        getattr(config, "text_config", config).hidden_size += 1
    with pytest.raises(ValueError):
        validate_drafter_compatibility(target, drafter, "dflash")


def split_checkpoint(
    path, config, weights, *, separate=False, indexed=False, **options
):
    (path / "config.json").write_text(json.dumps(config))
    shard = "mtp.safetensors" if separate else "model.safetensors"
    mx.save_safetensors(str(path / shard), weights)
    if separate or indexed:
        index = (
            {"model.foo": "model.safetensors"}
            if separate
            else dict.fromkeys(weights, shard)
        )
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )
    output = path / "draft"
    split_mtp(str(path), str(output), **options)
    config = json.loads((output / "config.json").read_text())
    weights = {
        k: v
        for shard in sorted(output.glob("*.safetensors"))
        for k, v in mx.load(str(shard)).items()
    }
    return config, weights, output


def test_deepseek_dspark_split_load_and_draft(tmp_path):
    arch = module("speculative.drafters.deepseek_v4_dspark")
    text = models.tiny_deepseek_config()
    cfg = arch.ModelConfig(
        text_config=text, **models.DATA["speculative"]["deepseek_dspark"]
    )
    source = models.dspark_source(arch.Model(cfg), cfg)
    source["mtp.2.confidence_head.proj.weight"] = mx.zeros((1, 24))
    source["mtp.0.ffn.gate.bias_vl"] = mx.zeros((2,))
    settings = {**text.to_dict(), "model_type": "deepseek_v4"}
    settings.update(
        dspark_block_size=4,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=8,
    )
    config, weights, output = split_checkpoint(
        tmp_path, settings, source, separate=True
    )
    assert config["model_type"] == "deepseek_v4_dspark"
    assert config["n_mtp_layers"] == 3 and config["target_layer_ids"] == [0, 1, 2]
    assert len(list(output.glob("model-*.safetensors"))) == 3
    assert (output / "model.safetensors.index.json").exists()
    fresh = arch.Model(arch.ModelConfig.from_dict(config))
    weights = fresh.sanitize(weights)
    fresh.load_weights(list(weights.items()), strict=True)
    assert not any("confidence_head" in key or "bias_vl" in key for key in weights)
    target = NS(
        embed_tokens=nn.Embedding(32, 16), lm_head=nn.Linear(16, 32, bias=False)
    )
    cache = fresh.reset(target)
    tokens = fresh.draft_block(3, mx.zeros((1, 5, 48)), cache, 5, greedy)
    mx.eval(tokens)
    assert tokens.shape == (1, 4)
    assert all(0 <= token < 32 for token in tokens[0].tolist())


@parametrize("accepted", [0, 2])
def test_eagle3_draft_replay(accepted):
    arch = module("speculative.drafters.eagle3")
    cfg = arch.ModelConfig(
        transformer_layer_config=models.dimensions(),
        **models.DATA["speculative"]["eagle3"],
    )
    drafter = arch.Model(cfg)
    drafter.d2t = mx.arange(16, dtype=mx.int32)
    drafter.reset(NS(embed_tokens=nn.Embedding(32, 16)))
    prompt, hidden = mx.array([[1, 2, 3]]), mx.zeros((1, 4, 48))
    drafter.prefill_from_target_hidden(prompt, hidden[:, :3], 4, greedy, greedy=True)
    before = drafter._next_position
    tokens = drafter.draft_block(4, hidden[:, -1:], None, 4, greedy, greedy=True)
    assert tokens.shape == (1, 3)
    kept = tokens[0, :accepted].tolist() + [7]
    drafter.accept_verified_tokens(
        hidden, tokens, accepted, kept, greedy, mx.int32, True
    )
    mx.eval(
        drafter._seed_token, drafter._seed_hidden, [c.state for c in drafter._cache]
    )
    assert drafter._round_appended == 0
    assert drafter._next_position == before + len(kept)
    assert drafter._seed_token.shape == (1, 1)
    assert drafter._draft_to_target(mx.array([[0, 1, 3]]), mx.int32).tolist() == [
        [0, 2, 6]
    ]


@parametrize("offsets,length", [([128], 8), ([10], 6), ([5, 8], 8), ([128, 10], 8)])
@parametrize("query_len", [1, 3])
def test_drafter_masks(offsets, length, query_len):
    masks = module("speculative.drafters.gemma4_assistant.masks")
    kv = (mx.zeros((len(offsets), 1, length, 4)),) * 2
    result = masks.make_drafter_masks(
        {kind: kv for kind in ("full_attention", "sliding_attention")},
        query_len=query_len,
        query_offset=offsets[0] if len(offsets) == 1 else mx.array(offsets),
        sliding_window=4,
        kv_valid_len=mx.array(offsets) if len(offsets) == 1 else None,
    )
    for kind, mask in result.items():
        rows = query_len if kind == "sliding_attention" else 1
        expected = np.full((len(offsets), 1, rows, length), -np.inf)
        for row, offset in enumerate(offsets):
            valid = min(offset, length)
            for query in range(rows):
                start = max(0, valid + query - 3) if kind == "sliding_attention" else 0
                expected[row, 0, query, start:valid] = 0
        assert mask.shape == expected.shape
        equal(mask, mx.array(expected))


def test_laguna_checkpoint_contract():
    arch = module("speculative.drafters.laguna_dflash.config")
    settings = models.values("laguna_dflash")
    config = arch.DFlashConfig.from_dict(settings)
    expected = arch.expected_laguna_dflash_weight_shapes(config)
    weights = {key: NS(shape=shape) for key, shape in expected.items()}
    arch.validate_laguna_dflash_weights(weights, config)
    arch.validate_laguna_dflash_target(
        config, target_model_config=NS(num_hidden_layers=2, vocab_size=32)
    )
    for field in ("hidden_size", "layer_types", "dflash_config"):
        malformed = dict(settings)
        del malformed[field]
        with pytest.raises(ValueError):
            arch.DFlashConfig.from_dict(malformed)
    weights["unexpected.weight"] = NS(shape=(1,))
    with pytest.raises(ValueError, match="weight keys"):
        arch.validate_laguna_dflash_weights(weights, config)


@parametrize("padding", [[5, 0], [5, 5]])
def test_padded_prefill_chunks(padding):
    lm, cfg = models.language("qwen")
    recurrent = ArraysCache(2)
    recurrent.left_padding = mx.array(padding)
    caches = [recurrent, BatchKVCache(padding)]
    for step in range(2):
        ids = [[0, 0, 0 if step == 0 else 4], [1, 2, 3]]
        if padding[1] == 5:
            ids[1] = ids[0]
        hidden = lm.model(mx.array(ids), cache=caches)
        mx.eval(hidden, caches[1].state)
        assert hidden.shape == (2, 3, cfg.hidden_size)
        assert mx.isfinite(hidden).all().item()
        assert caches[1].offset.tolist() == [3 * (step + 1) - p for p in padding]
        expected_padding = (
            padding if padding[1] == 0 else [max(5 - 3 * (step + 1), 0)] * 2
        )
        assert caches[1].left_padding.tolist() == expected_padding


@parametrize(
    "family,accepted", [("qwen", [1, 0]), ("qwen", [1, 1]), ("deepseek", [0, 0])]
)
def test_batched_drafter_commit_and_filter(family, accepted):
    cfg = models.TEXT[family][1]()
    drafter = models.mtp_drafter(family, cfg)
    target = NS(model=NS(embed_tokens=nn.Embedding(cfg.vocab_size, cfg.hidden_size)))
    qwen = family == "qwen"
    drafter.reset(
        NS(language_model=target), **({"left_padding": [0, 0]} if qwen else {})
    )
    position = mx.array([4, 4]) if qwen else 3
    valid = mx.full((2,), 4) if qwen else 4
    drafter.set_shared_kv({}, kv_offset=4, position=position, kv_valid_len=valid)
    shape = (cfg.hc_mult, 16) if family == "deepseek" else (16,)
    tokens = drafter.draft_block(
        mx.array([7, 8]),
        mx.zeros((2, 1, *shape)),
        None,
        3,
        greedy,
        mx.int32,
        greedy=True,
    )
    kept = [tokens[i, :n].tolist() + [5 + i] for i, n in enumerate(accepted)]
    drafter.accept_verified_tokens_batch(
        mx.zeros((2, 3, *shape)),
        tokens,
        accepted=accepted,
        new_tokens=kept,
        sampler=greedy,
        token_dtype=mx.int32,
        greedy=True,
    )
    mx.eval(drafter._seed_token, drafter._cache[0].state)
    assert drafter._round_appended == 0
    assert drafter._seed_token.shape == (2, 1)
    assert drafter._seed_hidden.shape == (2, 1, *shape)
    if qwen:
        assert drafter._cache[0].offset.tolist() == [n + 1 for n in accepted]
        assert drafter._cache[0].left_padding.tolist() == (
            [1, 2] if accepted == [1, 0] else [0, 0]
        )
        drafter.filter_batch(mx.array([1]))
        assert drafter._cache[0].keys.shape[0] == 1
        assert drafter._cache[0].left_padding.tolist() == [0]
        assert drafter._cache[0].offset.tolist() == [1 + accepted[1]]
        assert drafter._next_position.tolist() == [5 + accepted[1]]
        assert drafter._seed_token.shape == (1, 1)
    else:
        assert drafter._cache[0].offset == 1
        assert drafter._next_position == 5


@parametrize("layout", ["full_attention", "sliding_attention"])
@parametrize("nested", [False, True])
def test_gemma_dspark_contract_and_attention(layout, nested):
    arch = module("speculative.drafters.gemma4_dspark.gemma4_dspark")
    settings = models.values(
        "gemma_dspark",
        layer_types=[layout],
        rope_parameters={layout: dict(rope_theta=10000.0, rope_type="default")},
    )
    draft = deepcopy(models.DATA["speculative"]["gemma_dspark_draft"])
    settings.update({"dflash_config": draft} if nested else draft)
    cfg = arch.ModelConfig.from_dict(settings)
    assert (cfg.model_type, cfg.backbone_model_type) == ("gemma4_dspark", "gemma4_text")
    assert (cfg.proposal_length, cfg.block_size, cfg.target_layer_ids) == (3, 4, [0])
    drafter = arch.Model(cfg)
    layer = drafter.layers[0]
    result = layer(mx.zeros((1, 2, 16)), mx.zeros((1, 3, 16)), drafter.rope, KVCache())
    mx.eval(result)
    assert result.shape == (1, 2, 16) and mx.isfinite(result).all().item()
    assert hasattr(layer.self_attn, "v_proj") == (layout == "sliding_attention")
    assert cfg.final_logit_softcapping == 30.0
    for key, value in [
        ("mask_token_id", 32),
        ("target_layer_ids", [2]),
        ("markov_rank", 0),
    ]:
        malformed = deepcopy(settings)
        malformed.get("dflash_config", malformed)[key] = value
        with pytest.raises(ValueError):
            arch.ModelConfig.from_dict(malformed)


def sampling_spies(positioned):
    logits = Mock(side_effect=lambda hidden: hidden)
    sample = Mock(side_effect=lambda logits, **kw: greedy(logits))
    target = NS(speculative_logits_from_hidden=logits)
    return target, NS(sample_target=sample) if positioned else greedy, logits, sample


@parametrize("uniform", [False, True])
@parametrize("positioned", [False, True])
@parametrize("reject_first", [False, True])
def test_deferred_acceptance(uniform, positioned, reject_first):
    target, sampler, head_calls, sample_calls = sampling_spies(positioned)
    rows = [[2, 1, 3], [1, 2, 1] if reject_first else [0, 2, 1]]
    hidden = mx.eye(4)[mx.array(rows)] * 9
    walk = (
        mtp._speculative_walk_batch_deferred_uniform
        if uniform
        else mtp._speculative_walk_batch_deferred_greedy
    )
    accepted, tokens = walk(
        target,
        hidden,
        mx.array([[2, 3], [0, 2]]),
        sampler,
        budgets=[3, 2],
        row_ids=[10, 11],
        base_positions=[7, 12],
    )
    expected = (
        [0, 0]
        if reject_first and uniform
        else [1, 0] if reject_first else [1, 1] if uniform else [1, 2]
    )
    assert accepted == expected
    assert tokens == (
        [[2], [1]]
        if reject_first and uniform
        else [[2, 1], [1]] if reject_first else [[2, 1], [0, 2]]
    )
    calls = 1 if reject_first and uniform else 2 if uniform or reject_first else 3
    assert head_calls.call_count == calls
    if positioned:
        assert [
            (c.kwargs["row_ids"], c.kwargs["positions"])
            for c in sample_calls.call_args_list
        ] == [([10, 11], [7 + i, 12 + i]) for i in range(calls)]


@parametrize("family", ["glm4_moe_lite", "deepseek_v4"])
def test_native_checkpoint_layouts(tmp_path, family):
    deepseek = family == "deepseek_v4"
    cfg, weights = models.native_speculative_checkpoint(family)
    config, out, _ = split_checkpoint(
        tmp_path, cfg, weights, separate=deepseek, indexed=True
    )
    assert config["model_type"] == family + "_mtp" and config["block_size"] == 2
    if deepseek:
        for key in ("e_proj", "decoder.attn.wq_a"):
            assert config["quantization"][key]["mode"] == "mxfp8"
            assert out[key + ".weight"].dtype == mx.uint32
            assert key + ".scales" in out
        assert "enorm.weight" in out
        assert (
            out["decoder.ffn.switch_mlp.gate_proj.weight"].shape[0]
            == cfg["n_routed_experts"]
        )
        assert {
            "decoder.ffn.gate.e_score_correction_bias",
            "decoder.attn_hc.fn",
            "hc_head.scale",
        } <= out.keys()
    else:
        prefix = "model.mtp_block."
        assert "model.embed_tokens.weight" in out and "lm_head.weight" in out
        for key, shape in [
            ("self_attn.embed_q", (2, 4, 4)),
            ("self_attn.unembed_out", (2, 6, 4)),
            ("mlp.switch_mlp.gate_proj", (2, 4, 8)),
        ]:
            assert out[prefix + key + ".weight"].shape == shape
        assert out[prefix + "mlp.gate.e_score_correction_bias"].dtype == mx.float32
        assert not any(
            ".experts.0." in key or "rotary_emb" in key or "kv_b_proj" in key
            for key in out
        )


@parametrize("batch", [1, 2])
@parametrize("step", [1, 4])
def test_rotating_cache_commit_abort(batch, step):
    arch = module("models.cache")
    cache = (
        arch.BatchRotatingKVCache(8, [0] * batch)
        if batch > 1
        else arch.RotatingKVCache(8)
    )
    prefix = mx.broadcast_to(mx.arange(13)[None, None, :, None], (batch, 1, 13, 1))
    cache.update_and_fetch(prefix, prefix)
    reference = deepcopy(cache)
    for i, retained in enumerate((3, 1, 4, 0, None, 2)):
        incoming = mx.broadcast_to(
            (mx.arange(4) + 100 + i * 10)[None, None, :, None], (batch, 1, 4, 1)
        )
        transaction = start_speculative_cache([cache], 4)
        for start in range(0, 4, step):
            part = incoming[:, :, start : start + step]
            cache.update_and_fetch(part, part)
        if retained is None:
            transaction.abort()
        else:
            transaction.commit([retained] * batch)
            for start in range(0, retained, step):
                part = incoming[:, :, start : min(start + step, retained)]
                reference.update_and_fetch(part, part)
        assert "update_and_fetch" not in cache.__dict__
        equal(cache.state, reference.state)
        assert cache.meta_state == reference.meta_state


@parametrize("family", ["qwen3_5", "gemma4", "deepseek_v4"])
@parametrize("accepted", [[2, 2], [0, 2]])
def test_quantized_cache_rollback(family, accepted):
    cache = module("turboquant").BatchTurboQuantKVCache([0, 0], bits=3.5)
    keys = mx.arange(112, dtype=mx.float32).reshape(2, 1, 7, 8)
    cache.update_and_fetch(keys, keys + 100)
    generic = family == "deepseek_v4"
    rollback = (
        module("speculative.cache_state").rollback_speculative_cache
        if generic
        else module(
            f"models.{family}.language"
        ).LanguageModel.rollback_speculative_cache
    )
    args = ([cache], None) if generic else (None, [cache], None)
    if accepted[0] != accepted[1]:
        with pytest.raises(RuntimeError, match="uniform"):
            rollback(*args, mx.array(accepted), block_size=5)
    else:
        assert rollback(*args, mx.array(accepted), block_size=5) == 2
        assert cache._idx == 5


def test_shared_kv_padding():
    masks = module("speculative.drafters.gemma4_assistant.masks")
    keys = mx.array([0, 0, 0, 10, 11, 12, 13, 14, *range(20, 28)]).reshape(2, 1, 8, 1)
    states = masks.normalize_batched_shared_kv_states(
        {"full_attention": (keys, keys + 100)},
        kv_valid_len=mx.array([5, 8]),
        left_padding=mx.array([3, 0]),
    )
    for offset, tensor in zip((0, 100), states["full_attention"]):
        assert tensor[:, 0, :, 0].tolist() == [
            list(range(10 + offset, 15 + offset)) + [0] * 3,
            list(range(20 + offset, 28 + offset)),
        ]


def test_sparse_logits():
    arch = module("speculative.drafters.gemma4_assistant.masked_embedder")
    embedder = arch.MaskedEmbedder(
        NS(
            text_config=NS(hidden_size=2, vocab_size=8),
            num_centroids=2,
            centroid_intermediate_top_k=1,
        )
    )
    embedder.centroids.weight = mx.eye(2)
    embedder.token_ordering = mx.array([0, 2, 4, 6, 1, 3, 5, 7])
    weights = (mx.eye(2)[None] * mx.arange(1, 5)[:, None, None]).reshape(8, 2)
    hidden = mx.array([[[2.0, 0.5], [0.5, 2.0]]])
    assert (
        embedder.argmax(hidden, weights).tolist()
        == greedy(embedder(hidden, weights)).tolist()
        == [[6, 7]]
    )
    embedder.unfreeze()
    assert "token_ordering" not in dict(tree_flatten(embedder.trainable_parameters()))


def test_eagle_hot_verifier_includes_eos():
    class Backbone:
        embed_tokens = NS(
            weight=mx.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [3.5, -0.5]]
            )
        )

        def __call__(self, inputs, cache, capture_layer_ids, hidden_sink):
            hidden = mx.stack([inputs, inputs + 1], axis=-1).astype(mx.float32)
            hidden_sink.append(hidden)
            return hidden

    class Target:
        __module__ = "mlx_vlm.models.gemma4.language"
        config, model, final_logit_softcapping = NS(eos_token_id=4), Backbone(), None

        def logits_from_hidden(self, hidden):
            return self.model.embed_tokens.weight[None, : hidden.shape[1], :]

    result = module("speculative.eagle3")._eagle3_verify_target_hot(
        Target(),
        NS(d2t=mx.array([1, 1])),
        mx.array([[0, 1]]),
        prompt_cache=[],
        sampler=lambda logits: mx.array([[4]]),
        target_layer_ids=[1],
    )
    hidden, tokens, transaction = result
    assert hidden.tolist() == [[[0.0, 1.0], [1.0, 2.0]]]
    assert tokens.tolist() == [[1, 4]] and transaction is None


@parametrize("family", ["eagle3", "dflash"])
def test_adaptive_block_policy(family):
    if family == "eagle3":
        arch = module("speculative.eagle3")
        cfg = module("speculative.drafters.eagle3").ModelConfig(block_size=5)
        drafter = NS(config=cfg, accept_lens=[], draft_lens=[])
        assert arch._eagle3_block_settings(drafter, None) == (5, 5, False)
        cfg.adaptive_max_block_size = 12
        assert arch._eagle3_block_settings(drafter, None) == (12, 5, True)
        cases = [
            ([], [], 5, 5),
            ([0, 0, 1, 1, 0, 1], [4] * 6, 5, 12),
            ([4, 0, 4, 0, 4, 0, 4], [4] * 7, 5, 8),
            ([0, 0, 0, 1, 1, 1], [11] * 6, 12, 8),
        ]
        choose = lambda: arch._eagle3_next_block_size(
            drafter, 12, 5, 128, adaptive=True
        )
    else:
        drafter = NS()
        cases = [
            ([], [], 16, 16),
            ([1, 2], [15, 7], 16, 4),
            ([3, 2, 1, 3], [3] * 4, 4, 4),
        ]
        choose = lambda: module("speculative.utils")._dflash_next_block_size(
            drafter, 16, 20
        )
    for accepted, drafted, previous, expected in cases:
        drafter.accept_lens, drafter.draft_lens = accepted, drafted
        drafter._adaptive_block_size = previous
        assert choose() == expected


@parametrize("batch", [1, 4])
@parametrize("length", [3, 7, 8])
@parametrize("bits", [4, 5])
def test_wide_quantized_verifier(batch, length, bits):
    mx.random.seed(43)
    linear = bf16_parameters(
        nn.QuantizedLinear(6144, 32, bias=False, group_size=64, bits=bits)
    )
    inputs = mx.random.normal((batch, length, 6144)).astype(mx.bfloat16)
    # This verifier reproduces native GEMV separately for every token and row.
    expected = verifier_linear._target_verify_singletons(linear, inputs)
    actual = verifier_linear._target_verify_linear(linear, inputs)
    equal(actual, expected)
    equal(
        verifier_linear._target_verify_quantized_argmax(linear, inputs),
        greedy(expected),
    )


def test_glm_mtp_native_weight_fusion():
    cfg = models.tiny_glm_text_config()
    arch = module("speculative.drafters.glm5_next_mtp")
    weights = models.glm_mtp_checkpoint_weights(cfg)
    out = arch.Model.sanitize(NS(args=cfg), weights.copy())
    expected = deepcopy(models.DATA["speculative"]["glm_fused_shapes"])
    for key, shape in expected.items():
        assert out[f"mtp_block.{key}.weight"].shape == tuple(shape)
    equal(
        out["mtp_block.mlp.switch_mlp.gate_proj.weight"][1],
        weights["mtp_block.mlp.experts.1.gate_proj.weight"],
    )


@parametrize("positioned", [False, True])
def test_single_mtp_sampling_stops_at_rejection(positioned):
    target, sampler, head_calls, sample_calls = sampling_spies(positioned)
    hidden = mx.eye(4)[mx.array([[2, 3, 1, 0]])] * 9
    verify = mtp._MTPVerifyResult(hidden=hidden, shared_kv_states={})
    accepted, tokens = mtp._mtp_acceptance_walk(
        target,
        verify,
        mx.array([[2, 1, 3]]),
        sampler,
        budget=4,
        row_id=5,
        base_position=7 if positioned else None,
    )
    assert (accepted, tokens) == (1, [2, 3])
    assert head_calls.call_count == (1 if positioned else 2)
    assert [
        (c.kwargs["row_ids"], c.kwargs["positions"])
        for c in sample_calls.call_args_list
    ] == ([([5] * 4, [7, 8, 9, 10])] if positioned else [])


@parametrize(
    "batch,uniform,greedy_mode,positioned,expected",
    [
        (1, False, False, False, False),
        (2, False, True, False, False),
        (2, False, False, False, True),
        (2, False, False, True, False),
        (2, True, True, True, True),
    ],
)
def test_batch_sampling_policy(batch, uniform, greedy_mode, positioned, expected):
    sampler = NS(sample_target=greedy) if positioned else greedy
    assert (
        module("speculative.mtp")._mtp_use_uniform_deferred_walk(
            NS(requires_uniform_batch_acceptance=uniform),
            n_active=batch,
            greedy_sampling=greedy_mode,
            sampler=sampler,
        )
        is expected
    )


def test_mixed_positions_keep_shared_kv_and_round_aligned():
    calls, rounds = [], []
    keys = mx.array([10.0, 20.0]).reshape(2, 1, 1, 1)
    shared = {"full_attention": (keys, mx.zeros_like(keys))}
    drafter = NS(_shared_kv=shared, _draft_round=4)

    def set_shared(states, kv_offset, position=None, kv_valid_len=None, **kwargs):
        drafter._shared_kv = states
        calls.append((kv_offset, position, kv_valid_len))

    def draft(bonus, hidden, cache, block_size, sampler, dtype):
        rounds.append(drafter._draft_round)
        drafter._draft_round += 1
        base = drafter._shared_kv["full_attention"][0].item()
        return mx.full((1, block_size - 1), base + bonus, dtype=dtype)

    drafter.set_shared_kv, drafter.draft_block = set_shared, draft
    tokens = module("speculative.mtp")._mtp_draft_block_active(
        drafter, [3, 7], mx.zeros((2, 1, 1)), 2, greedy, mx.int32, positions=[11, 12]
    )
    assert tokens.tolist() == [[13], [27]]
    assert rounds == [4, 4] and drafter._draft_round == 5
    assert calls[:2] == [(11, 10, 11), (12, 11, 12)]
    assert calls[2][0] == 12 and calls[2][1].tolist() == [10, 11]
    assert calls[2][2].tolist() == [11, 12] and drafter._shared_kv is shared


def test_shared_kv_metadata_and_rotating_prompt_cache():
    mtp, cache_module = module("speculative.mtp"), module("models.cache")
    keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    cache = NS(state=(keys, keys + 1, "metadata"))
    target = NS(model=NS(layers=[NS(layer_type="full_attention")]))
    shared = mtp._mtp_shared_kv_from_prompt_cache(target, [cache])
    assert shared["full_attention"][0] is keys
    equal(shared["full_attention"][1], keys + 1)
    rotating = cache_module.RotatingKVCache(4, keep=0)
    rotating.update_and_fetch(keys, keys)
    caches = [cache_module.CacheList(rotating, cache_module.PoolingCache(4))]
    mtp._buffer_mtp_target_cache(caches, NS(config=NS(block_size=3)), None)
    assert isinstance(caches[0][0], cache_module.BufferedRotatingKVCache)
    assert isinstance(caches[0][1], cache_module.PoolingCache)
    equal(caches[0][0].state[0], keys)


def test_rotating_ragged_commit_masks_rejected_tokens():
    cache = module("models.cache").BatchRotatingKVCache(8, [0, 0])
    prefix = mx.broadcast_to(mx.arange(12)[None, None, :, None], (2, 1, 12, 1))
    cache.update_and_fetch(prefix, prefix)
    transaction = start_speculative_cache([cache], 4)
    incoming = mx.broadcast_to(mx.arange(100, 104)[None, None, :, None], (2, 1, 4, 1))
    cache.update_and_fetch(incoming, incoming)
    transaction.commit([3, 1])
    assert cache.offset.tolist() == [15, 13]
    mask = cache.make_mask(1)
    next_kv = mx.full((2, 1, 1, 1), 200)
    keys, _ = cache.update_and_fetch(next_kv, next_kv)
    for row, retained in enumerate((3, 1)):
        visible = mx.where(mask[row, 0, 0], keys[row, 0, :, 0], -1)
        expected = list(range(12)) + list(range(100, 100 + retained)) + [200]
        assert sorted(x for x in visible.tolist() if x >= 0) == sorted(expected[-8:])


def test_temporal_cache_boundaries_and_failed_transactions():
    cache = ArraysCache(1)
    initial = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    cache[0] = initial
    transaction = start_speculative_cache([cache], 3)
    cache[0] = initial + 3
    cache.record_speculative_states(
        0, mx.stack([initial + 1, initial + 2], 1), cache[0]
    )
    transaction.commit([0, 3])
    assert cache[0].tolist() == [initial[0].tolist(), (initial[1] + 3).tolist()]
    stale = start_speculative_cache([cache], 2)
    current = start_speculative_cache([cache], 2)
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit([1, 1])
    current.abort()
    before = cache[0]
    kv = KVCache()
    keys = mx.zeros((2, 1, 2, 1))
    kv.update_and_fetch(keys, keys)
    with pytest.raises(RuntimeError, match="without temporal records"):
        with start_speculative_cache([cache, kv], 2) as transaction:
            cache[0] = before + 1
            cache._qwen3_5_lengths_info = (mx.array([3, 3]), 3)
            kv.update_and_fetch(keys, keys)
            transaction.commit([1, 1])
    assert cache[0] is before and not cache.is_speculating and kv.offset == 2
    module("models.qwen3_5.language")._qwen3_5_lengths_info(cache)
    assert not hasattr(cache, "_qwen3_5_lengths_info")


@parametrize("batch", [1, 2])
def test_glm_rejection_restores_draft_pool(batch):
    mx.random.seed(2127)
    model, cfg = models.language("glm")
    drafter = models.mtp_drafter("glm", cfg)
    drafter.eval()
    drafter.reset(model, left_padding=[0] * batch if batch > 1 else None)
    hidden = mx.random.normal((batch, 1, cfg.hidden_size))
    mx.eval(drafter._forward_tokens(mx.array([[1]] * batch), hidden, mx.int32))
    reference = deepcopy(drafter)
    tokens = drafter.draft_block(
        2 if batch == 1 else mx.full((batch,), 2), hidden, None, 4, greedy, greedy=True
    )
    mx.eval(tokens)
    assert drafter._round_appended == 3
    transaction = drafter._round_transaction
    verified = mx.random.normal((batch, 4, cfg.hidden_size))
    for candidate in (drafter, reference):
        candidate.accept_verified_tokens_batch(
            verified, tokens, [0] * batch, [[8]] * batch, greedy, greedy=True
        )
        mx.eval(candidate.draft_eval_state())
    assert not transaction.active
    actual, expected = drafter._cache[0][2], reference._cache[0][2]
    equal(mx.array(actual.offset), mx.array(expected.offset))
    assert actual.remainder == expected.remainder
    equal(actual.state, expected.state)
    equal(drafter._seed_token, reference._seed_token)
    equal(drafter._seed_hidden, reference._seed_hidden)


@parametrize("ragged", [False, True])
def test_minimax_index_cache_rollback(ragged):
    arch = module("models.minimax_m3_vl.language")
    cache = arch.MiniMaxM3BatchKVCache([0, 0]) if ragged else arch.MiniMaxM3KVCache()
    keys = mx.ones((2 if ragged else 1, 1, 5, 4))
    cache.update_and_fetch(keys, keys)
    cache.update_index_and_fetch(keys)
    rollback = arch.LanguageModel.rollback_speculative_cache
    if ragged:
        with pytest.raises(RuntimeError, match="uniform"):
            rollback(None, [cache], None, mx.array([2, 0]), block_size=4)
    else:
        assert rollback(None, [cache], None, 1, block_size=4) == 1
        assert cache.offset == cache.index_offset == 3


@parametrize("batch", [1, 2])
def test_chunked_prefill_retains_all_drafter_features(batch):
    model, _ = models.language("deepseek")
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7]] * batch)
    drafter = NS(config=NS(target_layer_ids=[0]))
    cache = _make_cache(model, [0] * batch)
    expected = mx.concatenate(
        [
            model(
                tokens[:, start : start + 2], cache=cache, capture_layer_ids=[0]
            ).hidden_states[0]
            for start in range(0, 7, 2)
        ],
        axis=1,
    )
    processing = module("generate.ar").PromptProcessingBatch(
        model=model,
        uids=list(range(batch)),
        input_ids=tokens.tolist(),
        max_tokens=[8] * batch,
        inputs_embeds=model.model.embed_tokens(tokens),
        prompt_kwargs={},
        prefill_step_size=2,
        draft_model=drafter,
        draft_kind="dflash",
    )
    while processing.needs_processing():
        assert processing.prompt_step() <= 2
    generated = processing.generate(greedy, lambda *args: False, compute_logprobs=False)
    assert generated.hidden.shape[1] == tokens.shape[1]
    equal(generated.hidden, expected)


def test_argmax_fallback_does_not_append_twice():
    calls, caches = [], [ArraysCache(1)]
    caches[0][0] = mx.zeros((1, 1))

    def forward(inputs, cache):
        calls.append(inputs)
        return mx.ones((1, 2, 4)), {}, start_speculative_cache(cache, inputs.shape[1])

    target = NS(
        speculative_verify_hidden=forward,
        speculative_argmax_from_hidden=lambda h: None,
        speculative_logits_from_hidden=lambda h: h,
    )
    result = _mtp_verify_target(target, mx.array([[1, 2]]), caches, greedy)
    assert len(calls) == 1 and result.target_tokens.tolist() == [[0, 0]]
    result.abort()
    assert not caches[0].is_speculating


def test_chunked_verification_failure_restores_initial_cache():
    cache, calls = ArraysCache(1), []
    initial = mx.zeros((1, 1), dtype=mx.int32)
    cache[0] = initial

    def forward(tokens, cache):
        calls.append(tokens.shape[1])
        cache[0].update_window(0, mx.concatenate([cache[0][0], tokens], 1), 1)
        if len(calls) == 2:
            raise RuntimeError("second forward failed")
        return LanguageModelOutput(
            logits=tokens[..., None], hidden_states=[tokens[..., None]]
        )

    with pytest.raises(RuntimeError, match="second forward failed"):
        module("speculative.common").verify_forward(
            forward, mx.ones((1, 13), dtype=mx.int32), [cache]
        )
    assert calls == [8, 5] and cache[0] is initial and not cache.is_speculating


def test_qwen_quantized_cache_ragged_rollback():
    model, cfg = models.language(
        "qwen", hidden_size=64, intermediate_size=128, head_dim=32
    )
    recurrent = ArraysCache(2)
    recurrent.left_padding = mx.array([0, 0])
    kv = module("models.cache").BatchQuantizedKVCache([0, 0], group_size=32, bits=4)
    caches = [recurrent, kv]
    mx.eval(model(mx.array([[1, 2, 3], [4, 5, 6]]), cache=caches).logits)
    hidden, shared, transaction = model.speculative_verify_hidden(
        mx.array([[7, 8, 9], [10, 11, 12]]), caches
    )
    mx.eval(hidden, kv.state)
    assert hidden.shape == (2, 3, cfg.hidden_size) and shared == {}
    assert kv._idx == 6 and kv.offset.tolist() == [6, 6]
    assert (
        model.rollback_speculative_cache(
            caches, transaction, mx.array([0, 1]), block_size=3
        )
        == 1
    )
    assert kv._idx == 5 and kv.offset.tolist() == [4, 5]
    assert kv.left_padding.tolist() == [1, 0]


def test_lfm_ragged_rollback_matches_committed_prefixes():
    mx.random.seed(5)
    model = models.dflash_target("dspark-lfm2").language_model
    prompt, verify = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mx.array(
        [[9, 10, 11, 12], [13, 14, 15, 16]]
    )
    caches = [
        BatchKVCache([0, 0]) if layer.is_attention_layer else ArraysCache(1)
        for layer in model.model.layers
    ]
    model(prompt, cache=caches)
    out = model(verify, cache=caches, speculative_verify=True)
    model.rollback_speculative_cache(
        caches, out.gdn_states, mx.array([0, 2]), block_size=4
    )
    assert caches[1].offset.tolist() == [5, 7]
    probes, expected = mx.array([[17], [18]]), []
    for row, retained in enumerate((1, 3)):
        reference = model.make_cache()
        model(
            mx.concatenate([prompt[row], verify[row, :retained]])[None], cache=reference
        )
        expected.append(model(probes[row : row + 1], cache=reference).logits)
    equal(model(probes, cache=caches).logits, mx.concatenate(expected), atol=1e-4)


@parametrize("family", ["glm", "qwen"])
@parametrize("step", [1, 2, 5])
def test_temporal_layers_commit_each_row_prefix(family, step):
    mx.random.seed(2127)
    cfg = models.TEXT[family][1]()
    cfg.linear_head_dim = cfg.linear_key_head_dim = cfg.linear_value_head_dim = 32
    arch = module(f"models.{models.TEXT[family][0]}.language")
    layer = (
        arch.Glm5NextLinearAttention(cfg)
        if family == "glm"
        else arch.Qwen3_5GatedDeltaNet(cfg)
    )
    layer.eval()
    cache = ArraysCache(2, left_padding=[0] * 3)
    cache.prepare(lengths=[100] * 3)
    mx.eval(layer(mx.random.normal((3, 4, cfg.hidden_size)), cache=cache), cache.state)
    for retained in ([0, 2, 5], [3, 1, 4]):
        inputs = mx.random.normal((3, 5, cfg.hidden_size))
        reference, lengths = deepcopy(cache), cache.lengths
        states = [list(reference.state)]
        for i in range(5):
            mx.eval(layer(inputs[:, i : i + 1], cache=reference))
            states.append(list(reference.state))
        transaction = start_speculative_cache([cache], 5)
        for i in range(0, 5, step):
            mx.eval(layer(inputs[:, i : i + step], cache=cache))
        transaction.commit(retained)
        for slot in range(2):
            expected = mx.concatenate(
                [states[n][slot][row : row + 1] for row, n in enumerate(retained)]
            )
            equal(cache[slot], expected, rtol=0, atol=1e-6)
        equal(cache.lengths, lengths - mx.array(retained))
        assert cache.history_capacity == 0 and cache.nbytes == sum(
            v.nbytes for v in cache.state
        )


@parametrize("accepted", [1, 3, [0, 2]])
def test_laguna_rollback_and_feature_capture(accepted):
    arch = module("models.laguna.language")
    cfg = module("models.laguna.config").ModelConfig(**models.values("laguna"))
    model = arch.LanguageModel(cfg)
    caches = model.make_cache()
    out = model(
        mx.array([[1, 2, 3, 4]]),
        cache=caches,
        capture_layer_ids=[0, 1],
        speculative_verify=True,
    )
    mx.eval(out.logits, out.hidden_states)
    assert [h.shape for h in out.hidden_states] == [(1, 4, 16)] * 2
    assert [c.offset for c in caches] == [4, 4]
    if isinstance(accepted, list):
        with pytest.raises(RuntimeError, match="uniform"):
            model.rollback_speculative_cache(caches, None, accepted, block_size=4)
        assert [c.offset for c in caches] == [4, 4]
    else:
        model.rollback_speculative_cache(caches, None, accepted, block_size=4)
        assert [c.offset for c in caches] == [accepted + 1] * 2


@parametrize("from_anchor", [False, True])
def test_dspark_samples_correct_proposal_positions(from_anchor):
    drafter = models.dflash_drafter("dspark-qwen")
    cfg = drafter.config
    cfg.sample_from_anchor = from_anchor
    drafter._hidden = Mock(
        side_effect=lambda ids, h, cache: mx.zeros((1, ids.shape[1], cfg.hidden_size))
    )
    drafter._logits = Mock(
        side_effect=lambda h: mx.zeros((1, h.shape[1], cfg.vocab_size))
    )
    proposals = drafter.draft_block(
        3, mx.zeros((1, 1, cfg.hidden_size)), [], cfg.block_size, greedy
    )
    assert drafter._hidden.call_args.args[0].tolist() == [
        [3] + [31] * (cfg.proposal_length - int(from_anchor))
    ]
    assert drafter._logits.call_args.args[0].shape[1] == cfg.proposal_length
    assert proposals.shape == (1, cfg.proposal_length)


def test_local_mask_tracks_cache_width():
    align = module("models.deepseek_v4.language")._align_local_mask
    trimmed = align(mx.array([[[[False, True, True, False, True]]]]), 3)
    assert trimmed.tolist() == [[[[True, False, True]]]]
    assert align(trimmed, 5).tolist() == [[[[True, True, True, False, True]]]]


def test_filter_batch_keeps_padding_and_positions():
    drafter = models.mtp_drafter("qwen", models.tiny_qwen_text_config())
    drafter.reset(
        NS(model=NS(embed_tokens=nn.Embedding(32, 16))), left_padding=[0, 1, 2]
    )
    keys = mx.zeros((3, 1, 2, 8))
    drafter._cache[0].update_and_fetch(keys, keys)
    drafter._next_position = mx.array([4, 5, 6])
    drafter.filter_batch([0, 2])
    assert drafter._cache[0].keys.shape[0] == 2
    assert drafter._cache[0].left_padding.tolist() == [0, 2]
    assert drafter._cache[0].offset.tolist() == [2, 0]
    assert drafter._next_position.tolist() == [4, 6]


def test_speculative_dispatch_errors_and_hidden_state():
    with pytest.raises(ValueError):
        speculative.get_speculative_rounds_batch("nope")
    hidden = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    assert (
        speculative.speculative_hidden_state("mtp", NS(hidden_states=hidden))
        is hidden[-1]
    )


def test_speculative_lifetime_counters_survive_reset():
    drafter = NS(accept_lens=[], draft_lens=[])
    snapshot = common.speculative_stats_snapshot(drafter)
    assert common.speculative_stats_since(drafter, snapshot) == (None, None, None)
    common._record_speculative_round(drafter, 3, 7)
    common._record_speculative_round(drafter, 2.5, 7)
    drafter.accept_lens, drafter.draft_lens = [], []
    common._record_speculative_round(drafter, 1.5, 7)
    assert common.speculative_stats_since(drafter, snapshot) == (3, 7, 21)
    snapshot = common.speculative_stats_snapshot(drafter)
    common._record_speculative_round(drafter, 2, 7)
    assert common.speculative_stats_since(drafter, snapshot) == (1, 2, 7)
