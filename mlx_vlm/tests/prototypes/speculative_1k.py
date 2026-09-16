"""Opt-in speculative-suite prototype; run this file explicitly with pytest.

The 1,500-line budget includes speculative_fixtures.py. No legacy tests are imported.
"""

import importlib
import json
from copy import deepcopy
from math import prod
from types import SimpleNamespace as NS

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.generate.ar import _make_cache, generate_step
from mlx_vlm.models import fast_ops
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.models.linear import native_batch_linear
from mlx_vlm.models.quantized_verifier import (
    decode_quantized_argmax,
    decode_quantized_linear,
    exact_quantized_moe_hc_expand,
    exact_quantized_selected_linear,
)
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.common import _SpeculativeSamplerRNG
from mlx_vlm.speculative.dflash import _dflash_rounds, _dflash_rounds_batch
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.eagle3 import _eagle3_rounds, _eagle3_rounds_batch
from mlx_vlm.speculative.mtp import (
    _mtp_logits_from_hidden,
    _mtp_rounds,
    _mtp_rounds_batch,
)
from mlx_vlm.speculative.ops import linear as verifier_linear
from mlx_vlm.speculative.utils import _mtp_verify_target, _speculative_walk_batch
from mlx_vlm.split_mtp import split_mtp
from mlx_vlm.tests.speculative_fixtures import (
    tiny_deepseek_config,
    tiny_glm_text_config,
    tiny_qwen_text_config,
)
from mlx_vlm.utils import get_model_and_args

TEXT = {
    "qwen": ("qwen3_5", tiny_qwen_text_config),
    "glm": ("glm5_next", tiny_glm_text_config),
    "deepseek": ("deepseek_v4", tiny_deepseek_config),
}


def module(name):
    return importlib.import_module("mlx_vlm." + name)


def greedy(logits):
    return mx.argmax(logits, axis=-1)


def dimensions(**overrides):
    return (
        dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            max_position_embeddings=128,
        )
        | overrides
    )


def language(family, *, inference=False):
    name, factory = TEXT[family]
    config = factory()
    if family == "qwen":
        config.num_hidden_layers = config.full_attention_interval = 2
        if inference:
            config.linear_key_head_dim = config.linear_value_head_dim = 32
        outer = NS(
            model_type=name,
            text_config=config,
            vision_config=NS(spatial_merge_size=2),
            image_token_id=30,
            video_token_id=29,
            vision_start_token_id=28,
        )
        return module(f"models.{name}.language").LanguageModel(config, outer), config
    if family == "deepseek":
        config.compress_ratios = [4]
    return module(f"models.{name}.language").LanguageModel(config), config


def dflash_target(family):
    if family in ("dflash2", "dspark-qwen"):
        model, _ = language("qwen")
        model.set_dtype(mx.bfloat16)

        def embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
            positions, deltas = model.get_rope_index(input_ids, attention_mask=mask)
            return InputEmbeddingsFeatures(
                inputs_embeds=model.model.embed_tokens(input_ids),
                position_ids=positions,
                rope_deltas=deltas,
            )

        return NS(language_model=model, get_input_embeddings=embeddings)
    if family.startswith("dspark-lfm"):
        moe = family.endswith("moe")
        name = "lfm2_moe" if moe else "lfm2"
        arch = module("models." + name)
        values = dict(
            model_type=name,
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            rope_theta=10000.0,
            layer_types=["conv", "full_attention", "conv"],
            tie_word_embeddings=True,
        )
        values.update(
            dict(
                intermediate_size=16,
                moe_intermediate_size=8,
                num_experts=4,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                use_expert_bias=True,
                num_dense_layers=1,
            )
            if moe
            else dict(
                block_dim=8,
                block_ff_dim=16,
                block_multiple_of=1,
                block_ffn_dim_multiplier=1.0,
                block_auto_adjust_ff_dim=False,
                full_attn_idxs=[1],
            )
        )
        return arch.Model(arch.ModelConfig(**values))
    arch = module("models.muse_glimmer")
    text = arch.TextConfig(
        **dimensions(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            layer_rope_theta=[10000.0, 0],
        )
    )
    vision = arch.VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    return arch.Model(
        arch.ModelConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=7,
            video_token_id=6,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    )


def dflash_config(family):
    if family == "glimmer":
        return dimensions(
            model_type="muse_glimmer_assistant",
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=64,
            sliding_window=8,
            block_size=4,
            mask_token_id=63,
            target_layer_ids=[0, 1],
            num_target_layers=2,
        )
    lfm = family.startswith("dspark-lfm")
    config = dimensions(
        model_type="qwen3",
        hidden_size=8 if lfm else 16,
        intermediate_size=16 if lfm else 32,
        head_dim=4 if lfm else 8,
        architectures=["Lfm2DSparkDraftModel" if lfm else "DSparkDraftModel"],
        rms_norm_eps=1e-5 if lfm else 1e-6,
        rope_theta=10000.0,
        layer_types=["full_attention"],
        block_size=3,
        markov_rank=4,
        markov_head_type="vanilla",
        enable_confidence_head=True,
    )
    config["dflash_config"] = dict(
        mask_token_id=31, target_layer_ids=[0, 2] if lfm else [0]
    )
    if lfm:
        config["rope_is_neox_style"] = False
        config["dflash_config"]["num_target_layers"] = 3
    else:
        config["num_target_layers"] = 2
        config["dflash_config"]["projector_type"] = "dspark"
    if family == "dflash2":
        config.update(architectures=["DFlash2DraftModel"], is_causal=False)
        config["dflash_config"] = dict(
            block_size=3,
            runtime_block_size=3,
            conv_group_size=4,
            conv_kernel_size=2,
            mask_token_id=31,
            selector_rank=4,
            selector_top_k=4,
            target_layer_ids=[0],
        )
    return config


def dflash_drafter(family):
    config = dflash_config(family)
    arch, name = get_model_and_args(config)
    expected = (
        "dflash2"
        if family == "dflash2"
        else "muse_glimmer_assistant" if family == "glimmer" else "dspark"
    )
    assert name == expected
    return arch.Model(arch.ModelConfig.from_dict(config))


def mtp_drafter(family, config):
    arch = module(f"speculative.drafters.{TEXT[family][0]}_mtp")
    config.mtp_num_hidden_layers = 1
    drafter = arch.Model(arch.ModelConfig(text_config=config, block_size=4))
    drafter.prefer_requested_block_size = True
    return drafter


def generated(target, drafter=None, *, seed=41, temperature=0):
    options = {} if drafter is None else dict(draft_model=drafter, draft_kind="dflash")
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            mx.array([[1, 2, 3, 4]]),
            target,
            None,
            None,
            max_tokens=12,
            temperature=temperature,
            seed=seed,
            prefill_step_size=None,
            **options,
        )
    ]


@pytest.mark.parametrize(
    "family", ["dflash2", "glimmer", "dspark-lfm2", "dspark-lfm2-moe", "dspark-qwen"]
)
@pytest.mark.parametrize("temperature", [0, 0.5, 1.0])
def test_generation_and_request_reset(family, temperature):
    mx.random.seed(37)
    target, drafter = dflash_target(family), dflash_drafter(family)
    validate_drafter_compatibility(target, drafter, "dflash")
    expected = generated(target, temperature=temperature)
    for _ in range(2):
        assert generated(target, drafter, temperature=temperature) == expected
        assert drafter.draft_lens
        assert 999 not in drafter.accept_lens + drafter.draft_lens
        drafter.accept_lens.append(999)
        drafter.draft_lens.append(999)


@pytest.mark.parametrize("family", list(TEXT))
def test_mtp_generation(family):
    mx.random.seed(2127)
    model, config = language(family, inference=True)
    drafter = mtp_drafter(family, config)
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
    actual = [first] + [
        token
        for token, _ in _mtp_rounds(
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
    ]
    assert actual == expected
    assert drafter._round_appended == 0


@pytest.mark.parametrize("family", list(TEXT))
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_verify_commit_matches_decode(family, batch, dtype):
    mx.random.seed(2127)
    model, _ = language(family, inference=True)
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
            _mtp_logits_from_hidden(model, result.hidden),
            result.rollback_state,
        )

    def equal(actual, expected):
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
        equal(hidden, mx.concatenate(hidden_steps, 1))
        equal(logits, mx.concatenate(logit_steps, 1))
        assert mx.array_equal(
            greedy(logits), greedy(mx.concatenate(logit_steps, 1))
        ).item()
        transaction.commit([retained] * batch)
        for i in range(retained):
            _, _, transaction = verify(tokens[:, i : i + 1], reference)
            transaction.commit([1] * batch)
        probe = mx.full((batch, 1), 20)
        equal(model(probe, cache=reference).logits, model(probe, cache=actual).logits)


def formats(groups=(64,)):
    return [
        ("affine", bits, size) for size in groups for bits in (2, 3, 4, 5, 6, 8)
    ] + [("mxfp4", 4, 32), ("mxfp8", 8, 32), ("nvfp4", 4, 16)]


def bf16_parameters(linear):
    linear.scales = linear.scales.astype(mx.bfloat16)
    if linear.biases is not None:
        linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


@pytest.mark.parametrize("mode,bits,size", formats())
@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 9, 16, 32, 64, 127])
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
    actual = decode_quantized_linear(linear, inputs)
    assert mx.array_equal(native_batch_linear(linear, inputs), native).item()
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(
        decode_quantized_argmax(linear, inputs), greedy(expected)
    ).item()
    allowed = mx.arange(batch * 3, dtype=mx.int32).reshape(batch, 3) % 16
    mask = (mx.array(1, dtype=mx.int32) << allowed).reshape(-1, 1)
    assert mx.array_equal(
        decode_quantized_argmax(linear, inputs, token_mask=mask), allowed
    ).item()


@pytest.mark.parametrize("mode,bits,size", formats((32, 64, 128)))
@pytest.mark.parametrize("batch", [1, 4, 8, 64, 127])
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
    routed = exact_quantized_selected_linear(linear, inputs, indices)
    expected = fast_ops.exact_hc_expand(
        SwitchGLU._combine(routed, weights, shared), residual, post, comb
    )

    def fail(*args, **kwargs):
        raise AssertionError("supported formats must use the fused backend")

    monkeypatch.setattr(
        "mlx_vlm.models.quantized_verifier.exact_quantized_selected_linear", fail
    )
    actual = exact_quantized_moe_hc_expand(
        linear, inputs, indices, weights, shared, residual, post, comb
    )
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("widths", [(16, 24), (16, 24, 32), (8, 16, 24, 32)])
@pytest.mark.parametrize("length", [2, 3, 6, 8])
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
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))


class TransactionTarget:
    def __init__(self, token):
        self.token, self.transaction = token, None

    def __call__(self, inputs, cache, **kwargs):
        batch, length = inputs.shape
        self.transaction = start_speculative_cache(cache, length)
        states = cache[0][0][:, None] + mx.arange(1, length + 1)[None, :, None]
        cache[0][0] = states[:, -1]
        cache[0].record_speculative_states(0, states[:, :-1], states[:, -1])
        kv = mx.zeros((batch, 1, length, 1))
        cache[1].update_and_fetch(kv, kv)
        return LanguageModelOutput(
            logits=mx.broadcast_to(mx.eye(8)[self.token], (batch, length, 8)),
            hidden_states=[mx.zeros((batch, length, 4))],
            shared_kv_states={},
            gdn_states=self.transaction,
        )

    def speculative_verify_logits(self, inputs, cache, sampler):
        output = self(inputs, cache)
        try:
            return (
                output.hidden_states[0],
                {},
                output.gdn_states,
                sampler(output.logits),
            )
        except BaseException:
            output.gdn_states.abort()
            raise

    def rollback_speculative_cache(self, *args):
        raise AssertionError("transactions must own cache commit")


class TransactionDrafter:
    prefer_requested_block_size = True

    def __init__(self):
        self.config = NS(block_size=2, target_layer_ids=[0])
        self.accept_lens, self.draft_lens = [], []

    def reset(self, model, left_padding=None):
        return []

    def make_cache(self):
        return []

    def set_shared_kv(self, *args, **kwargs):
        pass

    def draft_block(
        self, bonus, hidden, cache, block_size, sampler, token_dtype, **kwargs
    ):
        return mx.full((hidden.shape[0], block_size - 1), 4, dtype=token_dtype)


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("token,failure", [(4, False), (7, False), (4, True)])
def test_round_commit_close_and_abort(kind, batch, token, failure):
    target = TransactionTarget(token)
    caches = [ArraysCache(1), BatchKVCache([0] * batch) if batch > 1 else KVCache()]
    caches[0][0] = mx.zeros((batch, 1))
    initial = mx.zeros((batch, 1, 2, 1))
    caches[1].update_and_fetch(initial, initial)

    def sample(logits):
        if failure:
            raise RuntimeError("injected sampler failure")
        return greedy(logits)

    functions = {
        "mtp": (_mtp_rounds, _mtp_rounds_batch),
        "dflash": (_dflash_rounds, _dflash_rounds_batch),
        "eagle3": (_eagle3_rounds, _eagle3_rounds_batch),
    }
    options = dict(
        first_bonus=1 if batch == 1 else mx.ones((batch,), dtype=mx.int32),
        max_tokens=4,
        sampler=sample,
        draft_block_size=2,
    )
    if kind == "mtp":
        options["shared_kv_states"] = {}
    generator = functions[kind][batch > 1](
        target, TransactionDrafter(), caches, mx.zeros((batch, 1, 4)), **options
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected sampler failure"):
            next(generator)
    else:
        next(generator)
    retained = 0 if failure else 2 if token == 4 else 1
    assert not target.transaction.active and not caches[0].is_speculating
    assert caches[0][0].tolist() == [[float(retained)]] * batch
    offset = caches[1].offset
    assert (offset.tolist() if isinstance(offset, mx.array) else [offset]) == [
        2 + retained
    ] * batch
    generator.close()


@pytest.mark.parametrize("retained", [0, 1, 2, 3])
@pytest.mark.parametrize("budget", [0, 1, 4])
def test_acceptance_walk(retained, budget):
    drafts = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    targets = mx.concatenate([drafts[:, :retained], mx.full((2, 4 - retained), 9)], 1)
    expected = [(row[:retained] + [9])[:budget] for row in drafts.tolist()]
    accepted, tokens = _speculative_walk_batch(drafts, targets, [budget, budget])
    assert accepted == [retained, retained]
    assert tokens == expected


@pytest.mark.parametrize("budgets", [[1], [1, 1, 1], [1, -1]])
def test_invalid_budgets(budgets):
    with pytest.raises(ValueError):
        _speculative_walk_batch(mx.ones((2, 1)), mx.ones((2, 2)), budgets)


def test_sampler_rng_isolation():
    logits = mx.zeros((1, 8))
    sample = lambda: mx.random.categorical(logits)
    mx.random.seed(123)
    expected = [sample(), sample()]
    mx.eval(*expected)
    mx.random.seed(123)
    drafter = NS(_seed_token=None)
    rng = _SpeculativeSamplerRNG(drafter, enabled=True)
    actual = [sample()]
    mx.eval(*actual)
    rng.target_sampled()
    rng.draft_call(lambda: setattr(drafter, "_seed_token", sample()))
    actual.append(sample())
    mx.eval(*actual)
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))
    assert drafter._seed_token is not None


@pytest.mark.parametrize(
    "family,quant", [("qwen", "mxfp8"), ("glm", "affine"), ("glm", "mxfp8")]
)
def test_split_and_requantize_checkpoint(tmp_path, family, quant):
    name, factory = TEXT[family]
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
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    output = tmp_path / "draft"
    kwargs = (
        dict(q_bits=4, q_group_size=64) if quant == "affine" else dict(q_mode="mxfp8")
    )
    split_mtp(str(tmp_path), str(output), **kwargs)
    result = json.loads((output / "config.json").read_text())
    written = mx.load(str(output / "model.safetensors"))
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


@pytest.mark.parametrize("family", ["dflash2", "glimmer", "dspark-lfm2", "dspark-qwen"])
def test_checkpoint_routing_and_validation(tmp_path, family):
    values = dflash_config(family)
    (tmp_path / "config.json").write_text(json.dumps(values))
    assert resolve_drafter_kind(tmp_path) == "dflash"
    drafter = dflash_drafter(family)
    assert drafter.config.block_size >= 2
    target = dflash_target(family)
    validate_drafter_compatibility(target, drafter, "dflash")
    if family == "glimmer":
        target.language_model.config.model_type = "other"
    else:
        config = target.language_model.config
        getattr(config, "text_config", config).hidden_size += 1
    with pytest.raises(ValueError):
        validate_drafter_compatibility(target, drafter, "dflash")


def dspark_source(model, cfg):
    """Reconstruct the ``mtp.<stage>.*`` checkpoint tensors from the drafter's
    own params, so the split->load round-trip can be exercised without the real
    (multi-GB) DeepSeek-V4-Flash-0731 checkpoint."""
    params = dict(tree_flatten(model.parameters()))
    text = cfg.text_config
    n_experts, o_groups, o_lora_rank = (
        text.n_routed_experts,
        text.o_groups,
        text.o_lora_rank,
    )
    last_stage = cfg.n_mtp_layers - 1
    proj_to_w = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    hc = {"attn_hc": "hc_attn", "ffn_hc": "hc_ffn"}
    src = {}
    for key, value in params.items():
        if key.startswith("markov_head."):
            # the model-level markov head lives under the last stage on disk
            src[f"mtp.{last_stage}.{key}"] = value
            continue
        _, stage, body = key.split(".", 2)
        prefix = f"mtp.{stage}."
        if body.startswith("ffn.switch_mlp."):
            w = proj_to_w[body.split(".")[-2]]
            for expert in range(n_experts):
                src[f"{prefix}ffn.experts.{expert}.{w}.weight"] = value[expert]
        elif body.startswith("ffn.shared_experts."):
            w = proj_to_w[body.split(".")[-2]]
            src[f"{prefix}ffn.shared_experts.{w}.weight"] = value
        elif body == "ffn.gate.e_score_correction_bias":
            src[f"{prefix}ffn.gate.bias"] = value
        elif body == "attn.wo_a.weight":
            src[f"{prefix}attn.wo_a.weight"] = (
                value.reshape(o_groups * o_lora_rank, -1) if value.ndim == 3 else value
            )
        elif body.startswith("attn_hc.") or body.startswith("ffn_hc."):
            module, param = body.split(".")
            src[f"{prefix}{hc[module]}_{param}"] = value
        elif body.startswith("hc_head."):
            src[f"{prefix}hc_head_{body.split('.')[-1]}"] = value
        else:
            src[f"{prefix}{body}"] = value
    return src


def test_deepseek_dspark_split_load_and_draft(tmp_path):
    arch = module("speculative.drafters.deepseek_v4_dspark")
    text = tiny_deepseek_config()
    cfg = arch.ModelConfig(
        text_config=text,
        n_mtp_layers=3,
        target_layer_ids=[0, 1, 2],
        mask_token_id=1,
        markov_rank=8,
        block_size=5,
    )
    original = arch.Model(cfg)
    source = dspark_source(original, cfg)
    source["mtp.2.confidence_head.proj.weight"] = mx.zeros((1, 24))
    source["mtp.0.ffn.gate.bias_vl"] = mx.zeros((2,))
    values = {**text.to_dict(), "model_type": "deepseek_v4"}
    values.update(
        dspark_block_size=4,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=8,
    )
    (tmp_path / "config.json").write_text(json.dumps(values))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.foo": "model.safetensors"}})
    )
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), source)
    output = tmp_path / "draft"
    split_mtp(str(tmp_path), str(output))
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "deepseek_v4_dspark"
    assert config["n_mtp_layers"] == 3 and config["target_layer_ids"] == [0, 1, 2]
    shards = sorted(output.glob("model-*.safetensors"))
    assert len(shards) == 3 and (output / "model.safetensors.index.json").exists()
    weights = {
        key: value for shard in shards for key, value in mx.load(str(shard)).items()
    }
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


@pytest.mark.parametrize("accepted", [0, 2])
def test_eagle3_draft_replay(accepted):
    arch = module("speculative.drafters.eagle3")
    cfg = arch.ModelConfig(
        transformer_layer_config=dimensions(),
        draft_vocab_size=16,
        target_hidden_size=16,
        target_layer_ids=[1, 2, 3],
        block_size=4,
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


@pytest.mark.parametrize("offset,length", [(128, 8), (10, 6)])
def test_rotating_drafter_masks(offset, length):
    masks = module("speculative.drafters.gemma4_assistant.masks")
    kv = (mx.zeros((1, 1, length, 4)), mx.zeros((1, 1, length, 4)))
    result = masks.make_drafter_masks(
        {"sliding_attention": kv},
        query_len=1,
        query_offset=offset,
        sliding_window=4,
        kv_valid_len=mx.array([offset]),
    )
    assert (
        result["sliding_attention"][0, 0, 0].tolist()
        == [-float("inf")] * (length - 3) + [0.0] * 3
    )


def test_laguna_checkpoint_contract():
    arch = module("speculative.drafters.laguna_dflash.config")
    values = dimensions(
        model_type="laguna",
        draft_vocab_size=32,
        rope_theta=10000.0,
        layer_types=["sliding_attention"],
        sliding_window=8,
        gating="per-head",
        eagle_aux_hidden_state_layer_ids=[1],
        dflash_config=dict(
            block_size=3,
            mask_token_id=31,
            target_layer_ids=[0],
            num_target_layers=2,
            causal=True,
        ),
    )
    config = arch.DFlashConfig.from_dict(values)
    expected = arch.expected_laguna_dflash_weight_shapes(config)
    weights = {key: NS(shape=shape) for key, shape in expected.items()}
    arch.validate_laguna_dflash_weights(weights, config)
    arch.validate_laguna_dflash_target(
        config, target_model_config=NS(num_hidden_layers=2, vocab_size=32)
    )
    for field in ("hidden_size", "layer_types", "dflash_config"):
        malformed = dict(values)
        del malformed[field]
        with pytest.raises(ValueError):
            arch.DFlashConfig.from_dict(malformed)
    weights["unexpected.weight"] = NS(shape=(1,))
    with pytest.raises(ValueError, match="weight keys"):
        arch.validate_laguna_dflash_weights(weights, config)


@pytest.mark.parametrize("padding", [[5, 0], [5, 5]])
def test_padded_prefill_chunks(padding):
    lm, cfg = language("qwen")
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


@pytest.mark.parametrize(
    "family,accepted", [("qwen", [1, 0]), ("qwen", [1, 1]), ("deepseek", [0, 0])]
)
def test_batched_drafter_commit_and_filter(family, accepted):
    cfg = TEXT[family][1]()
    drafter = mtp_drafter(family, cfg)
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


@pytest.mark.parametrize("layout", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("nested", [False, True])
def test_gemma_dspark_contract_and_attention(layout, nested):
    arch = module("speculative.drafters.gemma4_dspark.gemma4_dspark")
    values = dimensions(
        model_type="gemma4_text",
        architectures=["Gemma4DSparkModel"],
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        layer_types=[layout],
        sliding_window=8,
        final_logit_softcapping=30.0,
        rope_parameters={layout: dict(rope_theta=10000.0, rope_type="default")},
    )
    draft = dict(
        block_size=3,
        mask_token_id=31,
        target_layer_ids=[0],
        num_target_layers=2,
        markov_rank=4,
        enable_confidence_head=True,
    )
    values.update({"dflash_config": draft} if nested else draft)
    cfg = arch.ModelConfig.from_dict(values)
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
        malformed = deepcopy(values)
        malformed.get("dflash_config", malformed)[key] = value
        with pytest.raises(ValueError):
            arch.ModelConfig.from_dict(malformed)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("positioned", [False, True])
@pytest.mark.parametrize("reject_first", [False, True])
def test_deferred_acceptance(uniform, positioned, reject_first):
    mtp = module("speculative.mtp")
    head_calls, sample_calls = [], []

    def logits(hidden):
        head_calls.append(hidden.shape)
        return hidden

    def sample(logits, *, row_ids, positions):
        sample_calls.append((list(row_ids), list(positions)))
        return greedy(logits)

    rows = [[2, 1, 3], [1, 2, 1] if reject_first else [0, 2, 1]]
    hidden = mx.eye(4)[mx.array(rows)] * 9
    sampler = NS(sample_target=sample) if positioned else greedy
    walk = (
        mtp._speculative_walk_batch_deferred_uniform
        if uniform
        else mtp._speculative_walk_batch_deferred_greedy
    )
    accepted, tokens = walk(
        NS(speculative_logits_from_hidden=logits),
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
    assert len(head_calls) == calls
    if positioned:
        assert sample_calls == [([10, 11], [7 + i, 12 + i]) for i in range(calls)]


@pytest.mark.parametrize("family", ["glm4_moe_lite", "deepseek_v4"])
def test_native_checkpoint_layouts(tmp_path, family):
    deepseek = family == "deepseek_v4"
    cfg = (
        tiny_deepseek_config().to_dict()
        if deepseek
        else dict(
            hidden_size=8,
            vocab_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            qk_nope_head_dim=4,
            v_head_dim=6,
            kv_lora_rank=4,
            moe_intermediate_size=4,
            n_routed_experts=2,
            num_nextn_predict_layers=1,
            tie_word_embeddings=False,
        )
    )
    cfg["model_type"] = family
    if deepseek:
        weights = {
            f"mtp.0.{key}.weight": mx.zeros((4, 4), dtype=mx.uint8)
            for key in ("e_proj", "attn.wq_a")
        }
        weights.update(
            {key.replace(".weight", ".scale"): mx.ones((1, 1)) for key in list(weights)}
        )
        weights["mtp.0.enorm.weight"] = mx.ones((cfg["hidden_size"],))
        for expert in range(cfg["n_routed_experts"]):
            for proj in ("w1", "w2", "w3"):
                key = f"mtp.0.ffn.experts.{expert}.{proj}"
                weights[key + ".weight"] = mx.full((4, 16), expert, dtype=mx.uint8)
                weights[key + ".scale"] = mx.ones((4, 1), dtype=mx.uint8)
        weights["mtp.0.ffn.gate.bias"] = mx.zeros((cfg["n_routed_experts"],))
        weights["mtp.0.hc_attn_fn"] = mx.ones((2, 2))
        weights["mtp.0.hc_head_scale"] = mx.ones((1,))
    else:
        shapes = {
            "embed_tokens": (16, 8),
            "enorm": (8,),
            "hnorm": (8,),
            "eh_proj": (8, 16),
            "input_layernorm": (8,),
            "post_attention_layernorm": (8,),
            "self_attn.kv_b_proj": (20, 4),
            "self_attn.o_proj": (8, 12),
            "mlp.gate": (2, 8),
            "shared_head.norm": (8,),
            "shared_head.head": (16, 8),
        }
        for expert in ["shared_experts", "experts.0", "experts.1"]:
            for proj in ("gate", "up", "down"):
                shapes[f"mlp.{expert}.{proj}_proj"] = (
                    (8, 4) if proj == "down" else (4, 8)
                )
        weights = {
            f"model.layers.2.{key}.weight": mx.zeros(shape)
            for key, shape in shapes.items()
        }
        weights["model.layers.2.mlp.gate.e_score_correction_bias"] = mx.ones((2,))
        weights["model.layers.2.self_attn.rotary_emb.inv_freq"] = mx.ones((2,))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    shard = "mtp.safetensors" if deepseek else "model.safetensors"
    mx.save_safetensors(str(tmp_path / shard), weights)
    index = (
        {"model.foo": "model.safetensors"}
        if deepseek
        else {key: shard for key in weights}
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    output = tmp_path / "draft"
    split_mtp(str(tmp_path), str(output))
    config = json.loads((output / "config.json").read_text())
    out = mx.load(str(output / "model.safetensors"))
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


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("step", [1, 4])
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
        assert all(
            mx.array_equal(a, b).item() for a, b in zip(cache.state, reference.state)
        )
        assert cache.meta_state == reference.meta_state


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4", "deepseek_v4"])
@pytest.mark.parametrize("accepted", [[2, 2], [0, 2]])
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


@pytest.mark.parametrize("family", ["eagle3", "dflash"])
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


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [3, 7, 8])
@pytest.mark.parametrize("bits", [4, 5])
def test_wide_quantized_verifier(batch, length, bits):
    mx.random.seed(43)
    linear = bf16_parameters(
        nn.QuantizedLinear(6144, 32, bias=False, group_size=64, bits=bits)
    )
    inputs = mx.random.normal((batch, length, 6144)).astype(mx.bfloat16)
    # This verifier reproduces native GEMV separately for every token and row.
    expected = verifier_linear._target_verify_singletons(linear, inputs)
    actual = verifier_linear._target_verify_linear(linear, inputs)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(
        verifier_linear._target_verify_quantized_argmax(linear, inputs),
        greedy(expected),
    ).item()


def test_glm_mtp_native_weight_fusion():
    cfg = tiny_glm_text_config()
    arch = module("speculative.drafters.glm5_next_mtp")
    shapes = {
        "mlp.shared_experts.gate_proj": (8, 16),
        "mlp.shared_experts.up_proj": (8, 16),
        "self_attn.q_a_proj": (8, 16),
        "self_attn.kv_a_proj_with_mqa": (4, 16),
        "self_attn.kv_b_proj": (16, 4),
    }
    for expert in range(cfg.n_routed_experts):
        for proj in ("gate", "up", "down"):
            shapes[f"mlp.experts.{expert}.{proj}_proj"] = (
                (16, 8) if proj == "down" else (8, 16)
            )
    weights = {
        f"mtp_block.{key}.weight": mx.arange(prod(shape))
        .reshape(shape)
        .astype(mx.float32)
        for key, shape in shapes.items()
    }
    out = arch.Model.sanitize(NS(args=cfg), weights.copy())
    expected = {
        "mlp.shared_experts.gate_up_proj": (16, 16),
        "mlp.switch_mlp.gate_proj": (2, 8, 16),
        "self_attn.qkv_a_proj": (12, 16),
        "self_attn.embed_q": (2, 4, 4),
        "self_attn.unembed_out": (2, 4, 4),
    }
    for key, shape in expected.items():
        assert out[f"mtp_block.{key}.weight"].shape == shape
    assert mx.array_equal(
        out["mtp_block.mlp.switch_mlp.gate_proj.weight"][1],
        weights["mtp_block.mlp.experts.1.gate_proj.weight"],
    ).item()
