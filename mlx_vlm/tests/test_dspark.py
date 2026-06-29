"""DSpark self-speculative drafter (Gemma 4) regressions.

Covers config parsing, the drafter's checkpoint-key contract and forward path, drafter-kind
auto-detection / routing, compatibility validation, and the losslessness of the single- and
multi-sequence round loops against a history-independent Markov target.

The real published checkpoint is exercised when ``DSPARK_GEMMA4_CONFIG`` (and, for the load
test, ``DSPARK_GEMMA4_CKPT``) point at a local ``config.json`` / safetensors file.
"""

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.speculative.drafters import (
    DSPARK_ARCHITECTURE,
    _config_is_dspark,
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.gemma4_dspark import Model, ModelConfig
from mlx_vlm.speculative.dspark import (
    _confident_prefix_length,
    _dspark_rounds,
    _dspark_rounds_batch,
)
from mlx_vlm.utils import get_model_and_args

DSPARK_CONFIG_ENV = "DSPARK_GEMMA4_CONFIG"
DSPARK_CKPT_ENV = "DSPARK_GEMMA4_CKPT"

# A real DSpark Gemma 4 config (top-level draft hyper-parameters + architecture tag).
REAL_CONFIG = {
    "architectures": [DSPARK_ARCHITECTURE],
    "model_type": "gemma4_text",
    "hidden_size": 3840,
    "num_attention_heads": 16,
    "num_global_key_value_heads": 1,
    "global_head_dim": 512,
    "intermediate_size": 15360,
    "num_hidden_layers": 5,
    "rms_norm_eps": 1e-6,
    "attention_k_eq_v": True,
    "final_logit_softcapping": 30.0,
    "block_size": 7,
    "mask_token_id": 4,
    "markov_rank": 256,
    "enable_confidence_head": True,
    "target_layer_ids": [5, 17, 29, 41, 46],
    "vocab_size": 262144,
    "rope_parameters": {
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
    },
}

TINY_CONFIG = dict(
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
    intermediate_size=24,
    num_hidden_layers=2,
    target_layer_ids=[0, 1],
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    partial_rotary_factor=0.25,
    attention_k_eq_v=True,
    final_logit_softcapping=30.0,
    block_size=4,
    mask_token_id=23,
    markov_rank=8,
    vocab_size=24,
)


def _greedy_argmax(logits):
    return mx.argmax(logits, axis=-1).astype(mx.int32)


# --------------------------------------------------------------------------- config


def test_config_from_dict_maps_gemma_global_shape():
    cfg = ModelConfig.from_dict(REAL_CONFIG)
    assert cfg.hidden_size == 3840
    assert cfg.head_dim == 512  # global_head_dim
    assert cfg.num_key_value_heads == 1  # num_global_key_value_heads
    assert cfg.rope_theta == 1000000.0
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.block_size == 7
    assert cfg.mask_token_id == 4
    assert cfg.markov_rank == 256
    assert tuple(cfg.target_layer_ids) == (5, 17, 29, 41, 46)
    assert cfg.fc_in == 3840 * 5


# --------------------------------------------------------------------------- drafter


def test_drafter_param_keys_roundtrip_through_sanitize():
    model = Model(ModelConfig.from_dict(REAL_CONFIG))
    keys = {k for k, _ in tree_flatten(model.parameters())}
    # Standalone drafter: own embed/head, fc projection, heads, 5 layers.
    assert "embed_tokens.weight" in keys
    assert "lm_head.weight" in keys
    assert "fc.weight" in keys
    assert "markov_head.markov_w1.weight" in keys
    assert "markov_head.markov_w2.weight" in keys
    assert "confidence_head.proj.weight" in keys
    assert "confidence_head.proj.bias" in keys
    assert "layers.4.self_attn.q_proj.weight" in keys
    assert "layers.0.layer_scalar" in keys
    # k_eq_v: no v_proj; weightless v_norm has no parameter.
    assert not any("v_proj" in k for k in keys)
    assert not any("v_norm" in k for k in keys)
    # sanitize strips an optional leading "model." prefix and is otherwise identity.
    prefixed = {f"model.{k}": None for k in keys}
    assert set(model.sanitize(prefixed).keys()) == keys


def test_draft_block_shapes_and_context_only_cache_growth():
    cfg = ModelConfig.from_dict(TINY_CONFIG)
    model = Model(cfg)
    mx.eval(model.parameters())
    cache = model.reset(None)
    assert len(cache) == cfg.num_hidden_layers
    assert all(c.offset == 0 for c in cache)

    prompt_len = 5
    ctx = (mx.random.normal((1, prompt_len, cfg.fc_in)) * 0.1).astype(mx.bfloat16)
    drafts = model.draft_block(7, ctx, cache, cfg.block_size, _greedy_argmax)
    mx.eval(drafts, model._last_confidence)
    assert drafts.shape == (1, cfg.block_size)
    assert model._last_confidence.shape == (1, cfg.block_size)
    # Only the context (not the proposed block) is appended to the draft cache.
    assert all(c.offset == prompt_len for c in cache)
    ids = drafts.tolist()[0]
    assert all(0 <= t < cfg.vocab_size for t in ids)


# --------------------------------------------------------------------------- detection / routing


def _write_config(tmp_path, config) -> Path:
    d = Path(tmp_path)
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_config_is_dspark_by_architecture_and_markers():
    assert _config_is_dspark(REAL_CONFIG)
    # marker cluster without the architecture tag
    markers = {k: v for k, v in REAL_CONFIG.items() if k != "architectures"}
    assert _config_is_dspark(markers)
    # a plain Gemma 4 text config is not DSpark
    assert not _config_is_dspark({"model_type": "gemma4_text", "hidden_size": 3840})
    assert not _config_is_dspark(None)


def test_resolve_drafter_kind_autodetects_and_overrides(tmp_path):
    path = _write_config(tmp_path, REAL_CONFIG)
    assert resolve_drafter_kind(path, None) == "dspark"
    # an explicit wrong kind is corrected to dspark
    assert resolve_drafter_kind(path, "dflash") == "dspark"
    assert resolve_drafter_kind(path, "dspark") == "dspark"


def test_get_model_and_args_routes_to_gemma4_dspark():
    arch, model_type = get_model_and_args(dict(REAL_CONFIG))
    assert model_type == "gemma4_dspark"
    assert arch.Model is Model


def test_validate_compatibility_requires_matching_hidden_and_rollback():
    model = Model(ModelConfig.from_dict(REAL_CONFIG))

    class _LM:
        config = {"hidden_size": 3840}

        def rollback_speculative_cache(self, *a):
            return 0

    class _Target:
        language_model = _LM()

    validate_drafter_compatibility(_Target(), model, "dspark")  # ok

    class _BadHidden(_LM):
        config = {"hidden_size": 2048}

    class _TargetBad:
        language_model = _BadHidden()

    with pytest.raises(ValueError, match="hidden_size"):
        validate_drafter_compatibility(_TargetBad(), model, "dspark")

    class _NoRollback:
        config = {"hidden_size": 3840}

    with pytest.raises(ValueError, match="rollback_speculative_cache"):
        validate_drafter_compatibility(_NoRollback(), model, "dspark")


# --------------------------------------------------------------------------- losslessness


class _StubCache:
    def __init__(self):
        self.offset = 0

    def trim(self, n):
        self.offset -= n


class _MarkovTarget:
    """History-independent Markov target: logits at position i depend only on input_i."""

    def __init__(self, transition, embed, n_targets):
        self.transition = transition
        self.embed = embed
        self.n_targets = n_targets

    def __call__(self, inputs, cache=None, capture_layer_ids=None, **kw):
        L = inputs.shape[1]
        logits = self.transition[inputs]
        hid = self.embed[inputs]
        if cache is not None:
            for c in cache:
                c.offset += L
        n = len(capture_layer_ids) if capture_layer_ids else self.n_targets
        return LanguageModelOutput(logits=logits, hidden_states=[hid] * n)

    def rollback_speculative_cache(self, caches, gdn, accepted, block_size):
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item())
        n = int(accepted) + 1
        trim = block_size - n
        for c in caches:
            if trim > 0:
                c.trim(trim)
        return int(accepted)


def _markov_world(vocab, hid, n_targets, seed=0):
    rng = np.random.default_rng(seed)
    transition = mx.array(rng.standard_normal((vocab, vocab)).astype(np.float32))
    embed = mx.array((rng.standard_normal((vocab, hid)) * 0.1).astype(np.float32))
    return transition, embed, rng


def _plain_greedy(transition, start, n):
    out, t = [], int(start)
    for _ in range(n):
        t = int(mx.argmax(transition[t]).item())
        out.append(t)
    return out


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9])
def test_dspark_rounds_lossless(monkeypatch, threshold):
    monkeypatch.setenv("MLX_VLM_DRAFT_CONFIDENCE_THRESHOLD", str(threshold))
    vocab, hid, nt = TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"], 2
    transition, embed, rng = _markov_world(vocab, hid, nt)
    model = Model(ModelConfig.from_dict(TINY_CONFIG))
    mx.eval(model.parameters())
    target = _MarkovTarget(transition, embed, nt)

    prompt = mx.array(rng.integers(0, vocab, size=(1, 6)).astype(np.int32))
    prompt_cache = [_StubCache() for _ in range(nt)]
    for c in prompt_cache:
        c.offset = prompt.shape[1]
    last = int(prompt[0, -1].item())
    first_bonus = int(mx.argmax(transition[last]).item())
    hidden = mx.concatenate([embed[prompt]] * nt, axis=-1)

    max_tokens = 30
    out = [first_bonus]
    for tok, _ in _dspark_rounds(
        target,
        model,
        prompt_cache,
        hidden,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=_greedy_argmax,
    ):
        out.append(tok)

    assert out == _plain_greedy(transition, last, max_tokens)
    assert len(out) == max_tokens


def test_dspark_rounds_batch_lossless():
    vocab, hid, nt, B = TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"], 2, 3
    transition, embed, rng = _markov_world(vocab, hid, nt, seed=1)
    model = Model(ModelConfig.from_dict(TINY_CONFIG))
    mx.eval(model.parameters())
    target = _MarkovTarget(transition, embed, nt)

    prompt = mx.array(rng.integers(0, vocab, size=(B, 6)).astype(np.int32))
    prompt_cache = [_StubCache() for _ in range(nt)]
    for c in prompt_cache:
        c.offset = prompt.shape[1]
    lasts = [int(prompt[i, -1].item()) for i in range(B)]
    first_bonus = mx.array([int(mx.argmax(transition[t]).item()) for t in lasts])
    hidden = mx.concatenate([embed[prompt]] * nt, axis=-1)

    max_tokens = 20
    seqs = [[int(first_bonus[i].item())] for i in range(B)]
    for tokens_out, _ in _dspark_rounds_batch(
        target,
        model,
        prompt_cache,
        hidden,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=_greedy_argmax,
    ):
        for i in range(B):
            if tokens_out[i] is not None and len(seqs[i]) < max_tokens:
                seqs[i].append(tokens_out[i])

    for i in range(B):
        assert seqs[i] == _plain_greedy(transition, lasts[i], max_tokens), f"row {i}"


def test_confident_prefix_length():
    # logits: positive → high P(accept); negative → low. threshold 0.5 == sigmoid(0).
    logits = mx.array([[5.0, 5.0, -5.0, 5.0]])
    assert _confident_prefix_length(logits, 4, 0.0) == 4  # gating disabled
    assert _confident_prefix_length(logits, 4, 0.5) == 2  # truncates at first below
    assert _confident_prefix_length(mx.array([[5.0, 5.0, 5.0, 5.0]]), 4, 0.5) == 4


# --------------------------------------------------------------------------- real weights


@pytest.mark.skipif(
    not os.environ.get(DSPARK_CONFIG_ENV),
    reason=f"set {DSPARK_CONFIG_ENV} to a real DSpark Gemma 4 config.json",
)
def test_real_config_param_keys_match_checkpoint_header():
    config = json.load(open(os.environ[DSPARK_CONFIG_ENV]))
    ckpt = os.environ.get(DSPARK_CKPT_ENV)
    if not ckpt or not os.path.exists(ckpt):
        pytest.skip(f"set {DSPARK_CKPT_ENV} to the matching safetensors file")
    model = Model(ModelConfig.from_dict(config))
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    with open(ckpt, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    ckpt_keys = {k for k in header if k != "__metadata__"}
    ckpt_keys = set(model.sanitize({k: None for k in ckpt_keys}).keys())
    assert ckpt_keys == model_keys


@pytest.mark.skipif(
    not os.environ.get(DSPARK_CKPT_ENV) or not os.environ.get(DSPARK_CONFIG_ENV),
    reason=f"set {DSPARK_CONFIG_ENV} and {DSPARK_CKPT_ENV} to the real checkpoint",
)
def test_real_checkpoint_loads_and_drafts():
    config = json.load(open(os.environ[DSPARK_CONFIG_ENV]))
    ckpt = os.environ[DSPARK_CKPT_ENV]
    if not os.path.exists(ckpt):
        pytest.skip("checkpoint file not present")
    cfg = ModelConfig.from_dict(config)
    model = Model(cfg)
    weights = model.sanitize(mx.load(ckpt))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    cache = model.reset(None)
    ctx = (mx.random.normal((1, 8, cfg.fc_in)) * 0.1).astype(mx.bfloat16)
    drafts = model.draft_block(5, ctx, cache, cfg.block_size, _greedy_argmax)
    mx.eval(drafts)
    ids = drafts.tolist()[0]
    assert len(ids) == cfg.block_size
    assert all(0 <= t < cfg.vocab_size for t in ids)
    # Trained weights should not collapse the whole block to the mask token.
    assert any(t != cfg.mask_token_id for t in ids)
