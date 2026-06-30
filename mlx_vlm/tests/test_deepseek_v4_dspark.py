"""Tests for the DeepSeek-V4 DSpark self-speculative drafter.

Covers config mapping (Flash/Pro-driven), the checkpoint sanitize key roundtrip, draft_block
shapes, numerical parity of ``forward_spec`` against the dspark-mlx reference drafter (the
validated MLX port of the official ``inference/model.py``; env/import-gated), and the
losslessness of the single- and batched round loops.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.speculative.drafters import (
    _config_is_dspark,
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.deepseek_v4_dspark import (
    DeepseekV4DSparkConfig as ModelConfig,
)
from mlx_vlm.speculative.drafters.deepseek_v4_dspark import (
    DeepseekV4DSparkDraftModel as Model,
)
from mlx_vlm.speculative.dspark import _dspark_rounds, _dspark_rounds_batch
from mlx_vlm.utils import get_model_and_args

# Tiny config that exercises every path (score-routed MoE, HC, windowed attn, 2 stages).
TINY_CONFIG = dict(
    model_type="deepseek_v4_dspark",
    vocab_size=32,
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
    qk_rope_head_dim=8,
    q_lora_rank=16,
    o_lora_rank=8,
    o_groups=2,
    sliding_window=8,
    rope_theta=10000.0,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    num_hash_layers=0,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    norm_topk_prob=True,
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    num_hidden_layers=2,
    n_mtp_layers=2,
    block_size=4,
    noise_token_id=31,
    markov_rank=8,
    target_layer_ids=[0, 1],
)


def _greedy_argmax(logits):
    return mx.argmax(logits, axis=-1).astype(mx.int32)


# --------------------------------------------------------------------------- config


def test_config_from_dict_maps_dspark_fields():
    cfg = ModelConfig.from_dict(
        {
            "model_type": "deepseek_v4",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "n_routed_experts": 384,
            "o_groups": 16,
            "q_lora_rank": 1536,
            "sliding_window": 128,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [58, 59, 60],
            "dspark_markov_rank": 512,
        }
    )
    # Pro-scale dims absorbed; dspark_* fields mapped to the drafter's plain names.
    assert cfg.hidden_size == 7168
    assert cfg.n_routed_experts == 384
    assert cfg.o_groups == 16
    assert cfg.block_size == 5
    assert cfg.noise_token_id == 128799
    assert cfg.target_layer_ids == [58, 59, 60]
    assert cfg.markov_rank == 512
    assert cfg.fc_in == 7168 * 3


def test_config_defaults_are_flash_scale():
    cfg = ModelConfig()
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 43
    assert cfg.n_routed_experts == 256
    assert cfg.n_mtp_layers == 3
    assert cfg.fc_in == 4096 * 3


# --------------------------------------------------------------------------- routing


def test_config_is_dspark_detects_deepseek_v4():
    assert _config_is_dspark({"model_type": "deepseek_v4_dspark"})
    # the base combined checkpoint carries the dspark_* hyper-parameters
    assert _config_is_dspark({"model_type": "deepseek_v4", "dspark_block_size": 5})
    # a plain DeepSeek-V4 base config (the target) is NOT a DSpark drafter
    assert not _config_is_dspark({"model_type": "deepseek_v4", "hidden_size": 4096})


def test_resolve_drafter_kind_for_deepseek_v4_dspark(tmp_path):
    d = Path(tmp_path)
    (d / "config.json").write_text(json.dumps({"model_type": "deepseek_v4_dspark"}))
    assert resolve_drafter_kind(d, None) == "dspark"
    assert resolve_drafter_kind(d, "dspark") == "dspark"
    # a mistaken kind is corrected to dspark
    assert resolve_drafter_kind(d, "mtp") == "dspark"


def test_get_model_and_args_routes_to_deepseek_v4_dspark():
    arch, model_type = get_model_and_args(dict(TINY_CONFIG))
    assert model_type == "deepseek_v4_dspark"
    assert arch.Model is Model


def test_validate_compatibility_requires_matching_hidden_and_rollback():
    model = Model(ModelConfig.from_dict(TINY_CONFIG))

    class _LM:
        config = {"hidden_size": TINY_CONFIG["hidden_size"]}

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
        config = {"hidden_size": TINY_CONFIG["hidden_size"]}

    with pytest.raises(ValueError, match="rollback_speculative_cache"):
        validate_drafter_compatibility(_NoRollback(), model, "dspark")


# --------------------------------------------------------------------------- sanitize


def _drafter_key_to_checkpoint(key: str):
    """Invert sanitize: a drafter param path -> the checkpoint key(s) that produce it."""
    if key in ("embed.weight", "head.weight"):
        return [key]
    assert key.startswith("blocks.")
    ck = "mtp." + key[len("blocks.") :]
    ck = ck.replace(".ffn.gate.e_score_correction_bias", ".ffn.gate.bias")
    for sub in ("attn", "ffn"):
        for p in ("fn", "base", "scale"):
            ck = ck.replace(f".{sub}_hc.{p}", f".hc_{sub}_{p}")
    ck = (
        ck.replace(".hc_head.fn", ".hc_head_fn")
        .replace(".hc_head.base", ".hc_head_base")
        .replace(".hc_head.scale", ".hc_head_scale")
        .replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
        .replace(".shared_experts.down_proj.", ".shared_experts.w2.")
        .replace(".shared_experts.up_proj.", ".shared_experts.w3.")
    )
    return [ck]


def test_drafter_param_keys_roundtrip_through_sanitize():
    cfg = ModelConfig.from_dict(TINY_CONFIG)
    model = Model(cfg)
    mx.eval(model.parameters())
    params = dict(tree_flatten(model.parameters()))

    # Build a synthetic checkpoint (mtp.* + per-expert experts) from the param tree.
    checkpoint = {}
    w_inv = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    for key, value in params.items():
        if ".ffn.switch_mlp." in key:
            stage = key.split(".")[1]
            proj = key.split(".")[4]  # gate_proj / down_proj / up_proj
            for e in range(cfg.n_routed_experts):
                ck = f"mtp.{stage}.ffn.experts.{e}.{w_inv[proj]}.weight"
                checkpoint[ck] = value[e]
        else:
            for ck in _drafter_key_to_checkpoint(key):
                checkpoint[ck] = value
    # base-model keys must be dropped
    checkpoint["model.layers.0.attn.wkv.weight"] = mx.zeros((4, 4))

    out = model.sanitize(checkpoint)
    assert set(out.keys()) == set(params.keys())
    # the stacked routed experts are reconstructed at full [E, ...] shape
    sm = "blocks.0.ffn.switch_mlp.gate_proj.weight"
    assert out[sm].shape == params[sm].shape


# --------------------------------------------------------------------------- draft_block


def test_draft_block_shapes_and_window_growth():
    cfg = ModelConfig.from_dict(TINY_CONFIG)
    model = Model(cfg)
    mx.eval(model.parameters())
    cache = model.reset(None)
    assert len(cache.windows) == cfg.n_mtp_layers
    assert cache.offset == 0

    prompt_len = 5
    ctx = (mx.random.normal((1, prompt_len, cfg.fc_in)) * 0.1).astype(mx.float32)
    drafts = model.draft_block(7, ctx, cache, cfg.block_size, _greedy_argmax)
    mx.eval(drafts, model._last_confidence)
    assert drafts.shape == (1, cfg.block_size)
    assert model._last_confidence.shape == (1, cfg.block_size)
    # Only the committed context advances the window offset (not the drafted block).
    assert cache.offset == prompt_len
    assert all(0 <= t < cfg.vocab_size for t in drafts.tolist()[0])

    # A second round advances the window by the newly committed segment.
    seg = (mx.random.normal((1, 3, cfg.fc_in)) * 0.1).astype(mx.float32)
    model.draft_block(
        int(drafts[0, -1].item()), seg, cache, cfg.block_size, _greedy_argmax
    )
    assert cache.offset == prompt_len + 3


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


# --------------------------------------------------------------------------- split


def _synthetic_source_checkpoint(tmp_path) -> Path:
    """A tiny bf16 'combined' checkpoint: mtp.* draft + shared embed/head + a base key."""
    cfg = ModelConfig.from_dict(TINY_CONFIG)
    model = Model(cfg)
    mx.eval(model.parameters())
    params = dict(tree_flatten(model.parameters()))

    w_inv = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    checkpoint = {}
    for key, value in params.items():
        if ".ffn.switch_mlp." in key:
            stage, proj = key.split(".")[1], key.split(".")[4]
            for e in range(cfg.n_routed_experts):
                checkpoint[f"mtp.{stage}.ffn.experts.{e}.{w_inv[proj]}.weight"] = value[
                    e
                ].astype(mx.bfloat16)
        else:
            for ck in _drafter_key_to_checkpoint(key):
                checkpoint[ck] = value.astype(mx.bfloat16)
    # a base-model tensor that split must drop
    checkpoint["model.layers.0.attn.wkv.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)

    src = Path(tmp_path) / "source"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), checkpoint)
    source_config = {
        k: v
        for k, v in TINY_CONFIG.items()
        if k
        not in (
            "model_type",
            "block_size",
            "noise_token_id",
            "markov_rank",
            "target_layer_ids",
            "n_mtp_layers",
        )
    }
    source_config["model_type"] = "deepseek_v4"
    source_config["dspark_block_size"] = TINY_CONFIG["block_size"]
    source_config["dspark_noise_token_id"] = TINY_CONFIG["noise_token_id"]
    source_config["dspark_markov_rank"] = TINY_CONFIG["markov_rank"]
    source_config["dspark_target_layer_ids"] = TINY_CONFIG["target_layer_ids"]
    (src / "config.json").write_text(json.dumps(source_config))
    return src


def test_split_extracts_dequantized_drafter(tmp_path):
    from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (
        split_deepseek_v4_dspark,
    )

    src = _synthetic_source_checkpoint(tmp_path)
    out = split_deepseek_v4_dspark(str(src), str(Path(tmp_path) / "drafter"))

    out_config = json.loads((out / "config.json").read_text())
    assert out_config["model_type"] == "deepseek_v4_dspark"
    assert out_config["n_mtp_layers"] == TINY_CONFIG["n_mtp_layers"]
    assert "quantization" not in out_config and "quantization_config" not in out_config

    weights = mx.load(str(out / "model.safetensors"))
    # base-model tensors are dropped; mtp.* + shared embed/head are kept
    assert not any(k.startswith("model.layers.") for k in weights)
    assert "embed.weight" in weights and "head.weight" in weights
    assert any(k.startswith("mtp.0.") for k in weights)

    # the split drafter loads and drafts through the package's own sanitize
    cfg = ModelConfig.from_dict(out_config)
    drafter = Model(cfg)
    sanitized = drafter.sanitize(weights)
    assert set(sanitized.keys()) == {k for k, _ in tree_flatten(drafter.parameters())}
    drafter.load_weights(list(sanitized.items()))
    mx.eval(drafter.parameters())
    cache = drafter.reset(None)
    ctx = (mx.random.normal((1, 5, cfg.fc_in)) * 0.1).astype(mx.bfloat16)
    drafts = drafter.draft_block(7, ctx, cache, cfg.block_size, _greedy_argmax)
    mx.eval(drafts)
    assert drafts.shape == (1, cfg.block_size)


# --------------------------------------------------------------------- real weights (gated)

_SOURCE_ENV = "DSPARK_DEEPSEEK_V4_SOURCE"


@pytest.mark.skipif(
    not os.environ.get(_SOURCE_ENV),
    reason=f"set {_SOURCE_ENV} to a DeepSeek-V4-DSpark checkpoint dir/repo",
)
def test_real_checkpoint_splits_loads_and_drafts(tmp_path):
    from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (
        split_deepseek_v4_dspark,
    )

    out = split_deepseek_v4_dspark(os.environ[_SOURCE_ENV], str(Path(tmp_path) / "d"))
    cfg = ModelConfig.from_dict(json.loads((out / "config.json").read_text()))
    drafter = Model(cfg)
    weights = drafter.sanitize(mx.load(str(out / "model.safetensors")))
    drafter.load_weights(list(weights.items()))
    mx.eval(drafter.parameters())

    cache = drafter.reset(None)
    ctx = (mx.random.normal((1, 8, cfg.fc_in)) * 0.1).astype(mx.bfloat16)
    drafts = drafter.draft_block(5, ctx, cache, cfg.block_size, _greedy_argmax)
    mx.eval(drafts)
    ids = drafts.tolist()[0]
    assert len(ids) == cfg.block_size
    assert all(0 <= t < cfg.vocab_size for t in ids)
    assert len(set(ids)) > 1  # trained weights shouldn't collapse the block


# --------------------------------------------------------- numerical parity (oracle-gated)


def _make_oracle_args():
    """Equivalent DSparkArgs for the dspark-mlx reference drafter."""
    from dspark_mlx.model.config import DSparkArgs

    return DSparkArgs(
        dim=TINY_CONFIG["hidden_size"],
        moe_inter_dim=TINY_CONFIG["moe_intermediate_size"],
        n_layers=TINY_CONFIG["num_hidden_layers"],
        n_mtp_layers=TINY_CONFIG["n_mtp_layers"],
        n_hash_layers=0,
        n_heads=TINY_CONFIG["num_attention_heads"],
        n_routed_experts=TINY_CONFIG["n_routed_experts"],
        n_shared_experts=1,
        n_activated_experts=TINY_CONFIG["num_experts_per_tok"],
        score_func="sqrtsoftplus",
        route_scale=TINY_CONFIG["routed_scaling_factor"],
        swiglu_limit=TINY_CONFIG["swiglu_limit"],
        q_lora_rank=TINY_CONFIG["q_lora_rank"],
        head_dim=TINY_CONFIG["head_dim"],
        rope_head_dim=TINY_CONFIG["qk_rope_head_dim"],
        o_groups=TINY_CONFIG["o_groups"],
        o_lora_rank=TINY_CONFIG["o_lora_rank"],
        window_size=TINY_CONFIG["sliding_window"],
        vocab_size=TINY_CONFIG["vocab_size"],
        hc_mult=TINY_CONFIG["hc_mult"],
        hc_sinkhorn_iters=TINY_CONFIG["hc_sinkhorn_iters"],
        dspark_block_size=TINY_CONFIG["block_size"],
        dspark_noise_token_id=TINY_CONFIG["noise_token_id"],
        dspark_target_layer_ids=tuple(TINY_CONFIG["target_layer_ids"]),
        dspark_markov_rank=TINY_CONFIG["markov_rank"],
        temperature=0.0,
    )


def _bridge_oracle_key(name: str) -> str:
    """dspark-mlx DSparkDrafter param path -> this drafter's param path."""
    nk = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
    for a, b in (
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
        (".hc_head_fn", ".hc_head.fn"),
        (".hc_head_base", ".hc_head.base"),
        (".hc_head_scale", ".hc_head.scale"),
        (".shared_experts.w1.", ".shared_experts.gate_proj."),
        (".shared_experts.w2.", ".shared_experts.down_proj."),
        (".shared_experts.w3.", ".shared_experts.up_proj."),
    ):
        nk = nk.replace(a, b)
    if nk.endswith(".ffn.w1"):
        nk = nk[: -len(".w1")] + ".switch_mlp.gate_proj.weight"
    elif nk.endswith(".ffn.w3"):
        nk = nk[: -len(".w3")] + ".switch_mlp.up_proj.weight"
    elif nk.endswith(".ffn.w2"):
        nk = nk[: -len(".w2")] + ".switch_mlp.down_proj.weight"
    return nk


@pytest.mark.parametrize("ctx_scale", [1.0, 2.0])
def test_forward_spec_matches_dspark_mlx_oracle(ctx_scale):
    """Numerical parity vs the validated dspark-mlx reference (== official model.py)."""
    pytest.importorskip("dspark_mlx")
    from dspark_mlx.model.drafter import DSparkDrafter

    mx.random.seed(11)
    args = _make_oracle_args()
    oracle = DSparkDrafter(args, max_seq_len=64)
    mine = Model(ModelConfig.from_dict(TINY_CONFIG), max_seq_len=64)
    mx.eval(oracle.parameters(), mine.parameters())

    # Identical random weights into both (large shared-expert weights trip the clamp).
    rand = {}
    for k, v in tree_flatten(oracle.parameters()):
        scale = 4.0 if ".shared_experts." in k else (0.5 if ".ffn.w" in k else 0.2)
        rand[k] = (mx.random.normal(v.shape) * scale).astype(v.dtype)
    oracle.update(tree_unflatten(list(rand.items())))
    mine.update(tree_unflatten([(_bridge_oracle_key(k), v) for k, v in rand.items()]))
    mx.eval(oracle.parameters(), mine.parameters())

    S = 6
    ctx = (mx.random.normal((1, S, args.main_proj_in)) * ctx_scale).astype(mx.float32)
    anchor = mx.array([7], dtype=mx.int32)

    oracle.forward_spec(anchor, ctx, start_pos=0)
    o_ids, o_logits, o_conf = oracle.forward_spec(
        anchor, ctx[:, -1:, :], start_pos=S - 1
    )
    mine.forward_spec(anchor, ctx, start_pos=0)
    m_ids, m_logits, m_conf = mine.forward_spec(anchor, ctx[:, -1:, :], start_pos=S - 1)
    mx.eval(o_ids, o_logits, o_conf, m_ids, m_logits, m_conf)

    assert bool(mx.all(o_ids == m_ids).item())
    assert mx.max(mx.abs(o_logits - m_logits)).item() < 1e-3
    assert mx.max(mx.abs(o_conf - m_conf)).item() < 1e-3


def test_advance_window_matches_dspark_mlx_oracle():
    """Parity of the advance/multi-step windowing path vs the reference."""
    pytest.importorskip("dspark_mlx")
    from dspark_mlx.model.drafter import DSparkDrafter

    mx.random.seed(13)
    args = _make_oracle_args()
    oracle = DSparkDrafter(args, max_seq_len=64)
    mine = Model(ModelConfig.from_dict(TINY_CONFIG), max_seq_len=64)
    mx.eval(oracle.parameters(), mine.parameters())
    rand = {}
    for k, v in tree_flatten(oracle.parameters()):
        rand[k] = (mx.random.normal(v.shape) * 0.2).astype(v.dtype)
    oracle.update(tree_unflatten(list(rand.items())))
    mine.update(tree_unflatten([(_bridge_oracle_key(k), v) for k, v in rand.items()]))
    mx.eval(oracle.parameters(), mine.parameters())

    S0 = 4
    full = mx.random.normal((1, S0 + 2, args.main_proj_in)).astype(mx.float32)
    seed, c1, c2 = full[:, :S0], full[:, S0], full[:, S0 + 1]
    anchor = mx.array([9], dtype=mx.int32)

    for drafter in (oracle, mine):
        drafter.forward_spec(anchor, seed, start_pos=0)
        drafter.advance(c1, S0)
        drafter.advance(c2, S0 + 1)

    o = oracle.forward_spec(anchor, full[:, S0 + 1 : S0 + 2], start_pos=S0 + 1)
    m = mine.forward_spec(anchor, full[:, S0 + 1 : S0 + 2], start_pos=S0 + 1)
    mx.eval(*o, *m)

    assert bool(mx.all(o[0] == m[0]).item())
    assert mx.max(mx.abs(o[1] - m[1])).item() < 1e-3
    assert mx.max(mx.abs(o[2] - m[2])).item() < 1e-3
