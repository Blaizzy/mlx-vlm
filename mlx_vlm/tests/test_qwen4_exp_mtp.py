import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.speculative.drafters.qwen4_exp_mtp import Model as Qwen4ExpMTPDraftModel
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import ModelConfig as Qwen4ExpMTPConfig
from mlx_vlm.tests.test_qwen4_exp import TEXT_CONFIG, make_model


def make_drafter(**text_overrides):
    text = dict(TEXT_CONFIG, mtp_num_hidden_layers=1, **text_overrides)
    config = Qwen4ExpMTPConfig.from_dict(
        {"model_type": "qwen4_exp_mtp", "text_config": text}
    )
    drafter = Qwen4ExpMTPDraftModel(config)
    drafter.eval()
    mx.eval(drafter.parameters())
    return drafter


def test_registered_in_the_mtp_splitter_table():
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter

    splitter = get_mtp_splitter("qwen4_exp")
    assert splitter is not None
    assert splitter.output_model_type == "qwen4_exp_mtp"
    assert splitter.depth_field == "mtp_num_hidden_layers"


def test_draft_runs_dense_and_carries_no_ple():
    drafter = make_drafter()
    layer = drafter.layers[0]
    # A sparse-attention-shaped layer, but the indexer is switched off so the
    # attention runs dense and needs no block cache of its own.
    assert not layer.is_linear
    assert layer.self_attn.indexer is None
    assert layer.ple is None
    # No final norm: the mixer is what collapses the streams.
    assert not hasattr(drafter, "norm")


def test_the_checkpoint_mtp_tensor_names_map_onto_the_drafter():
    """The real checkpoint's `mtp.*` names must cover exactly the drafter params."""
    drafter = make_drafter()
    args = drafter.args
    hc_dim = args.hc_count * args.hidden_size

    weights = {
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((args.hidden_size,)),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((hc_dim,)),
        "mtp.fc_embedding.weight": mx.zeros((args.hidden_size, args.hidden_size)),
        "mtp.fc_hidden.weight": mx.zeros((args.hidden_size, args.hidden_size)),
        "mtp.hyper_connection_mixer.hc_norm.weight": mx.zeros((hc_dim,)),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": mx.zeros(
            (args.hc_lowrank, hc_dim)
        ),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": mx.zeros(
            (hc_dim, args.hc_lowrank)
        ),
        # the indexer ships in the checkpoint but the dense draft drops it
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": mx.zeros((24, 32)),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": mx.zeros((8,)),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": mx.zeros((8,)),
        # the MoE experts arrive packed, as in the trunk
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.zeros(
            (args.num_experts, 2 * args.moe_intermediate_size, args.hidden_size)
        ),
        "mtp.layers.0.mlp.experts.down_proj": mx.zeros(
            (args.num_experts, args.hidden_size, args.moe_intermediate_size)
        ),
    }
    for name, shape in (
        (
            "self_attn.q_proj.weight",
            (2 * args.num_attention_heads * args.head_dim, args.hidden_size),
        ),
        (
            "self_attn.k_proj.weight",
            (args.num_key_value_heads * args.head_dim, args.hidden_size),
        ),
        (
            "self_attn.v_proj.weight",
            (args.num_key_value_heads * args.head_dim, args.hidden_size),
        ),
        (
            "self_attn.o_proj.weight",
            (args.hidden_size, args.num_attention_heads * args.head_dim),
        ),
        ("self_attn.q_norm.weight", (args.head_dim,)),
        ("self_attn.k_norm.weight", (args.head_dim,)),
        ("mlp.gate.weight", (args.num_experts, args.hidden_size)),
        (
            "mlp.shared_expert.gate_proj.weight",
            (args.shared_expert_intermediate_size, args.hidden_size),
        ),
        (
            "mlp.shared_expert.up_proj.weight",
            (args.shared_expert_intermediate_size, args.hidden_size),
        ),
        (
            "mlp.shared_expert.down_proj.weight",
            (args.hidden_size, args.shared_expert_intermediate_size),
        ),
        ("mlp.shared_expert_gate.weight", (1, args.hidden_size)),
    ):
        weights[f"mtp.layers.0.{name}"] = mx.zeros(shape)
    for block in ("attn_hyper_connection", "mlp_hyper_connection"):
        weights[f"mtp.layers.0.{block}.hc_norm.weight"] = mx.zeros((hc_dim,))
        weights[f"mtp.layers.0.{block}.input_mix_weight_down.weight"] = mx.zeros(
            (args.hc_lowrank, hc_dim)
        )
        weights[f"mtp.layers.0.{block}.input_mix_weight_up.weight"] = mx.zeros(
            (hc_dim, args.hc_lowrank)
        )
        weights[f"mtp.layers.0.{block}.block_inject_weight.weight"] = mx.zeros(
            (args.hc_count, hc_dim)
        )

    sanitized = drafter.sanitize(weights)
    expected = {k for k, _ in tree_flatten(drafter.parameters())}
    assert set(sanitized) == expected
    # loading must succeed under the strict name/shape check
    Qwen4ExpMTPDraftModel(drafter.config).load_weights(list(sanitized.items()))


def test_splitter_offsets_the_zero_centered_fusion_norms():
    """`pre_fc_norm_*` do not end in `norm.weight`, so they need naming by hand."""
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import Qwen4ExpMTPSplitter

    tensors = {
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((4,)),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((8,)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((4,)),
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight": mx.zeros((8,)),
        "mtp.fc_hidden.weight": mx.zeros((4, 4)),
    }
    out = Qwen4ExpMTPSplitter().rename(dict(tensors), dict(TEXT_CONFIG))
    for key in tensors:
        if key.endswith(".weight") and tensors[key].ndim == 1:
            assert float(out[key].min()) == 1.0, key
    # a matrix is left alone
    assert float(out["mtp.fc_hidden.weight"].max()) == 0.0


def test_verify_hands_over_the_pre_mixer_streams():
    """`speculative_verify_hidden` must yield streams, not the collapsed hidden."""
    target = make_model()
    lm = target.language_model
    args = lm.args
    ids = mx.array([mx.random.randint(4, 90, (12,)).tolist()])

    hidden, _, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    assert hidden.shape == (1, ids.shape[1], args.hc_count * args.hidden_size)
    # logits from those streams go through the mixer first
    logits = lm.speculative_logits_from_hidden(hidden)
    assert logits.shape == (1, ids.shape[1], args.vocab_size)


def test_plain_return_hidden_hands_over_streams_without_switching_to_verify():
    """The MTP prefill shape: `return_hidden` with no capture at all.

    It has to yield the pre-mixer streams, because the head cannot use anything
    else -- but it must not introduce a `capture_layer_ids`, since that also builds
    `gdn_sink` and puts the whole trunk on its per-token target-verify path. So the
    streams come from the trunk directly and `gdn_states` stays empty.
    """
    target = make_model()
    lm = target.language_model
    ids = mx.array([mx.random.randint(4, 90, (8,)).tolist()])
    out = lm(ids, cache=lm.make_cache(), return_hidden=True)

    assert out.gdn_states is None, "a capture was introduced; prefill went to verify"
    assert out.hidden_states[-1].shape[-1] == lm.args.hc_count * lm.args.hidden_size

    # ...and the holder is emptied, so a later call cannot pick up a stale activation
    assert lm.model.streams.value is None
    plain = lm(ids, cache=lm.make_cache())
    assert plain.hidden_states is None
    assert lm.model.streams.value is None


def test_draft_block_predicts_from_the_pre_mixer_streams():
    target = make_model()
    drafter = make_drafter()
    drafter.reset(target.language_model)

    lm = target.language_model
    ids = mx.array([mx.random.randint(4, 90, (12,)).tolist()])
    cache = lm.make_cache()
    out = lm(ids, cache=cache, capture_layer_ids=[], return_hidden=True)

    hidden = out.hidden_states[-1]
    assert hidden.shape == (
        1,
        ids.shape[1],
        drafter.hc_count * drafter.args.hidden_size,
    )

    drafter.set_shared_kv({}, ids.shape[1])
    tokens = drafter.draft_block(
        int(
            mx.argmax(
                lm.speculative_logits_from_hidden(hidden[:, -1:, :]), axis=-1
            ).item()
        ),
        hidden[:, -1:, :],
        None,
        3,
        None,
        greedy=True,
    )
    assert tokens.shape == (1, 2)
    assert not bool(mx.any(tokens < 0))


def test_prefill_skips_seeding_when_given_the_collapsed_hidden():
    target = make_model()
    drafter = make_drafter()
    drafter.reset(target.language_model)
    ids = mx.array([[7, 11, 5, 3, 21, 33]])
    drafter.prefill_from_target_hidden(
        ids,
        mx.zeros((1, ids.shape[1], drafter.args.hidden_size)),
        9,
        None,
        greedy=True,
    )
    # nothing seeded, nothing appended
    assert drafter._seed_token is None
    assert drafter._cache[0].offset == 0


def test_post_mixer_hidden_is_rejected_with_a_clear_error():
    drafter = make_drafter()
    drafter.reset(make_model().language_model)
    with pytest.raises(ValueError, match="pre-mixer hyper-connection"):
        drafter._target_streams(mx.zeros((1, 2, drafter.args.hidden_size)))


def test_detect_finds_the_splitter_for_a_released_style_config(tmp_path):
    """Detection keys off `text_config.model_type`, which is `qwen4_exp_text`.

    Registering only `qwen4_exp` made `convert --mtp` silently skip the drafter --
    it prints "no registered splitter" and carries on, so nothing failed loudly.
    """
    import json

    import mlx.core as mx

    from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import Qwen4ExpMTPSplitter

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {**TEXT_CONFIG, "model_type": "qwen4_exp_text"},
            }
        )
    )
    # detection also insists the tensors are really there
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"mtp.fc_hidden.weight": mx.zeros((2, 2))}
    )

    assert isinstance(detect_mtp_splitter(tmp_path), Qwen4ExpMTPSplitter)

    # ...and a checkpoint with no MTP tensors must still resolve to None
    (tmp_path / "model.safetensors").unlink()
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"model.embed_tokens.weight": mx.zeros((2, 2))},
    )
    assert detect_mtp_splitter(tmp_path) is None
