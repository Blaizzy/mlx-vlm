"""Native MTP SFT alignment, masking, gradients, and checkpoint round trips."""

from dataclasses import asdict
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.trainer import (
    Trainer,
    VLMTrainingArgs,
    attach_mtp_head,
    prepare_model_for_training,
)
from mlx_vlm.trainer.common.utils import save_trainable_weights
from mlx_vlm.trainer.losses.cross_entropy import cross_entropy
from mlx_vlm.trainer.vlm.sft.runtime import vlm_sft_loss


def tiny_model(moe=False, tied=False):
    mx.random.seed(0)
    config = TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=2,
        max_position_embeddings=64,
        head_dim=8,
        full_attention_interval=1,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 1, 2],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    config.tie_word_embeddings = tied
    language_model = LanguageModel
    if moe:
        from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig
        from mlx_vlm.models.qwen3_5_moe.language import (
            LanguageModel as MoeLanguageModel,
        )

        values = asdict(config)
        values.pop("intermediate_size")
        config = MoeTextConfig(
            **values,
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=16,
            moe_intermediate_size=16,
        )
        language_model = MoeLanguageModel

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="qwen3_5_moe" if moe else "qwen3_5"
            )
            self.language_model = language_model(config)

        def __call__(self, ids, pixels=None, mask=None, **kwargs):
            positions = mx.broadcast_to(
                mx.arange(ids.shape[1])[None, None, :], (3, ids.shape[0], ids.shape[1])
            )
            return self.language_model(ids, position_ids=positions, **kwargs)

    model = Model()
    head = Qwen3_5MTPDraftModel(Qwen3_5MTPConfig(text_config=config))
    return attach_mtp_head(model, head)


def batch():
    return {
        "input_ids": mx.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
        "attention_mask": mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "completion_mask": mx.array([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]]),
    }


@pytest.mark.parametrize("moe,tied", [(False, False), (False, True), (True, False)])
def test_mtp_uses_next_next_targets_and_completion_padding_masks(moe, tied):
    model = tiny_model(moe=moe, tied=tied)
    data = batch()
    loss, metrics = vlm_sft_loss(
        model, data, train_on_completions=True, mtp_loss_weight=0.3
    )
    output = model(data["input_ids"][:, :-1], return_hidden=True)
    logits = model.mtp_head.training_logits(
        data["input_ids"][:, 1:-1],
        output.hidden_states[-1][:, :-1],
        mx.array([[1, 1, 1], [1, 1, 0]]),
        model,
    )
    expected, _ = cross_entropy(
        logits, mx.array([[3, 4, 0], [7, 0, 0]]), mx.array([[1, 1, 0], [1, 0, 0]])
    )
    np.testing.assert_allclose(metrics["mtp_loss"].item(), expected.item(), rtol=1e-5)
    np.testing.assert_allclose(
        loss.item(), (metrics["nll"] + 0.3 * expected).item(), rtol=1e-5
    )
    assert metrics["num_mtp_tokens"].item() == 3
    assert metrics["num_tokens"].item() == 3


def test_mtp_gradients_reach_native_head_and_base_adapters(tmp_path):
    model = prepare_model_for_training(tiny_model(), lora_rank=2, verbose=False)
    loss_fn = lambda m, b: vlm_sft_loss(m, b, mtp_loss_weight=0.3)
    (_, _), gradients = nn.value_and_grad(model, loss_fn)(model, batch())
    flat = dict(tree_flatten(gradients))
    assert mx.abs(flat["mtp_head.fc.weight"]).sum().item() > 0
    assert any(
        mx.abs(g).sum().item() > 0 for k, g in flat.items() if k.endswith("lora_b")
    )
    path = tmp_path / "weights.safetensors"
    save_trainable_weights(model, path)
    saved = mx.load(str(path))
    assert "mtp_head.fc.weight" in saved
    restored = prepare_model_for_training(
        tiny_model(), checkpoint_path=path, verbose=False
    )
    np.testing.assert_allclose(
        loss_fn(model, batch())[0].item(),
        loss_fn(restored, batch())[0].item(),
        rtol=1e-5,
    )


def test_disabled_mtp_does_not_run_head_or_request_hidden():
    model = tiny_model()

    class UnusedHead(nn.Module):
        def training_logits(self, *args):
            raise AssertionError("disabled head called")

    model.mtp_head = UnusedHead()
    loss, metrics = vlm_sft_loss(model, batch())
    assert np.isfinite(loss.item())
    assert "mtp_loss" not in metrics


def test_short_and_unsupervised_batches_have_zero_auxiliary_loss():
    model = tiny_model()
    for data in (
        {"input_ids": mx.array([[1, 2]]), "attention_mask": mx.ones((1, 2))},
        {**batch(), "attention_mask": mx.zeros((2, 5))},
    ):
        loss, metrics = vlm_sft_loss(model, data, mtp_loss_weight=0.3)
        assert np.isfinite(loss.item())
        assert metrics["mtp_loss"].item() == 0
        assert metrics["num_mtp_tokens"].item() == 0


def test_configuration_and_missing_head_fail_early():
    for weight in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mtp_loss_weight"):
            VLMTrainingArgs(mtp_loss_weight=weight).validate()
    model = tiny_model()
    del model["mtp_head"]
    with pytest.raises(ValueError, match="attach_mtp_head"):
        Trainer(model, VLMTrainingArgs(mtp_loss_weight=0.3), prepared=True)


def test_legacy_loss_uses_same_mtp_objective():
    from mlx_vlm.trainer.vlm.sft.trainer import vision_language_loss_fn

    model = tiny_model()
    expected, _ = vlm_sft_loss(model, batch(), mtp_loss_weight=0.3)
    actual, metrics = vision_language_loss_fn(model, batch(), mtp_loss_weight=0.3)
    np.testing.assert_allclose(actual.item(), expected.item(), rtol=1e-5)
    assert metrics["num_mtp_tokens"].item() == 3


def test_base_quantization_preserves_floating_point_mtp_head():
    model = prepare_model_for_training(
        tiny_model(),
        lora_rank=2,
        quantization_bits=4,
        quantization_group_size=32,
        verbose=False,
    )
    assert any(
        isinstance(layer, nn.QuantizedLinear)
        for _, layer in model.language_model.named_modules()
    )
    assert isinstance(model.mtp_head.fc, nn.Linear)
    assert not isinstance(model.mtp_head.fc, nn.QuantizedLinear)
    assert "fc" in model.mtp_head.trainable_parameters()


def test_cli_exposes_mtp_options():
    from mlx_vlm.train import parse_args

    args, settings = parse_args(
        ["--model", "target", "--mtp-model", "head", "--mtp-loss-weight", "0.3"]
    )
    assert args.mtp_model == "head"
    assert settings.mtp_loss_weight == 0.3


def test_native_head_is_causal_and_has_no_inference_cache_side_effects():
    model = tiny_model()
    hidden = mx.random.normal((1, 3, 16))
    tokens = mx.array([[1, 2, 3]])
    logits = model.mtp_head.training_logits(tokens, hidden, mx.ones((1, 3)), model)
    changed = model.mtp_head.training_logits(
        mx.array([[1, 2, 9]]), hidden, mx.ones((1, 3)), model
    )
    np.testing.assert_allclose(
        np.array(logits[:, :2]), np.array(changed[:, :2]), rtol=1e-5
    )
    assert model.mtp_head._cache == []
    assert model.mtp_head._input_embed is None


def test_trainer_updates_and_evaluates_auxiliary_objective(tmp_path):
    import mlx.optimizers as optim

    model = prepare_model_for_training(tiny_model(), lora_rank=2, verbose=False)
    before = mx.array(model.mtp_head.fc.weight)
    mx.eval(before)
    rows = [{key: value[i] for key, value in batch().items()} for i in range(2)]
    trainer = Trainer(
        model,
        VLMTrainingArgs(
            mtp_loss_weight=0.3,
            iters=1,
            batch_size=2,
            pad_to_multiple=1,
            val_batches=1,
            steps_per_report=1,
            adapter_file=str(tmp_path / "weights.safetensors"),
        ),
        optimizer=optim.Adam(learning_rate=0.001),
        train_dataset=rows,
        eval_dataset=rows,
        prepared=True,
    )
    trainer.train()
    metrics = trainer.evaluate()
    assert metrics["val_num_mtp_tokens"] == 3
    assert metrics["val_mtp_loss"] > 0
    assert mx.abs(model.mtp_head.fc.weight - before).sum().item() > 0
