"""Training workflows, adapter loading and numerical gradient contracts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.tests.test_models import tiny_config
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.lora import LoRaLayer
from mlx_vlm.trainer.lora_layers import LoRALinear
from mlx_vlm.trainer.sft_trainer import (
    TrainingArgs,
    iterate_batches,
    train,
    vision_language_loss_fn,
)
from mlx_vlm.trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
)
from mlx_vlm.utils import load

# Training workflows, adapter loading and numerical gradient contracts.


@pytest.fixture
def dataset_setup(monkeypatch):
    source = MagicMock()
    source.__getitem__.return_value = {
        "images": ["image1.jpg"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    template, prepare = Mock(return_value=""), Mock()
    monkeypatch.setattr("mlx_vlm.trainer.datasets.apply_chat_template", template)
    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", prepare)
    config, processor = {
        "model_type": "test_model",
        "image_token_index": 1,
    }, MagicMock()
    dataset = VisionDataset(source, config, processor)
    assert len(dataset) == len(source)
    assert dataset.config == config and dataset.processor == processor
    return dataset, template, prepare


def test_dataset_getitem(dataset_setup):
    dataset, _, prepare = dataset_setup
    expected = dict(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        image_sizes=[224, 224],
        pixel_values=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    )
    prepare.return_value = {key: mx.array(value) for key, value in expected.items()}
    prepare.return_value["image_grid_thw"] = (1, 1, 1)
    prepare.return_value["image_sizes"] = expected["image_sizes"]
    result = dataset[0]
    prepare.assert_called_once()
    assert result["image_grid_thw"] == (1, 1, 1)
    for key, value in expected.items():
        assert mx.array_equal(result[key], mx.array(value)).item(), key


def test_dataset_getitem_raises_when_image_token_keys_missing(dataset_setup):
    dataset, _, _ = dataset_setup
    del dataset.config["image_token_index"]
    with pytest.raises(ValueError, match="image_token_index"):
        dataset[0]


def test_dataset_adds_completion_mask_from_chat_template(dataset_setup):
    dataset, template, prepare = dataset_setup
    dataset.train_on_completions = True
    messages = [
        {"role": "user", "content": "Use the tool."},
        {"role": "assistant", "content": "tool call"},
    ]
    dataset.dataset.__getitem__.return_value = {"messages": messages}

    def render(_processor, _config, conversation, add_generation_prompt, **kwargs):
        assert conversation == (messages[:-1] if add_generation_prompt else messages)
        return "prefix" if add_generation_prompt else "full"

    template.side_effect = render
    prepare.side_effect = lambda **kwargs: (
        {"input_ids": mx.array([[1, 2]])}
        if kwargs["prompts"] == ["prefix"]
        else {
            "input_ids": mx.array([[1, 2, 3, 4]]),
            "attention_mask": mx.ones((1, 4), mx.int32),
        }
    )
    assert mx.array_equal(
        dataset[0]["completion_mask"], mx.array([[0, 0, 1, 1]])
    ).item()


@pytest.mark.parametrize(
    "completion", [False, True], ids=["variable_images", "completion_mask"]
)
def test_batch_collation(completion):
    dataset = []
    for row, ids in enumerate(([1, 2, 3], [4, 5])):
        item = dict(input_ids=mx.array(ids), attention_mask=mx.ones(len(ids), mx.int32))
        item["pixel_values"] = (
            None if completion else mx.full((row + 2, 4), row, mx.float32)
        )
        item.update(
            {"completion_mask": mx.array([0] + [1] * (len(ids) - 1))}
            if completion
            else {"image_grid_thw": mx.array([[1, 1, row + 2]])}
        )
        dataset.append(item)
    batch = next(iterate_batches(dataset, batch_size=2, max_seq_length=32))
    assert batch["input_ids"].shape == (2, 32)
    if completion:
        assert batch["completion_mask"].shape == (2, 32)
        for row, expected in enumerate(([0, 1, 1], [0, 1])):
            assert batch["completion_mask"][row, : len(expected)].tolist() == expected
    else:
        assert batch["pixel_values"].shape == (5, 4)
        assert batch["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


@pytest.mark.parametrize("counts,limit", [([10, 12], 8), ([3], 32)])
def test_image_token_truncation(counts, limit):
    """Reject truncation that would desynchronize image features and placeholders."""

    class ImageDataset(list):
        config = {"image_token_index": 99}

    dataset = ImageDataset(
        [
            dict(
                input_ids=mx.array([1] + [99] * count),
                attention_mask=mx.ones(count + 1, mx.int32),
                pixel_values=mx.zeros((2, 4)),
            )
            for count in counts
        ]
    )
    batches = iterate_batches(dataset, batch_size=1, max_seq_length=limit)
    if limit == 8:
        with pytest.raises(ValueError, match="No trainable examples"):
            next(batches)
    else:
        assert (next(batches)["input_ids"] == 99).sum().item() == 3


def test_call_accepts_mask_as_third_positional_argument():
    from mlx_vlm.models.idefics3.idefics3 import Model

    assert list(inspect.signature(Model.__call__).parameters)[1:4] == [
        "input_ids",
        "pixel_values",
        "mask",
    ]


@pytest.mark.parametrize("missing_adapter", [False, True])
def test_training_updates_and_saves(monkeypatch, missing_adapter):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.zeros((1,))

        def __call__(self, *args, **kwargs):
            return NS(logits=mx.zeros((4, 3, 10)))

    batch = dict(
        input_ids=mx.array([[1, 2, 3]] * 4),
        attention_mask=mx.ones((4, 3), mx.int32),
        pixel_values=mx.array([[[0.1, 0.2]]] * 4),
        labels=mx.array([[0, 1, 2]] * 4),
    )
    save, optimizer = Mock(), MagicMock(learning_rate=1e-4)
    monkeypatch.setattr("mlx_vlm.trainer.sft_trainer.mx.save_safetensors", save)
    monkeypatch.setattr(
        "mlx_vlm.trainer.sft_trainer.iterate_batches", Mock(return_value=iter([batch]))
    )
    options = dict(steps_per_save=1, adapter_file=None) if missing_adapter else {}
    result = train(
        Model(),
        optimizer,
        MagicMock(__len__=lambda self: 4),
        None,
        TrainingArgs(iters=1, batch_size=4, **options),
    )
    assert result is None
    optimizer.update.assert_called()
    save.assert_called()
    if missing_adapter:
        assert [call.args[0] for call in save.call_args_list] == [
            "adapters.safetensors",
            "0000001_adapters.safetensors",
            "adapters.safetensors",
        ]


def test_completion_mask_is_used_without_passing_to_model():
    class CaptureModel:
        model_type = "test_model"

        def __call__(self, input_ids, pixel_values, mask, **kwargs):
            self.mask, self.kwargs = mask, kwargs
            return NS(logits=mx.zeros((*input_ids.shape, 10)))

    model = CaptureModel()
    batch = dict(
        input_ids=mx.array([[1, 2, 3, 4]]),
        attention_mask=mx.ones((1, 4), mx.int32),
        completion_mask=mx.array([[0, 0, 1, 1]]),
        pixel_values=None,
    )
    vision_language_loss_fn(model, batch, train_on_completions=True, assistant_id=999)
    assert model.mask.tolist() == [[1, 1, 1]]
    assert "completion_mask" not in model.kwargs


def test_b_zero_init_gives_no_lora_contribution():
    linear = nn.Linear(4, 4)
    lora = LoRaLayer(linear, rank=8, alpha=16.0, dropout=0.0)
    inputs = mx.ones((1, 4))
    assert mx.allclose(linear(inputs), lora(inputs)).item()


def adapter_model(**layers):
    model = nn.Module()
    model.language_model, model.config = nn.Module(), NS()
    for name, layer in layers.items():
        setattr(model.language_model, name, layer)
    return model


def test_get_peft_model(monkeypatch):
    freeze, report = Mock(), Mock()
    monkeypatch.setattr("mlx_vlm.trainer.utils.freeze_model", freeze)
    monkeypatch.setattr("mlx_vlm.trainer.utils.print_trainable_parameters", report)
    model = adapter_model(
        layer1=nn.Linear(256, 512), layer2=nn.QuantizedLinear(256, 512, 8)
    )
    assert get_peft_model(model, ["layer1", "layer2"]) is model
    freeze.assert_called()
    report.assert_called()
    assert all(
        isinstance(getattr(model.language_model, name), LoRALinear)
        for name in ("layer1", "layer2")
    )
    config = model.config.lora
    assert (config["fine_tune_type"], config["num_layers"]) == ("lora", -1)
    assert config["lora_parameters"]["rank"] == 10
    assert config["lora_parameters"]["scale"] == pytest.approx(0.01, abs=5e-8, rel=0)
    assert set(config["lora_parameters"]["keys"]) == {
        "language_model.layer1",
        "language_model.layer2",
    }


def test_find_all_linear_names():
    model = MagicMock()
    model.named_modules.return_value = [
        (
            name,
            (
                nn.QuantizedLinear(256, 512, 8)
                if name == "layer2"
                else nn.Linear(256, 512)
            ),
        )
        for name in ("layer1", "layer2", "mm_projector", "lm_head")
    ]
    assert set(find_all_linear_names(model)) == {"layer1", "layer2"}


@pytest.mark.parametrize(
    "legacy", [True, False], ids=["legacy_schema", "native_schema"]
)
def test_apply_lora_layers(tmp_path, monkeypatch, legacy):
    config = (
        dict(rank=4, alpha=8, dropout=0.0)
        if legacy
        else dict(
            fine_tune_type="lora",
            num_layers=-1,
            lora_parameters=dict(
                rank=4, dropout=0.0, scale=2.0, keys=["language_model.proj"]
            ),
        )
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    weights = tmp_path / "adapters.safetensors"
    weights.touch()
    model = MagicMock() if legacy else adapter_model(proj=nn.Linear(8, 8))
    model.load_weights = Mock()
    if legacy:
        model.language_model.named_modules.return_value = []
        peft = Mock(return_value=model)
        monkeypatch.setattr("mlx_vlm.trainer.utils.get_peft_model", peft)
    assert apply_lora_layers(model, str(tmp_path)) is model
    model.load_weights.assert_called_once_with(str(weights), strict=False)
    if legacy:
        peft.assert_called_once_with(
            model, [], rank=4, alpha=8, dropout=0.0, legacy=True
        )
    else:
        assert isinstance(model.language_model.proj, LoRALinear)


def test_chunked_update_allows_value_and_grad():
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_chunked

    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 4, 1, 4)) for _ in range(3))
    g, beta, state = (
        mx.full((1, 4, 1), 0.9),
        mx.full((1, 4, 1), 0.5),
        mx.zeros((1, 1, 4, 4)),
    )

    def loss(q):
        output, state_out = gated_delta_chunked(q, k, v, g, beta, state, C=2)
        return output.astype(mx.float32).sum() + state_out.astype(mx.float32).sum()

    value, grad = mx.value_and_grad(loss)(q)
    mx.eval(value, grad)
    assert mx.isfinite(value).item() and mx.all(mx.isfinite(grad)).item()
    assert grad.shape == q.shape


@pytest.mark.parametrize(
    "kind,style,pos_ndim",
    [("fused", style, ndim) for style in ("interleaved", "chunked") for ndim in (2, 3)]
    + [("even_odd", style, None) for style in ("half", "full")]
    + [
        ("sectioned", style, None)
        for style in ("sectioned_half_split", "sectioned_even_odd")
    ],
)
def test_rotary_gradients_match_pure_mlx(monkeypatch, kind, style, pos_ndim):
    """Both input gradients must match the pure-MLX reference for every rotary layout."""
    from mlx_vlm.models import rope_utils

    mx.random.seed(1)
    angle3, angle4 = mx.random.normal((1, 8, 64)), mx.random.normal((3, 1, 8, 64))
    cos, sin = (
        (mx.cos(angle3), mx.sin(angle3))
        if kind == "even_odd"
        else (mx.cos(angle4), mx.sin(angle4))
    )
    positions = mx.arange(8, dtype=mx.int32)[None]
    if pos_ndim == 3:
        positions = mx.broadcast_to(positions[None], (3, 1, 8))

    def loss(q, k):
        if kind == "fused":
            outputs = rope_utils.MRoPERotaryEmbedding(dim=64, style=style).apply_rotary(
                q, k, positions
            )
        elif kind == "even_odd":
            outputs = rope_utils.apply_rotary_pos_emb_even_odd(
                q, k, cos, sin, cos_layout=style
            )
        else:
            outputs = rope_utils.apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, mrope_section=[4, 6, 6], style=style
            )
        return sum(output.astype(mx.float32).sum() for output in outputs)

    mx.random.seed(0)
    q, k = (mx.random.normal((1, 2, 8, 64)) for _ in range(2))
    grads = mx.value_and_grad(loss, argnums=(0, 1))
    _, actual = grads(q, k)
    mx.eval(actual)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(rope_utils, "_HAS_METAL", False)
            rope_utils._compiled_rotary_apply.cache_clear()
            _, expected = grads(q, k)
            mx.eval(expected)
    finally:
        rope_utils._compiled_rotary_apply.cache_clear()
    for gradient, reference in zip(actual, expected):
        assert mx.all(mx.isfinite(gradient)).item()
        assert mx.allclose(gradient, reference, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("family", ["glm", "deepseek"])
def test_shared_moe_preserves_expert_gradients_and_unweighted_replacements(family):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4MoE
    from mlx_vlm.models.glm5_next.language import Glm5NextMoE
    from mlx_vlm.models.switch_layers import SwitchGLU

    config = tiny_config(family)
    if family == "glm":
        module = Glm5NextMoE(config)
        kwargs = {}
    else:
        module = DeepseekV4MoE(config, 0)
        kwargs = dict(input_ids=mx.array([[1, 2, 3]]))
    inputs = mx.random.normal((1, 3, config.hidden_size))
    module.gate.freeze()
    value, grad = nn.value_and_grad(module, lambda m: m(inputs, **kwargs).sum())(module)
    mx.eval(value, grad)
    assert mx.isfinite(value).item()
    module.eval()
    original = module.switch_mlp
    expected = module(inputs, **kwargs)

    class ExternalExperts(nn.Module):
        def __call__(self, x, indices):
            return original(x, indices)

    module.switch_mlp = ExternalExperts()
    actual = module(inputs, **kwargs)
    assert mx.allclose(actual, expected, atol=1e-5).item()
    assert not isinstance(module.switch_mlp, SwitchGLU)


# Loading and utility contracts


def test_load_delegates_adapter_loading_to_trainer_entrypoint():
    model = MagicMock()
    adapted_model = MagicMock()
    processor = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path", return_value=Path("/tmp/model")),
        patch("mlx_vlm.utils.load_model", return_value=model),
        patch("mlx_vlm.utils.apply_lora_layers", return_value=adapted_model) as apply,
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
        patch("mlx_vlm.utils.load_processor", return_value=processor),
    ):
        result_model, result_processor = load("model-id", adapter_path="adapter-dir")

    apply.assert_called_once_with(model, "adapter-dir")
    adapted_model.eval.assert_called_once()
    assert result_model is adapted_model
    assert result_processor is processor
