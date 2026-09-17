"""JSON-driven model contracts and cached-image support declarations."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import math
import unittest
from math import prod
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.utils import get_model_and_args


class ModelChecks(unittest.TestCase):
    """Reusable assertions; each JSON case constructs fresh configs and models."""

    def forward_cache_test_runner(self, model, vocab_size):
        model.eval()
        mx.eval(model.parameters())
        ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
        assert model(ids).logits.shape == (1, 8, vocab_size)
        cache = model.language_model.make_cache()
        model(ids[:, :-1], cache=cache)
        assert model(ids[:, -1:], cache=cache).logits.shape == (1, 1, vocab_size)

    def _check_returns_input_embeddings_features(self, model, model_name):
        """Helper to test get_input_embeddings returns InputEmbeddingsFeatures."""

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids=input_ids)
        self.assertIsInstance(
            result,
            InputEmbeddingsFeatures,
            f"{model_name}: expected InputEmbeddingsFeatures, got {type(result).__name__}",
        )
        self.assertIsNotNone(result.inputs_embeds)

    def _assert_qwen_request_owned_mrope_kwargs(self, model):
        stale_position_ids = mx.array([[[9, 9, 9]]], dtype=mx.int32)
        stale_rope_deltas = mx.array([[99]], dtype=mx.int32)
        model.language_model._position_ids = stale_position_ids
        model.language_model._rope_deltas = stale_rope_deltas

        result = model.get_input_embeddings(
            input_ids=mx.array([[1, 2, 3]], dtype=mx.int32)
        )

        self.assertIsNotNone(result.position_ids)
        self.assertIsNotNone(result.rope_deltas)
        self.assertEqual(result.position_ids.shape, (1, 3))
        self.assertEqual(result.position_ids.tolist(), [[0, 1, 2]])
        self.assertEqual(result.rope_deltas.tolist(), [[0]])
        self.assertTrue(
            mx.array_equal(
                model.language_model._position_ids, stale_position_ids
            ).item()
        )
        self.assertTrue(
            mx.array_equal(model.language_model._rope_deltas, stale_rope_deltas).item()
        )

    def _assert_qwen_chunked_prefill_slices_mrope_position_ids(self, model):
        language_model = model.language_model
        hidden_size = language_model.args.hidden_size
        captured = {}

        class _CapturingModel:
            fa_idx = 0

            class _Embed:
                @staticmethod
                def as_linear(x):
                    return x

            embed_tokens = _Embed()

            def __call__(self, inputs, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], hidden_size))

        class _StubCache:
            _idx = 2
            offset = mx.array(2)

        language_model.model = _CapturingModel()
        language_model.lm_head = lambda x: x

        full_position_ids = mx.arange(15, dtype=mx.int32).reshape(3, 1, 5)
        language_model(
            mx.array([[7, 8]], dtype=mx.int32),
            inputs_embeds=mx.zeros((1, 2, hidden_size), dtype=mx.float32),
            cache=[_StubCache()],
            position_ids=full_position_ids,
        )

        self.assertEqual(captured["position_ids"].shape, (3, 1, 2))
        self.assertEqual(
            captured["position_ids"].tolist(), full_position_ids[:, :, 2:4].tolist()
        )

    def language_test_runner(self, model, config, *, num_layers=None):
        model_type = config.model_type
        vocab_size = config.vocab_size
        if num_layers is None:
            num_layers = config.num_hidden_layers
        self.assertEqual(model.model_type, model_type)
        self.assertEqual(len(model.layers), num_layers)

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 2, vocab_size))
            self.assertEqual(logits.dtype, t)

            outputs = model(mx.argmax(logits[0, -1:, :], keepdims=True), cache=None)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 1, vocab_size))
            self.assertEqual(logits.dtype, t)

    def mm_projector_test_runner(
        self,
        mm_projector,
        vision_hidden_size,
        text_hidden_size,
        *,
        grid_hw=None,
        downsample_ratio=1,
    ):
        batch_size = math.prod(grid_hw) if grid_hw else 1
        output_tokens = (
            math.prod(math.ceil(n / downsample_ratio) for n in grid_hw)
            if grid_hw
            else 1
        )
        kwargs = dict(zip(("n_h", "n_w"), grid_hw)) if grid_hw else {}

        for t in [mx.float32, mx.float16]:
            mm_projector.update(
                tree_map(lambda p: p.astype(t), mm_projector.parameters())
            )

            vision_features = mx.random.uniform(
                shape=(batch_size, vision_hidden_size), dtype=t
            )
            input_tensor = mx.array(vision_features)

            outputs = mm_projector(input_tensor, **kwargs)
            self.assertEqual(outputs.shape, (output_tokens, text_hidden_size))
            self.assertEqual(outputs.dtype, t)

    def vision_test_runner(
        self,
        vision_tower,
        model_type,
        vision_hidden_size,
        num_channels,
        image_size: tuple,
        vision_feature_layer=-2,
        channel_first=False,
        **kwargs,
    ):
        if model_type == "llama4_vision_model":
            vision_hidden_size = kwargs.pop("projector_output_dim", vision_hidden_size)
        batch_size = kwargs.pop("batch_size", 1)

        for t in [mx.float32, mx.float16]:
            vision_tower.update(
                tree_map(lambda p: p.astype(t), vision_tower.parameters())
            )
            if model_type is not None:
                self.assertEqual(vision_tower.model_type, model_type)

            if len(image_size) > 2:
                input_tensor = mx.random.uniform(shape=image_size)
            elif model_type in [
                "qwen2_5_vl",
                "qwen3_5",
                "qwen3_5_moe",
                "qwen4_exp",
                "glm4v_moe",
                "glm4v",
                "hunyuan_vl",
                "siglip2_vision_model",
            ]:
                input_tensor = mx.random.uniform(shape=(image_size[0], image_size[1]))
            else:
                shape = (
                    (batch_size, num_channels, image_size[0], image_size[1])
                    if channel_first
                    else (batch_size, image_size[0], image_size[1], num_channels)
                )
                input_tensor = mx.random.uniform(shape=shape)

            if "image_masks" in inspect.signature(vision_tower.__call__).parameters:
                input_tensor = input_tensor.transpose(0, 3, 1, 2)
                image_masks = mx.ones((batch_size, num_channels, image_size[0]))
                kwargs["image_masks"] = image_masks

            input_tensor = input_tensor.astype(t)

            if (
                "output_hidden_states"
                in inspect.signature(vision_tower.__call__).parameters
            ):
                hidden_states = vision_tower(
                    input_tensor, output_hidden_states=True, **kwargs
                )
            else:
                hidden_states = vision_tower(input_tensor, **kwargs)

            if vision_feature_layer is not None:
                hidden_states = hidden_states[vision_feature_layer]

            # Check vision hidden feature layer's shape matches the expected hidden size
            if channel_first:
                self.assertEqual(hidden_states.shape[1], vision_hidden_size)
            else:
                self.assertEqual(hidden_states.shape[-1], vision_hidden_size)

            self.assertEqual(hidden_states.dtype, t)

    def _assert_audio_features(self, features, shape, dtype):
        mx.eval(features)
        self.assertEqual(features.shape, shape)
        self.assertEqual(features.dtype, dtype)
        self.assertTrue(mx.all(mx.isfinite(features)).item())

    def audio_test_runner(self, model, config, model_name, *, frames=32, lengths=None):
        if model_name not in {"inkling", "gemma3n", "gemma4", "gemma4_unified"}:
            raise ValueError(f"Unsupported audio model: {model_name}")
        audio_config = config.audio_config
        text_width = config.text_config.hidden_size
        lengths = [frames, frames // 2] if lengths is None else lengths
        self.assertTrue(lengths and all(0 <= n <= frames for n in lengths))
        batch = len(lengths)
        valid_mask = mx.arange(frames)[None, :] < mx.array(lengths)[:, None]
        components = [
            getattr(model, name, None) for name in ("audio_tower", "embed_audio")
        ]
        for dtype in (mx.float32, mx.float16):
            for component in components:
                if component is not None:
                    component.eval()
                    component.update(
                        tree_map(lambda p: p.astype(dtype), component.parameters())
                    )

            if model_name == "inkling":
                bins = audio_config.n_mel_bins
                inputs = mx.arange(batch * frames * bins, dtype=mx.int32)
                inputs = (
                    inputs.reshape(batch, frames, bins) % audio_config.mel_vocab_size
                )
                features = model.audio_tower(inputs)
                shape, output_dtype = (batch, frames, text_width), dtype
            elif model_name == "gemma4_unified":
                inputs = mx.random.normal(
                    (batch, frames, audio_config.output_proj_dims)
                ).astype(dtype)
                projected = model.get_audio_features(inputs)
                self._assert_audio_features(
                    projected, (batch * frames, text_width), dtype
                )
                features = model.get_audio_features(inputs, valid_mask)
                shape, output_dtype = (sum(lengths), text_width), dtype
            else:
                is_gemma3n = model_name == "gemma3n"
                # Gemma 4's subsampling projection requires 128 input mel bins.
                bins = audio_config.input_feat_size if is_gemma3n else 128
                inputs = mx.random.normal((batch, frames, bins)).astype(dtype)
                encoded, mask = model.audio_tower(inputs, ~valid_mask)
                stride = (
                    math.prod(s[0] for s in audio_config.sscp_conv_stride_size)
                    * max(1, audio_config.conf_reduction_factor)
                    if is_gemma3n
                    else 4
                )
                steps = (frames + stride - 1) // stride
                width = (
                    getattr(audio_config, "output_proj_dims", None)
                    or audio_config.hidden_size
                )
                # Both Gemma encoders currently promote float16 inputs to float32.
                output_dtype = mx.float32
                self._assert_audio_features(
                    encoded, (batch, steps, width), output_dtype
                )
                self.assertEqual(mask.shape, (batch, steps))
                self.assertEqual(mask.dtype, mx.bool_)
                self.assertTrue(mx.array_equal(mask, ~valid_mask[:, ::stride]).item())
                self.assertTrue(
                    mx.all(mx.where(mask[..., None], encoded == 0, True)).item()
                )
                features = (
                    model.embed_audio(inputs_embeds=encoded)
                    if is_gemma3n
                    else model.embed_audio(encoded)
                )
                shape = (batch, steps, text_width)
            self._assert_audio_features(features, shape, output_dtype)

    def _assert_mrope_decode_uses_cache_idx(self, language_model, hidden_size):
        """Shared assertion: MRoPE decode-step reads RoPE position from
        ``cache[0]._idx`` (Python int) rather than ``cache[0].offset.item()``
        — the latter forces a per-step GPU sync. Regression guard for the
        cache._idx refactor in PR #1055.
        """
        # Skip the prefill branch: pretend deltas have already been computed.
        language_model._rope_deltas = mx.array([[0]])
        language_model._position_ids = None

        captured = {}

        class _CapturingModel:
            """Stand-in for the inner Qwen text model — captures position_ids
            and exposes ``embed_tokens.as_linear`` so the tied-weights branch
            in ``LanguageModel.__call__`` doesn't crash.
            """

            class _Embed:
                @staticmethod
                def as_linear(x):
                    return x

            embed_tokens = _Embed()

            def __call__(self, inputs, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], hidden_size))

        language_model.model = _CapturingModel()
        language_model.lm_head = lambda x: x  # bypass the real linear (untied path)

        class _StubCacheWithIdx:
            """``_idx`` (Python int) deliberately differs from ``offset``. If
            extraction reads ``offset.item()`` the captured position is 3;
            reading ``_idx`` gives 10. ``offset`` is 0-d so the per-sequence
            ``cache_offsets`` / ``cache_offset_array`` branch is skipped
            uniformly across qwen2_vl, qwen2_5_vl, and qwen3_vl.
            """

            def __init__(self):
                self._idx = 10
                self.offset = mx.array(3)  # 0-d -> never the per-seq path

        language_model(mx.array([[5]]), cache=[_StubCacheWithIdx()])

        position_ids = captured["position_ids"]
        self.assertIsNotNone(position_ids)
        self.assertIn(tuple(position_ids.shape), {(1, 1), (3, 1, 1)})
        # Decode position == cache._idx (10), not cache.offset[0].item() (3).
        if position_ids.ndim == 3:
            self.assertEqual(position_ids[0, 0, 0].item(), 10)
        else:
            self.assertEqual(position_ids[0, 0].item(), 10)

    def _assert_mrope_decode_uses_rope_deltas_kwarg(self, language_model, hidden_size):
        """Shared assertion: under continuous batching, an explicit
        ``rope_deltas`` kwarg passed by ``GenerationBatch._step()`` must
        override the mutable ``language_model._rope_deltas`` attribute. The
        latter can be clobbered mid-decode when a newer request's prefill
        runs ``get_input_embeddings`` on the same GPU thread.
        """
        # Stale per-model state — simulates a newer request having just
        # prefilled and overwritten ``_rope_deltas``.
        language_model._rope_deltas = mx.array([[99]])
        language_model._position_ids = None

        captured = {}

        class _CapturingModel:
            class _Embed:
                @staticmethod
                def as_linear(x):
                    return x

            embed_tokens = _Embed()
            # ``fa_idx`` lets the qwen3_5 / qwen3_5_moe cache-indexing path
            # (``cache[self.model.fa_idx]``) resolve to the stub cache below.
            fa_idx = 0

            def __call__(self, inputs, position_ids=None, **kwargs):
                captured["position_ids"] = position_ids
                return mx.zeros((inputs.shape[0], inputs.shape[1], hidden_size))

        language_model.model = _CapturingModel()
        language_model.lm_head = lambda x: x

        class _StubCacheWithIdx:
            def __init__(self):
                self._idx = 10
                self.offset = mx.array(3)  # 0-d -> scalar decode branch

        # Caller-supplied kwarg (the row-local delta from ``GenerationBatch``)
        # disagrees with the stale ``_rope_deltas`` (99). Position must
        # follow the kwarg.
        kwarg_delta = mx.array([[5]])
        language_model(
            mx.array([[7]]), cache=[_StubCacheWithIdx()], rope_deltas=kwarg_delta
        )

        position_ids = captured["position_ids"]
        self.assertIsNotNone(position_ids)
        self.assertEqual(tuple(position_ids.shape), (3, 1, 1))
        # Position == cache._idx (10) + kwarg delta (5) == 15.
        # Pre-fix behavior would have read self._rope_deltas (99) -> 109.
        self.assertEqual(position_ids[0, 0, 0].item(), 15)


CHECKS = {
    "forward_cache": "forward_cache_test_runner",
    "language": "language_test_runner",
    "projector": "mm_projector_test_runner",
    "vision": "vision_test_runner",
    "audio": "audio_test_runner",
    "input_embeddings": "_check_returns_input_embeddings_features",
    "mrope_cache_index": "_assert_mrope_decode_uses_cache_idx",
    "mrope_deltas": "_assert_mrope_decode_uses_rope_deltas_kwarg",
    "request_positions": "_assert_qwen_request_owned_mrope_kwargs",
    "chunked_positions": "_assert_qwen_chunked_prefill_slices_mrope_position_ids",
}
CONFIG_TYPES = {
    "text_config": "TextConfig",
    "vision_config": "VisionConfig",
    "projector_config": "ProjectorConfig",
    "perceiver_config": "PerceiverConfig",
    "audio_config": "AudioConfig",
    "thinker_config": "ThinkerConfig",
    "talker_config": "TalkerConfig",
    "code_predictor_config": "CodePredictorConfig",
    "code2wav_config": "Code2WavConfig",
    "vit_config": "config.VitConfig",
    "adapter_config": "config.AdapterConfig",
}
DATA = json.loads(
    Path(__file__).with_name("model_cases.json").read_text(encoding="utf-8")
)
if DATA["version"] != 2:
    raise ValueError(f"Unsupported model_cases.json version: {DATA['version']}")


def build_config(module, values, config_type="ModelConfig"):
    """Construct the model family's config classes from ordinary nested data."""
    fields = copy.deepcopy(values)
    for name, value in fields.items():
        if name in CONFIG_TYPES and isinstance(value, dict):
            fields[name] = build_config(module, value, CONFIG_TYPES[name])
    return attrgetter(config_type)(module)(**fields)


def first_attribute(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def check_arguments(kind, case, model, config):
    """Keep shared component selection and dimension wiring in Python."""
    if kind == "forward_cache":
        return (model, getattr(config, "text_config", config).vocab_size), {}
    if kind == "input_embeddings":
        return (model, case["module"]), {}
    if kind == "audio":
        return (model, config, case["module"]), case.get("audio", {})
    if kind in {"request_positions", "chunked_positions"}:
        return (model,), {}
    if kind in {"language", "mrope_cache_index", "mrope_deltas"}:
        language_model = model.language_model
        text_config = getattr(config, "text_config", config)
        # Phi3-V keeps its language dimensions on the top-level config.
        if case["module"] == "phi3_v":
            text_config = config
        if kind == "language":
            options = {}
            if not hasattr(text_config, "num_hidden_layers"):
                options["num_layers"] = text_config.n_layers
            return (language_model, text_config), options
        return (language_model, text_config.hidden_size), {}
    if kind == "projector":
        projector = attrgetter(case.get("projector_path", "multi_modal_projector"))(
            model
        )
        if case["module"] == "deepseek_v4":
            return (projector, config.vision_dim, config.hidden_size), {
                "grid_hw": case["vision"]["grid_hw"],
                "downsample_ratio": config.vision_downsample_ratio,
            }
        return (
            projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        ), {}
    if kind == "vision":
        vision = attrgetter(case.get("vision_path", "vision_tower"))(model)
        options = case.get("vision", {})
        if case["module"] == "deepseek_v4":
            return (
                vision,
                None,
                config.vision_dim,
                3,
                tuple(options["input_shape"]),
            ), {
                "vision_feature_layer": options["feature_layer"],
                "n_h": options["grid_hw"][0],
                "n_w": options["grid_hw"][1],
            }
        vision_config = config.vision_config
        image_size = options.get("input_shape")
        if image_size is None:
            image_size = (vision_config.image_size, vision_config.image_size)
        hidden_size = first_attribute(
            vision_config,
            "out_hidden_size",
            "hidden_size",
            "d_model",
            "width",
            "text_hidden_size",
        )
        # Molmo's hidden_size is the projector intermediate width.
        if case["module"] == "molmo":
            hidden_size = vision_config.d_model
        channels = first_attribute(vision_config, "num_channels", "in_channels")
        kwargs = {}
        if "feature_layer" in options:
            kwargs["vision_feature_layer"] = options["feature_layer"]
        if "channel_first" in options:
            kwargs["channel_first"] = options["channel_first"]
        if "grid_thw" in options:
            kwargs["grid_thw"] = mx.array(
                options["grid_thw"],
                dtype=getattr(mx, options.get("grid_dtype", "int64")),
            )
        if vision_config.model_type == "llama4_vision_model":
            kwargs["projector_output_dim"] = vision_config.projector_output_dim
        return (
            vision,
            # MiniMax's vision wrapper does not expose model_type.
            None if case["module"] == "minimax_m3_vl" else vision_config.model_type,
            hidden_size,
            channels,
            tuple(image_size),
        ), kwargs
    raise ValueError(f"Unknown model check: {kind}")


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_model_contract(case):
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    config = build_config(module, case["config"])
    model = module.Model(config)
    checks = ModelChecks()
    for kind in case["checks"]:
        args, kwargs = check_arguments(kind, case, model, config)
        getattr(checks, CHECKS[kind])(*args, **kwargs)


@pytest.mark.parametrize("name", DATA["dense"])
def test_dense_model(name):
    module = importlib.import_module("mlx_vlm.models." + name)
    config = DATA["dense"][name]
    model = module.Model(module.ModelConfig.from_dict(copy.deepcopy(config)))
    ModelChecks().forward_cache_test_runner(model, config["vocab_size"])


@pytest.mark.parametrize(
    "model_module",
    [
        "llava.llava",
        "llava_bunny.llava_bunny",
        "llava_next.llava_next",
        "gemma3.gemma3",
        "gemma4.gemma4",
        "paligemma.paligemma",
        "qwen2_5_vl.qwen2_5_vl",
        "qwen2_vl.qwen2_vl",
        "qwen3_vl.qwen3_vl",
        "qwen3_5.qwen3_5",
        "qwen3_vl_moe.qwen3_vl_moe",
        "internvl_chat.internvl_chat",
        "mistral3.mistral3",
        "pixtral.pixtral",
        "aya_vision.aya_vision",
        "fastvlm.fastvlm",
        "glm4v.glm4v",
        "glm4v_moe.glm4v_moe",
        "glm_ocr.glm_ocr",
        "kimi_vl.kimi_vl",
        "dots_ocr.dots_ocr",
        "hunyuan_vl.hunyuan_vl",
        "paddleocr_vl.paddleocr_vl",
        "ernie4_5_moe_vl.ernie4_5_moe_vl",
        "mllama.mllama",
        "granite_vision.granite_vision",
        "granite4_vision.granite4_vision",
        "deepseek_vl_v2.deepseek_vl_v2",
        "deepseek_v4.deepseek_v4",
        "multi_modality.multi_modality",
        "lfm2_vl.lfm2_vl",
        "idefics2.idefics2",
        "idefics3.idefics3",
        "phi4mm.phi4mm",
        "falcon_ocr.falcon_ocr",
        "falcon_perception.falcon_perception",
        "florence2.florence2",
        "molmo.molmo",
        "molmo2.molmo2",
        "moondream3.moondream3",
        "gemma3n.gemma3n",
        "phi3_v.phi3_v",
        "minicpmo.minicpmo",
        "jina_vlm.jina_vlm",
        "qwen3_omni_moe.thinker",
    ],
)
def test_cached_image_features_in_source(model_module):
    """Verify cached_image_features kwarg appears in get_input_embeddings source."""
    try:
        mod = importlib.import_module(f"mlx_vlm.models.{model_module}")
    except Exception as e:
        pytest.skip(f"Cannot import {model_module}: {e}")

    # Find the class that has get_input_embeddings
    target_cls = None
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if hasattr(obj, "get_input_embeddings") and obj.__module__ == mod.__name__:
            target_cls = obj
            break

    assert (
        target_cls is not None
    ), f"No class with get_input_embeddings in {model_module}"

    source = inspect.getsource(target_cls.get_input_embeddings)
    assert "cached_image_features" in source, (
        f"{model_module}.{target_cls.__name__}.get_input_embeddings "
        f"missing cached_image_features check"
    )


# Shared tiny models and checkpoint tensors used by speculation and training.


def module(name):
    return importlib.import_module("mlx_vlm." + name)


def values(name, **overrides):
    return copy.deepcopy(
        DATA["speculative"]["defaults"] | DATA["speculative"][name] | overrides
    )


def tiny_qwen_text_config():
    return module("models.qwen3_5").TextConfig(**values("qwen"))


def tiny_deepseek_config():
    return module("models.deepseek_v4").ModelConfig(**values("deepseek"))


def tiny_glm_text_config():
    return module("models.glm5_next").TextConfig(**values("glm"))


TEXT = {
    "qwen": ("qwen3_5", tiny_qwen_text_config),
    "glm": ("glm5_next", tiny_glm_text_config),
    "deepseek": ("deepseek_v4", tiny_deepseek_config),
}


def dimensions(**overrides):
    return values("defaults", **(dict(intermediate_size=32, head_dim=8) | overrides))


def language(family, *, inference=False, **overrides):
    name, factory = TEXT[family]
    config = factory()
    for key, value in overrides.items():
        setattr(config, key, value)
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
        config = values(family.removeprefix("dspark-"))
        arch = module("models." + config["model_type"])
    else:
        arch = module("models.muse_glimmer")
        config = dict(
            text_config=values(
                "glimmer",
                layer_types=["sliding_attention", "full_attention"],
                layer_rope_theta=[10000.0, 0],
            ),
            vision_config=DATA["speculative"]["glimmer_vision"],
            image_token_id=7,
            video_token_id=6,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    return arch.Model(build_config(arch, config))


def dflash_config(family):
    config = values("glimmer" if family == "glimmer" else "dflash")
    config.update(copy.deepcopy(DATA["speculative"]["dflash_variants"][family]))
    if family.startswith("dspark-lfm"):
        del config["num_target_layers"]
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


def dspark_source(model, cfg):
    """Build native checkpoint names from tiny tensors for the split/load round trip."""
    text = cfg.text_config
    proj_to_w = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    hc = {"attn_hc": "hc_attn", "ffn_hc": "hc_ffn"}
    src = {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("markov_head."):
            # the model-level markov head lives under the last stage on disk
            src[f"mtp.{cfg.n_mtp_layers - 1}.{key}"] = value
            continue
        _, stage, body = key.split(".", 2)
        prefix = f"mtp.{stage}."
        if body.startswith("ffn.switch_mlp."):
            w = proj_to_w[body.split(".")[-2]]
            for expert in range(text.n_routed_experts):
                src[f"{prefix}ffn.experts.{expert}.{w}.weight"] = value[expert]
        elif body.startswith("ffn.shared_experts."):
            w = proj_to_w[body.split(".")[-2]]
            src[f"{prefix}ffn.shared_experts.{w}.weight"] = value
        elif body == "ffn.gate.e_score_correction_bias":
            src[f"{prefix}ffn.gate.bias"] = value
        elif body == "attn.wo_a.weight":
            src[f"{prefix}attn.wo_a.weight"] = (
                value.reshape(text.o_groups * text.o_lora_rank, -1)
                if value.ndim == 3
                else value
            )
        elif body.startswith("attn_hc.") or body.startswith("ffn_hc."):
            module, param = body.split(".")
            src[f"{prefix}{hc[module]}_{param}"] = value
        elif body.startswith("hc_head."):
            src[f"{prefix}hc_head_{body.split('.')[-1]}"] = value
        else:
            src[f"{prefix}{body}"] = value
    return src


def native_speculative_checkpoint(family):
    deepseek = family == "deepseek_v4"
    cfg = (
        tiny_deepseek_config().to_dict()
        if deepseek
        else copy.deepcopy(DATA["speculative"]["glm4_checkpoint"])
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
        shapes = copy.deepcopy(DATA["speculative"]["glm4_checkpoint_shapes"])
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
    return cfg, weights


def glm_mtp_checkpoint_weights(cfg):
    shapes = copy.deepcopy(DATA["speculative"]["glm_fusion_shapes"])
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
    return weights


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
