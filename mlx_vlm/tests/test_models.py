"""Shared model contracts driven by the readable cases in model_cases.json."""

import copy
import importlib
import inspect
import json
import unittest
from operator import attrgetter
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_map


class ModelChecks(unittest.TestCase):
    """Reusable assertions; each JSON case constructs fresh configs and models."""

    def _check_returns_input_embeddings_features(self, model, model_name):
        """Helper to test get_input_embeddings returns InputEmbeddingsFeatures."""
        from mlx_vlm.models.base import InputEmbeddingsFeatures

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
        self, mm_projector, vision_hidden_size, text_hidden_size
    ):

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            mm_projector.update(
                tree_map(lambda p: p.astype(t), mm_projector.parameters())
            )

            vision_features = mx.random.uniform(
                shape=(batch_size, vision_hidden_size), dtype=t
            )
            input_tensor = mx.array(vision_features)

            outputs = mm_projector(input_tensor)
            self.assertEqual(outputs.shape, (batch_size, text_hidden_size))
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
            self.assertEqual(vision_tower.model_type, model_type)

            if model_type in [
                "qwen2_5_vl",
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

            hidden_states = hidden_states[vision_feature_layer]

            # Check vision hidden feature layer's shape matches the expected hidden size
            if channel_first:
                self.assertEqual(hidden_states.shape[1], vision_hidden_size)
            else:
                self.assertEqual(hidden_states.shape[-1], vision_hidden_size)

            self.assertEqual(hidden_states.dtype, t)

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
    "language": "language_test_runner",
    "projector": "mm_projector_test_runner",
    "vision": "vision_test_runner",
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
    if kind == "input_embeddings":
        return (model, case["module"]), {}
    if kind in {"request_positions", "chunked_positions"}:
        return (model,), {}
    if kind in {"language", "mrope_cache_index", "mrope_deltas"}:
        language_model = model if case.get("language_only") else model.language_model
        text_config = config.text_config
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
        return (
            projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        ), {}
    if kind == "vision":
        vision = attrgetter(case.get("vision_path", "vision_tower"))(model)
        vision_config = config.vision_config
        options = case.get("vision", {})
        image_size = options.get("input_shape")
        if image_size is None:
            image_size = (vision_config.image_size, vision_config.image_size)
        hidden_size = first_attribute(
            vision_config, "out_hidden_size", "hidden_size", "d_model", "width"
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
            vision_config.model_type,
            hidden_size,
            channels,
            tuple(image_size),
        ), kwargs
    raise ValueError(f"Unknown model check: {kind}")


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_model_contract(case):
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    config = build_config(module, case["config"])
    model = (
        module.LanguageModel(config.text_config, config)
        if case.get("language_only")
        else module.Model(config)
    )
    checks = ModelChecks()
    for kind in case["checks"]:
        args, kwargs = check_arguments(kind, case, model, config)
        getattr(checks, CHECKS[kind])(*args, **kwargs)


@pytest.mark.parametrize("name", DATA["dense"])
def test_dense_model(name):
    module = importlib.import_module("mlx_vlm.models." + name)
    config = DATA["dense"][name]
    model = module.Model(module.ModelConfig.from_dict(copy.deepcopy(config)))
    model.eval()
    mx.eval(model.parameters())
    ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
    assert model(ids).logits.shape == (1, 8, config["vocab_size"])
    cache = model.language_model.make_cache()
    model(ids[:, :-1], cache=cache)
    assert model(ids[:, -1:], cache=cache).logits.shape == (1, 1, config["vocab_size"])
