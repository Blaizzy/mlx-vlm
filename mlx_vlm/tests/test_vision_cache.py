from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.vision_cache import VisionFeatureCache


class TestVisionFeatureCache:
    def test_put_and_get(self):
        cache = VisionFeatureCache(max_size=2)
        features = mx.ones((1, 280, 1536))
        cache.put("image1.jpg", features)
        result = cache.get("image1.jpg")
        assert result is not None
        assert mx.array_equal(result, features)

    def test_cache_miss(self):
        cache = VisionFeatureCache()
        assert cache.get("nonexistent.jpg") is None

    def test_lru_eviction(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("b.jpg", mx.ones((1, 10, 64)) * 2)
        cache.put("c.jpg", mx.ones((1, 10, 64)) * 3)  # evicts a.jpg
        assert cache.get("a.jpg") is None
        assert cache.get("b.jpg") is not None
        assert cache.get("c.jpg") is not None

    def test_lru_touch(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("b.jpg", mx.ones((1, 10, 64)) * 2)
        cache.get("a.jpg")  # touch a, making b the LRU
        cache.put("c.jpg", mx.ones((1, 10, 64)) * 3)  # evicts b
        assert cache.get("a.jpg") is not None
        assert cache.get("b.jpg") is None
        assert cache.get("c.jpg") is not None

    def test_multi_image_key(self):
        cache = VisionFeatureCache()
        features = mx.ones((1, 560, 1536))
        cache.put(["img1.jpg", "img2.jpg"], features)
        assert cache.get(["img1.jpg", "img2.jpg"]) is not None
        assert cache.get(["img2.jpg", "img1.jpg"]) is None  # order matters

    def test_multi_image_feature_collection(self):
        cache = VisionFeatureCache()
        features = [mx.ones((4, 8)), mx.full((6, 8), 2)]

        cache.put(["img1.jpg", "img2.jpg"], features)
        cached = cache.get(["img1.jpg", "img2.jpg"])

        assert isinstance(cached, list)
        assert len(cached) == 2
        assert mx.array_equal(cached[0], features[0])
        assert mx.array_equal(cached[1], features[1])

    def test_url_key(self):
        cache = VisionFeatureCache()
        url = "https://example.com/image.jpg"
        features = mx.ones((1, 280, 1536))
        cache.put(url, features)
        assert cache.get(url) is not None

    def test_clear(self):
        cache = VisionFeatureCache()
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("b.jpg", mx.ones((1, 10, 64)))
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a.jpg") is None

    def test_contains(self):
        cache = VisionFeatureCache()
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        assert "a.jpg" in cache
        assert "b.jpg" not in cache

    def test_overwrite_existing_key(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("a.jpg", mx.ones((1, 10, 64)) * 5)
        assert len(cache) == 1
        result = cache.get("a.jpg")
        assert mx.array_equal(result, mx.ones((1, 10, 64)) * 5)

    def test_default_max_size(self):
        cache = VisionFeatureCache()
        assert cache.max_size == 20

    def test_clear_releases_all(self):
        cache = VisionFeatureCache()
        for i in range(5):
            cache.put(f"img{i}.jpg", mx.ones((1, 10, 64)) * i)
        assert len(cache) == 5
        cache.clear()
        assert len(cache) == 0
        for i in range(5):
            assert cache.get(f"img{i}.jpg") is None


class TestContentKey:
    """Keys must track the pixels the encoder sees, not the image's name.

    A path or URL is not a safe key: the bytes behind a stable name change
    between requests (a camera overwriting frame.jpg, a snapshot URL), which
    previously served the earlier frame's features and silently answered
    about the wrong image.
    """

    def test_same_pixels_same_key(self):
        pixels = mx.zeros((1, 3, 8, 8))
        assert VisionFeatureCache.content_key(pixels) == VisionFeatureCache.content_key(
            mx.zeros((1, 3, 8, 8))
        )

    def test_different_pixels_different_key(self):
        assert VisionFeatureCache.content_key(
            mx.zeros((1, 3, 8, 8))
        ) != VisionFeatureCache.content_key(mx.ones((1, 3, 8, 8)))

    def test_rewritten_frame_misses(self):
        cache = VisionFeatureCache()
        features = mx.ones((1, 4, 8))
        cache.put(VisionFeatureCache.content_key(mx.zeros((1, 3, 8, 8))), features)
        assert (
            cache.get(VisionFeatureCache.content_key(mx.zeros((1, 3, 8, 8))))
            is not None
        )
        assert cache.get(VisionFeatureCache.content_key(mx.ones((1, 3, 8, 8)))) is None

    def test_grid_metadata_separates_identical_pixels(self):
        """Qwen-style towers consume grid metadata alongside the pixels.

        Two solid-colour images with transposed grids preprocess to the same
        flattened patches, so the pixels alone are not a complete key.
        """
        pixels = mx.ones((8, 12))
        wide = VisionFeatureCache.content_key(
            pixels, {"image_grid_thw": mx.array([[1, 2, 4]])}
        )
        tall = VisionFeatureCache.content_key(
            pixels, {"image_grid_thw": mx.array([[1, 4, 2]])}
        )
        assert wide is not None and tall is not None
        assert wide != tall

    def test_nested_fragments_are_unambiguous(self):
        """Distinct structures must not serialise to the same byte stream."""
        pixels = mx.ones((2, 2))
        a = VisionFeatureCache.content_key(pixels, {"a": [1, 23]})
        b = VisionFeatureCache.content_key(pixels, {"a": [12, 3]})
        assert a is not None and b is not None
        assert a != b
        assert VisionFeatureCache.content_key(
            pixels, {"ab": 1}
        ) != VisionFeatureCache.content_key(pixels, {"a": "b1"})

    def test_unhashable_extra_disables_caching(self):
        assert (
            VisionFeatureCache.content_key(mx.ones((2, 2)), {"proc": object()}) is None
        )

    def test_unhashable_pixels_disable_caching(self):
        cache = VisionFeatureCache()
        key = VisionFeatureCache.content_key(object())
        assert key is None
        cache.put(key, mx.ones((1, 4, 8)))
        assert len(cache) == 0
        assert cache.get(key) is None

    def test_server_keys_on_pixels_not_image_source(self):
        """Regression for the server's continuous-batching path.

        _gpu_embed used to set _image_key to the request's image sources, so
        two different frames posted under one path shared a cache entry.
        """
        from mlx_vlm.server.generation import ResponseGenerator

        captured = {}

        class _Model:
            def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kw):
                captured["key"] = kw.get("_image_key")
                return SimpleNamespace(to_dict=lambda: {})

        gen = object.__new__(ResponseGenerator)
        gen.model = _Model()
        gen.vision_cache = VisionFeatureCache()

        same_source = ["/tmp/frame.jpg"]
        gen._gpu_embed(
            {"input_ids": mx.zeros((1, 4)), "pixel_values": mx.zeros((1, 3, 8, 8))},
            images=same_source,
        )
        first = captured["key"]
        gen._gpu_embed(
            {"input_ids": mx.zeros((1, 4)), "pixel_values": mx.ones((1, 3, 8, 8))},
            images=same_source,
        )
        second = captured["key"]

        assert first is not None and second is not None
        assert first != second, "two different frames shared one cache key"


class TestCachedImageFeaturesKwarg:
    """Verify that cached_image_features kwarg is respected by model
    get_input_embeddings without loading weights — tests the code path only."""

    def _make_mock_model_class(self, model_module_name):
        """Import a model module and inspect its Model class for the
        cached_image_features kwarg in get_input_embeddings source."""
        import importlib
        import inspect

        mod = importlib.import_module(f"mlx_vlm.models.{model_module_name}")
        model_cls = None
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name == "Model":
                model_cls = obj
                break
        return model_cls

    # Verify that the cached_image_features kwarg path exists in source code
    # for a representative set of models — no weights needed
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
    def test_cached_image_features_in_source(self, model_module):
        """Verify cached_image_features kwarg appears in get_input_embeddings source."""
        import importlib
        import inspect

        try:
            mod = importlib.import_module(f"mlx_vlm.models.{model_module}")
        except (ImportError, Exception) as e:
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
