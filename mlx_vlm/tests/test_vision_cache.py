import pytest
import mlx.core as mx

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
        assert cache.max_size == 8
