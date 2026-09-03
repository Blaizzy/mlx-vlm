import unittest

import mlx.core as mx

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.mla import latent_length, max_absorbed_queries

# Dims chosen only to steer the gate; they do not affect the attention maths.
FORCE_MATERIALIZED = (1, 1, 1)  # threshold 1
FORCE_ABSORBED = (512, 511, 511)  # threshold ~261k


def _dense_attentions():
    """Every MLA attention whose __call__ is (x, mask, cache)."""
    from mlx_vlm.models.deepseek_v3.config import ModelConfig as DSV3Config
    from mlx_vlm.models.deepseek_v3.language import DeepseekV3Attention
    from mlx_vlm.models.glm4_moe_lite.config import ModelConfig as GlmConfig
    from mlx_vlm.models.glm4_moe_lite.language import Glm4MoeLiteAttention

    common = dict(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    yield "deepseek_v3", DeepseekV3Attention(
        DSV3Config(model_type="deepseek_v3", **common)
    )
    yield "glm4_moe_lite", Glm4MoeLiteAttention(
        GlmConfig(model_type="glm4_moe_lite", **common)
    )


class TestMaxAbsorbedQueries(unittest.TestCase):
    def test_asymptotic_matches_mlx_lm(self):
        # cache_len omitted -> the T -> inf limit, as in ml-explore/mlx-lm#1817
        self.assertEqual(max_absorbed_queries(512, 128, 128), 170)
        self.assertEqual(max_absorbed_queries(512, 192, 256), 398)

    def test_cold_cache_rejects_the_absorbed_path(self):
        # At T == L materializing is cheaper for every current model, so the
        # threshold must fall below L rather than admit it.
        for r, n, v in ((512, 128, 128), (512, 192, 256)):
            for L in (2, 32, 64, 169, 398):
                self.assertLess(
                    max_absorbed_queries(r, n, v, cache_len=L),
                    L,
                    f"cold cache admitted L={L} for dims {(r, n, v)}",
                )

    def test_warm_cache_approaches_the_asymptote(self):
        r, n, v = 512, 128, 128
        limit = max_absorbed_queries(r, n, v)
        self.assertEqual(max_absorbed_queries(r, n, v, cache_len=32768), limit - 1)
        self.assertLess(max_absorbed_queries(r, n, v, cache_len=1024), limit)
        # monotonic in cache length
        seq = [max_absorbed_queries(r, n, v, cache_len=t) for t in (256, 1024, 8192)]
        self.assertEqual(seq, sorted(seq))

    def test_degenerate_dims_keep_the_decode_path(self):
        self.assertEqual(max_absorbed_queries(64, 128, 128), 1)
        self.assertGreaterEqual(max_absorbed_queries(1, 1, 1), 1)


class TestLatentLength(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(latent_length(mx.zeros((1, 1, 37, 8))), 37)

    def test_quantized_tuple(self):
        # A quantized KV cache hands back a 3-tuple rather than an array.
        q = (mx.zeros((1, 1, 37, 2)), mx.zeros((1, 1, 37, 1)), mx.zeros((1, 1, 37, 1)))
        self.assertEqual(latent_length(q), 37)


class TestGateWiring(unittest.TestCase):
    """Run the real __call__ and force each branch, so the gates are executed."""

    def _both_branches(self, attn, L, cache_len):
        mx.random.seed(0)
        outs = []
        for dims in (FORCE_MATERIALIZED, FORCE_ABSORBED):
            mx.random.seed(0)
            x = mx.random.normal((1, L, 256))
            cache = KVCache()
            if cache_len:
                warm = mx.random.normal((1, L, 256))
                mx.random.seed(0)
                attn._absorbed_dims = dims
                attn(warm, cache=cache)
            attn._absorbed_dims = dims
            outs.append(attn(x, cache=cache))
        mx.eval(outs)
        return outs

    def test_branches_agree_through_call(self):
        for name, attn in _dense_attentions():
            mx.eval(attn.parameters())
            with self.subTest(model=name):
                materialized, absorbed = self._both_branches(attn, L=4, cache_len=8)
                self.assertTrue(
                    mx.allclose(materialized, absorbed, atol=1e-4, rtol=1e-4),
                    f"{name}: absorbed and materialized disagree through __call__",
                )


class TestGatePairing(unittest.TestCase):
    """The two gates must never disagree: one boolean drives both."""

    MODELS = [
        "deepseek_v3",
        "deepseek_v32",
        "kimi_k3",
        "kimi_linear",
        "longcat_flash",
        "longcat_flash_sparse",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "glm5_next",
        "youtu_vl",
    ]

    def test_one_boolean_two_uses(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "models"
        for m in self.MODELS:
            src = (root / m / "language.py").read_text()
            with self.subTest(model=m):
                self.assertEqual(
                    src.count("absorbed = L == 1 or L <= max_absorbed_queries("),
                    1,
                    f"{m}: expected exactly one gate decision",
                )
                self.assertEqual(
                    src.count("if absorbed:"), 2, f"{m}: expected both gates to use it"
                )


class TestIndexerGateUnchanged(unittest.TestCase):
    """The sparse top-k gather stays at L == 1.

    It selects with ``topk_indices[:, :, 0, :]``, the first query's top-k, so
    widening it would apply one query's selection to all of them.
    """

    SPARSE = [
        "deepseek_v32",
        "longcat_flash_sparse",
        "glm5_next",
        "glm_moe_dsa",
    ]

    def test_first_query_gather_is_still_gated_on_one_query(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1] / "models"
        gate = re.compile(
            r"if L == 1:\s*\n\s*(?:clamped = mx\.clip\(topk_indices|idx = topk_indices)"
        )
        for m in self.SPARSE:
            src = (root / m / "language.py").read_text()
            with self.subTest(model=m):
                self.assertRegex(
                    src,
                    gate,
                    f"{m}: the first-query top-k gather is no longer gated on L == 1",
                )


if __name__ == "__main__":
    unittest.main()
