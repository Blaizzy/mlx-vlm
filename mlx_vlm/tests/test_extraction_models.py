"""Text and vision extraction, privacy tagging, checkpoints, and quantized inference."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.gliner import (
    GLiNER2,
    _CharSplitter,
    _resolve_flat_overlaps,
    _schema_tokens,
    _WhitespaceSplitter,
)
from mlx_vlm.models.gliner2_5 import Model as GlinerModel
from mlx_vlm.models.gliner2_5 import ModelConfig as GlinerConfig
from mlx_vlm.models.gliner2_5.boundary import (
    DocumentCandidatePool,
    Marginals,
    PooledCandidates,
    SharedPoolScorer,
)
from mlx_vlm.models.openai_privacy_filter import Model as PrivacyModel
from mlx_vlm.models.openai_privacy_filter import ModelConfig as PrivacyConfig
from mlx_vlm.privacy_filter import PrivacyFilter
from mlx_vlm.utils import get_model_and_args, load_config


def test_checkpoint_key_sanitization():
    weights = {
        "encoder.embeddings.LayerNorm.weight": mx.ones((4,)),
        "encoder.encoder.layer.0.attention.self.query_proj.weight": mx.ones((4, 4)),
        "boundary_head.boundary_proposer.end_key_projection.weight": mx.ones((4, 4)),
        "boundary_head.shared_pool_builder.start_projection.weight": mx.ones((4, 4)),
    }

    sanitized = GlinerModel.sanitize(None, weights)

    assert "encoder.embeddings.layer_norm.weight" in sanitized
    assert "encoder.encoder.layers.0.attention.self_attn.query_proj.weight" in sanitized
    assert "boundary_head.boundary_proposer.end_key_projection.weight" not in sanitized
    assert "boundary_head.shared_pool_builder.start_projection.weight" in sanitized


def test_word_splitters_preserve_offsets():
    text = "Email Me@Example.com 北京"

    whitespace = _WhitespaceSplitter()(text)
    characters = _CharSplitter()(text)

    assert whitespace[1] == ("me@example.com", 6, 20)
    assert characters[-2:] == [("北", 21, 22), ("京", 22, 23)]


def test_shared_pool_scorer_keeps_candidate_major_layout():
    """Span-marginal gathers must stay candidate-major.

    Indexing a ``(queries, positions)`` plane with a candidate vector puts the
    advanced axis first, so the gather is ``(candidates, queries)`` -- matching
    ``score``. Transposing it instead makes the scorer unusable for any input
    where the candidate and query counts differ.
    """
    batch, positions, candidates, queries = 1, 6, 5, 3
    hidden_size, boundary_dim = 16, 8
    scorer = SharedPoolScorer(
        hidden_size, boundary_dim, {"pair_dim": 8, "content_dim": 4}
    )

    starts = mx.array([[0, 1, 2, 0, 3]])
    ends = mx.array([[2, 3, 4, 5, 5]])
    pooled = PooledCandidates(
        indices=mx.stack((starts, ends), axis=-1),
        mask=mx.ones((batch, candidates), dtype=mx.bool_),
        compat_logits=mx.zeros((batch, candidates)),
    )
    marginals = Marginals(
        start_logits=mx.zeros((batch, queries, positions)),
        end_logits=mx.zeros((batch, queries, positions)),
        inside_prefix=mx.zeros((batch, queries, positions)),
        inside_mean=mx.zeros((batch, queries, 1)),
    )

    logits = scorer(
        mx.zeros((batch, positions, boundary_dim)),
        mx.zeros((batch, queries, hidden_size)),
        mx.ones((batch, queries), dtype=mx.bool_),
        pooled,
        marginals,
        mx.zeros((batch, positions, hidden_size)),
        mx.ones((batch, positions)),
    )

    assert logits.shape == (batch, candidates, queries)


def _zeroed_pool(boundary_dim, settings):
    pool = DocumentCandidatePool(boundary_dim, settings)
    # Zero the projections so pair compatibility drops out and the ranking is
    # exactly start_logits + end_logits.
    for projection in (pool.start_projection, pool.end_projection):
        projection.weight = mx.zeros_like(projection.weight)
        projection.bias = mx.zeros_like(projection.bias)
    return pool


def test_candidate_pool_promotes_each_query_best_pair():
    """Every active query must get its top-scoring pair into the shared pool.

    Uses 3 queries and 6 boundaries so the query axis and the pair axis have
    different lengths -- a gather that transposes them would pick the wrong
    pairs (or fail outright) instead of silently agreeing.
    """
    boundaries, queries = 6, 3
    settings = {"pool_boundary_top_k": 4, "pool_size": 6, "min_pool_per_query": 2}
    pool = _zeroed_pool(8, settings)

    start_logits = mx.zeros((1, queries, boundaries))
    end_logits = mx.zeros((1, queries, boundaries))
    # query 0 -> (0, 5), query 1 -> (1, 4), query 2 -> (2, 3)
    wanted = [(0, 5), (1, 4), (2, 3)]
    for query, (start, end) in enumerate(wanted):
        start_logits[0, query, start] = 10.0 + query
        end_logits[0, query, end] = 10.0 + query

    pooled = pool(
        mx.zeros((1, boundaries, 8)),
        mx.ones((1, boundaries), dtype=mx.bool_),
        mx.ones((1, queries), dtype=mx.bool_),
        start_logits,
        end_logits,
    )

    assert pooled.indices.shape == (1, settings["pool_size"], 2)
    assert pooled.mask.shape == (1, settings["pool_size"])
    selected = {
        (start, end)
        for (start, end), keep in zip(
            pooled.indices[0].tolist(), pooled.mask[0].tolist()
        )
        if keep
    }
    assert all(end > start for start, end in selected)
    for pair in wanted:
        assert pair in selected, f"{pair} missing from {sorted(selected)}"


def test_schema_tokens_carry_prompt_and_descriptions():
    schema = _schema_tokens(
        "sentiment",
        ["positive", "negative"],
        "[L]",
        prompt="rate the review",
        descriptions={"positive": "approving tone"},
    )

    prompt_text = schema[2]
    assert prompt_text.startswith("sentiment: rate the review")
    assert "[DESCRIPTION] positive: approving tone" in prompt_text
    # Undescribed labels must not gain a description clause.
    assert "negative:" not in prompt_text


def test_flat_overlap_resolution_keeps_disjoint_spans():
    spans = [(0.9, 0, 2), (0.8, 3, 5), (0.7, 6, 7)]

    assert _resolve_flat_overlaps(spans) == spans


@pytest.mark.parametrize(
    "added,kwargs,error",
    [(0, {"word_splitter": "bpe"}, "word_splitter"), (4, {}, "special tokens")],
)
def test_gliner2_rejects_incompatible_tokenization(added, kwargs, error):
    tokenizer = SimpleNamespace(add_special_tokens=lambda _: added)
    with pytest.raises(ValueError, match=error):
        GLiNER2(object(), tokenizer, **kwargs)


def test_quantized_encoder_still_runs():
    """A quantized checkpoint must still encode.

    ``mlx_vlm.convert -q`` turns ``rel_embeddings`` into a QuantizedEmbedding,
    whose ``.weight`` is packed uint32 with a narrower last dimension. Reading
    that attribute directly instead of calling the module makes the relative
    embedding LayerNorm reject the shape, so every quantized model fails at
    inference even though conversion reports success.
    """
    config = GlinerConfig.from_dict(
        {
            "model_type": "extractor",
            "architecture": "boundary",
            "encoder_config": {
                "vocab_size": 512,
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "position_buckets": 64,
                "max_relative_positions": 128,
            },
        }
    )
    model = GlinerModel(config)
    model.eval()
    ids = mx.zeros((1, 16), dtype=mx.int32)
    mask = mx.ones((1, 16), dtype=mx.bool_)
    reference = model.encode(ids, mask)
    mx.eval(reference)

    nn.quantize(model.encoder, group_size=64, bits=4)
    mx.eval(model.parameters())

    assert isinstance(
        model.encoder.encoder.rel_embeddings, nn.QuantizedEmbedding
    ), "rel_embeddings should have been quantized for this test to be meaningful"
    encoded = model.encode(ids, mask)
    mx.eval(encoded)

    assert encoded.shape == reference.shape
    assert bool(mx.all(mx.isfinite(encoded.astype(mx.float32))))


def _write_checkpoint(root, *, sidecar=True, inline=False):
    encoder = {"vocab_size": 128, "hidden_size": 64, "num_attention_heads": 4}
    config = {
        "model_type": "extractor",
        "architecture": "boundary",
        "architectures": ["BoundaryExtractor"],
    }
    if inline:
        config["encoder_config"] = encoder
    (root / "config.json").write_text(json.dumps(config))
    if sidecar:
        (root / "encoder_config").mkdir()
        (root / "encoder_config" / "config.json").write_text(json.dumps(encoder))
    return root


@pytest.mark.parametrize(
    "sidecar,inline",
    [(True, False), (False, True), (False, False)],
    ids=["sidecar", "inline", "missing"],
)
def test_gliner_encoder_config_loading(tmp_path, sidecar, inline):
    root = _write_checkpoint(tmp_path, sidecar=sidecar, inline=inline)
    if sidecar or inline:
        assert load_config(root)["encoder_config"]["hidden_size"] == 64
    else:
        with pytest.raises(FileNotFoundError, match="encoder config not found"):
            load_config(root)


def test_boundary_architecture_selects_the_gliner_module():
    _, model_type = get_model_and_args(
        {
            "model_type": "extractor",
            "architecture": "boundary",
            "architectures": ["BoundaryExtractor"],
        }
    )

    assert model_type == "gliner2_5"


LABELS = {0: "O", 1: "B-x", 2: "I-x", 3: "E-x", 4: "S-x"}


def _privacy_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "sliding_window": 2,
        "max_position_embeddings": 32,
        "default_n_ctx": 32,
        "pad_token_id": 31,
        "eos_token_id": 31,
        "num_labels": 5,
        "id2label": LABELS,
        "attention_chunk_size": 3,
        "moe_chunk_size": 3,
    }
    values.update(overrides)
    return PrivacyConfig(**values)


def test_checkpoint_expert_layout_sanitization():
    gate_up = mx.arange(2 * 3 * 8).reshape(2, 3, 8)
    down = mx.arange(2 * 4 * 3).reshape(2, 4, 3)
    weights = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((2, 8)),
        "model.layers.0.mlp.experts.down_proj": down,
        "model.layers.0.mlp.experts.down_proj_bias": mx.zeros((2, 3)),
        "score.weight": mx.zeros((5, 3)),
    }

    sanitized = PrivacyModel.sanitize(None, weights)

    gate_key = "model.layers.0.mlp.experts.gate_proj.weight"
    up_key = "model.layers.0.mlp.experts.up_proj.weight"
    down_key = "model.layers.0.mlp.experts.down_proj.weight"
    transposed = gate_up.swapaxes(-1, -2)
    assert sanitized[gate_key].shape == (2, 4, 3)
    assert sanitized[up_key].shape == (2, 4, 3)
    assert sanitized[down_key].shape == (2, 3, 4)
    assert mx.array_equal(sanitized[gate_key], transposed[:, :4]).item()
    assert mx.array_equal(sanitized[up_key], transposed[:, 4:]).item()
    assert mx.array_equal(sanitized[down_key], down.swapaxes(-1, -2)).item()
    assert "model.layers.0.mlp.experts.gate_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.up_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.down_proj.bias" in sanitized
    assert "score.weight" in sanitized


@pytest.mark.parametrize(
    "quantization",
    [
        None,
        (64, 4, "affine"),
        (64, 5, "affine"),
        (64, 6, "affine"),
        (64, 8, "affine"),
        (32, 4, "mxfp4"),
        (16, 4, "nvfp4"),
        (32, 8, "mxfp8"),
    ],
    ids=[
        "float-chunk-boundaries",
        "affine4",
        "affine5",
        "affine6",
        "affine8",
        "mxfp4",
        "nvfp4",
        "mxfp8",
    ],
)
def test_privacy_forward(quantization):
    overrides = (
        dict(hidden_size=64, intermediate_size=64, head_dim=16, num_attention_heads=4)
        if quantization
        else {}
    )
    model = PrivacyModel(_privacy_config(**overrides))
    if quantization:
        group_size, bits, mode = quantization
        nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(input_ids, attention_mask=mx.ones_like(input_ids)).logits
    mx.eval(logits)
    assert logits.shape == (1, 5, 5)
    assert mx.all(mx.isfinite(logits)).item()


class _PrivacyTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "Alice emailed bob@example.com"
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 5), (5, 13), (13, 29)]}


class _PrivacyModel:
    def __init__(self):
        labels = {
            0: "O",
            1: "B-private_person",
            2: "I-private_person",
            3: "E-private_person",
            4: "S-private_person",
            5: "B-private_email",
            6: "I-private_email",
            7: "E-private_email",
            8: "S-private_email",
        }
        self.config = SimpleNamespace(
            id2label=labels, num_labels=len(labels), default_n_ctx=16
        )

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        logits = mx.full((1, input_ids.shape[1], 9), -10.0)
        logits[0, 0, 4] = 10.0
        logits[0, 1, 0] = 10.0
        logits[0, 2, 8] = 10.0
        return SimpleNamespace(logits=logits)


def test_high_level_api_returns_offsets_and_redacted_text():
    detector = PrivacyFilter(_PrivacyModel(), _PrivacyTokenizer())

    result = detector("Alice emailed bob@example.com")

    assert [(span.label, span.start, span.end) for span in result.spans] == [
        ("private_person", 0, 5),
        ("private_email", 14, 29),
    ]
    assert result.redacted_text == ("<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>")
    assert result.to_dict()["spans"][0]["text"] == "Alice"


class TestSapiens2(unittest.TestCase):
    def _tiny_config(self, task="backbone", **overrides):
        from mlx_vlm.models.sapiens2.config import ModelConfig

        archs = {
            "backbone": "Sapiens2Model",
            "seg": "Sapiens2ForSemanticSegmentation",
            "pose": "Sapiens2ForPoseEstimation",
            "normal": "Sapiens2ForNormalEstimation",
            "pointmap": "Sapiens2ForPointmapEstimation",
            "matting": "Sapiens2ForImageMatting",
        }
        args = dict(
            architectures=[archs[task]],
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_first_full_attention_layers=1,
            num_last_full_attention_layers=1,
            intermediate_size=128,
            image_size=[32, 24],
            patch_size=8,
            num_register_tokens=2,
        )
        args.update(overrides)
        return ModelConfig(**args)

    def _deconv_head(self, **kw):
        from mlx_vlm.models.sapiens2.config import HeadConfig

        return HeadConfig(
            upsample_out_channels=[32, 16],
            upsample_kernel_sizes=[4, 4],
            conv_out_channels=[8],
            conv_kernel_sizes=[1],
            **kw,
        )

    def _pixel_shuffle_head(self, **kw):
        from mlx_vlm.models.sapiens2.config import HeadConfig

        return HeadConfig(
            upsample_out_channels=[32, 16, 8],
            upsample_kernel_sizes=[3, 3, 3],
            conv_out_channels=[8],
            conv_kernel_sizes=[3],
            use_pixel_shuffle=True,
            **kw,
        )

    def _predictor(self, task, head_config, num_labels, **overrides):
        from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        config = self._tiny_config(
            task, head_config=head_config, num_labels=num_labels, **overrides
        )
        return Sapiens2Predictor(Model(config))

    @staticmethod
    def _image(h=40, w=30):
        return np.random.default_rng(0).integers(0, 255, (h, w, 3), np.uint8)

    def test_registry_exposes_model(self):
        """The package resolves through the shared loader like other models."""
        import dataclasses

        from mlx_vlm.utils import get_model_and_args

        model_module, model_type = get_model_and_args({"model_type": "sapiens2"})
        self.assertEqual(model_type, "sapiens2")
        config = model_module.ModelConfig.from_dict(
            dataclasses.asdict(self._tiny_config())
        )
        self.assertIsInstance(model_module.Model(config), model_module.Model)

    def test_config_from_hf_dict(self):
        """from_dict parses an official-style config.json (nested head_config,
        unknown keys ignored, num_labels from id2label)."""
        from mlx_vlm.models.sapiens2.config import HeadConfig, ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "sapiens2",
                "architectures": ["Sapiens2ForSemanticSegmentation"],
                "hidden_size": 1024,
                "stage_names": ["stem", "stage1"],  # unknown key
                "id2label": {str(i): f"L{i}" for i in range(29)},
                "head_config": {
                    "model_type": "sapiens2_head",
                    "upsample_out_channels": [512, 256, 128, 64],
                    "upsample_kernel_sizes": [4, 4, 4, 4],
                    "conv_out_channels": [64, 64],
                    "conv_kernel_sizes": [1, 1],
                    "use_pixel_shuffle": None,
                    "chunk_size_feed_forward": 0,  # unknown key
                },
            }
        )
        self.assertEqual(config.task, "seg")
        self.assertEqual(config.num_labels, 29)
        self.assertIsInstance(config.head_config, HeadConfig)
        self.assertEqual(config.head_config.upsample_out_channels, [512, 256, 128, 64])

    def test_kv_heads_per_layer(self):
        """Middle layers use half the query heads (GQA); edges full MHSA."""
        config = self._tiny_config()
        self.assertEqual(config.kv_heads_per_layer, [4, 2, 4])
        explicit = self._tiny_config(num_key_value_heads_per_layer=[4, 4, 4])
        self.assertEqual(explicit.kv_heads_per_layer, [4, 4, 4])

    def test_backbone_shapes(self):
        """Tokens include 1 cls + R register + H*W/patch^2 patch tokens."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        model = Model(self._tiny_config())
        out = model(mx.random.normal((2, 32, 24, 3)))
        # grid 4x3 = 12 patches + 3 prefix tokens
        self.assertEqual(out["last_hidden_state"].shape, (2, 15, 64))
        self.assertEqual(out["pooler_output"].shape, (2, 64))

    def test_deconv_head_shapes(self):
        """Seg/pose heads upsample x2 per deconv block."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        for task, labels in (("seg", 29), ("pose", 308)):
            config = self._tiny_config(
                task, head_config=self._deconv_head(), num_labels=labels
            )
            model = Model(config)
            out = model(mx.random.normal((1, 32, 24, 3)))
            key = "logits" if task == "seg" else "heatmaps"
            self.assertEqual(out[key].shape, (1, 16, 12, labels))

    def test_pixel_shuffle_head_shapes(self):
        """Normal/matting heads upsample x2 per PixelShuffle block."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        config = self._tiny_config(
            "normal", head_config=self._pixel_shuffle_head(), num_labels=3
        )
        model = Model(config)
        out = model(mx.random.normal((1, 32, 24, 3)))
        self.assertEqual(out["normals"].shape, (1, 32, 24, 3))

        config = self._tiny_config(
            "matting", head_config=self._pixel_shuffle_head(), num_labels=4
        )
        model = Model(config)
        out = model(mx.random.normal((1, 32, 24, 3)))
        self.assertEqual(out["alphas"].shape, (1, 32, 24, 1))
        self.assertEqual(out["foregrounds"].shape, (1, 32, 24, 3))
        # sigmoid output range
        self.assertLessEqual(float(out["alphas"].max()), 1.0)
        self.assertGreaterEqual(float(out["alphas"].min()), 0.0)

    def test_pointmap_scale_branch(self):
        """Pointmap head also returns a per-image scale."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        head = self._pixel_shuffle_head(
            scale_conv_out_channels=[32, 16, 8],
            scale_conv_kernel_sizes=[1, 1, 1],
            scale_final_input_size=8,  # grid 4x3 -> 1x1 after 3 stride-2 convs
            scale_final_hidden_sizes=[16, 8],
        )
        config = self._tiny_config("pointmap", head_config=head, num_labels=3)
        model = Model(config)
        out = model(mx.random.normal((1, 32, 24, 3)))
        self.assertEqual(out["pointmaps"].shape, (1, 32, 24, 3))
        self.assertEqual(out["scales"].shape, (1, 1))

    def test_tokenizer_backbone(self):
        """The 4K tokenizer reduces the token grid by the window size."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        config = self._tiny_config(
            image_size=[32, 32], use_tokenizer=True, num_tokenizer_layers=2
        )
        model = Model(config)
        out = model(mx.random.normal((1, 32, 32, 3)))
        # grid 4x4, window 4 -> 1 token + 3 prefix
        self.assertEqual(out["last_hidden_state"].shape, (1, 4, 64))

    def test_weight_names_match_checkpoint(self):
        """Parameter names follow the official checkpoint layout, except the
        q/k/v projections which ``sanitize`` merges into ``wqkv``."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        config = self._tiny_config(
            "seg",
            head_config=self._deconv_head(),
            num_labels=29,
        )
        keys = {k for k, _ in tree_flatten(Model(config).parameters())}
        expected = {
            "backbone.patch_embed.projection.weight",
            "backbone.cls_token",
            "backbone.storage_tokens",
            "backbone.rope_embed.periods",
            "backbone.blocks.0.ln1.weight",
            "backbone.blocks.0.attn.wqkv.weight",
            "backbone.blocks.0.attn.q_norm.weight",
            "backbone.blocks.0.attn.gamma.weight",
            "backbone.blocks.0.ffn.w12.weight",
            "backbone.blocks.0.ffn.w3.weight",
            "backbone.ln1.weight",
            "decode_head.deconv_layers.0.weight",
            "decode_head.deconv_layers.3.weight",
            "decode_head.conv_layers.0.weight",
            "decode_head.conv_seg.weight",
        }
        self.assertTrue(expected.issubset(keys))

    def test_sanitize_relays_conv_weights(self):
        """sanitize converts torch conv layouts and prefixes bare backbone keys."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        config = self._tiny_config(
            "seg", head_config=self._deconv_head(), num_labels=29
        )
        model = Model(config)
        params = dict(tree_flatten(model.parameters()))

        weights = {}
        for k, v in params.items():
            if k.endswith("projection.weight"):
                weights[k] = mx.zeros((v.shape[0], v.shape[3], 8, 8))  # torch Conv2d
            elif "deconv_layers" in k:
                weights[k] = mx.zeros((v.shape[3], v.shape[0], 4, 4))  # torch ConvT
            else:
                weights[k] = v
        sanitized = model.sanitize(weights)
        for k, v in sanitized.items():
            self.assertEqual(v.shape, params[k].shape, f"{k} not relayed out correctly")

        # bare (pretrain-style) keys get the backbone. prefix
        bare = model.sanitize({"patch_embed.projection.bias": mx.zeros((64,))})
        self.assertIn("backbone.patch_embed.projection.bias", bare)

    def test_pixel_shuffle_matches_torch_layout(self):
        """Channel-last PixelShuffle equals torch's channel-last view."""

        from mlx_vlm.models.sapiens2.heads import PixelShuffle

        x = np.arange(2 * 2 * 4 * 16, dtype=np.float32).reshape(2, 2, 4, 16)
        out = np.array(PixelShuffle(2)(mx.array(x)))
        # torch PixelShuffle on (B, C*4, H, W) == channel-last reference
        B, H, W, C4 = x.shape
        ref = x.reshape(B, H, W, 4, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        ref = ref.reshape(B, H * 2, W * 2, 4)
        self.assertTrue(np.array_equal(out, ref))

    def test_udp_decode_recovers_peak(self):
        """A single-peak heatmap decodes to the peak location (DARK ~exact)."""
        from mlx_vlm.models.sapiens2.pose import udp_decode_batch

        K, H, W = 2, 256, 192
        heatmaps = np.zeros((H, W, K), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for k, (cy, cx) in enumerate([(100.0, 60.0), (30.5, 120.5)]):
            heatmaps[..., k] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6**2))
        kps, scores = udp_decode_batch(mx.array(heatmaps)[None], input_size=(768, 1024))
        kps, scores = np.array(kps), np.array(scores)
        self.assertEqual(kps.shape, (1, K, 2))
        # input_size scaling: x * 768, y * 1024 over (W-1, H-1)
        want_x = np.array([60.0, 120.5]) / (W - 1) * 768
        want_y = np.array([100.0, 30.5]) / (H - 1) * 1024
        self.assertTrue(np.allclose(kps[0, :, 0], want_x, atol=1.0))
        self.assertTrue(np.allclose(kps[0, :, 1], want_y, atol=1.0))
        self.assertTrue(np.all(scores > 0.9))

    def test_flip_indices(self):
        from mlx_vlm.models.sapiens2.pose import flip_indices_from_pairs

        idx = flip_indices_from_pairs(4, [[1, 2]])
        self.assertEqual(idx.tolist(), [0, 2, 1, 3])

    def test_sanitize_merges_qkv(self):
        """Checkpoint wq/wk/wv (weight + bias) become one wqkv, q|k|v order."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        model = Model(self._tiny_config())
        attn = model.backbone.blocks[1].attn  # GQA layer: 4 q heads, 2 kv heads
        d, kv = attn.embed_dims, attn.kv_size
        parts = {
            "q": (mx.ones((d, 64)), mx.full((d,), 1.0)),
            "k": (mx.full((kv, 64), 2.0), mx.full((kv,), 2.0)),
            "v": (mx.full((kv, 64), 3.0), mx.full((kv,), 3.0)),
        }
        weights = {}
        for name, (w, b) in parts.items():
            weights[f"backbone.blocks.1.attn.w{name}.weight"] = w
            weights[f"backbone.blocks.1.attn.w{name}.bias"] = b
        out = model.sanitize(weights)
        self.assertEqual(
            set(out),
            {
                "backbone.blocks.1.attn.wqkv.weight",
                "backbone.blocks.1.attn.wqkv.bias",
            },
        )
        w = np.array(out["backbone.blocks.1.attn.wqkv.weight"])
        self.assertEqual(w.shape, (d + 2 * kv, 64))
        self.assertTrue((w[:d] == 1).all() and (w[d : d + kv] == 2).all())
        self.assertTrue((w[d + kv :] == 3).all())
        # already-merged keys pass through
        again = model.sanitize(out)
        self.assertEqual(set(again), set(out))

    def test_sanitize_unifies_mixed_dtypes(self):
        """A bf16 checkpoint with float32 norms/biases loads as all-bf16."""
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        model = Model(self._tiny_config())
        weights = {}
        for k, v in tree_flatten(model.parameters()):
            keep = k.endswith((".bias", "ln1.weight", "ln2.weight"))
            weights[k] = v.astype(mx.float32 if keep else mx.bfloat16)
        out = model.sanitize(weights)
        self.assertTrue(all(v.dtype == mx.bfloat16 for v in out.values()))
        # single-dtype checkpoints are left alone
        out32 = model.sanitize(dict(tree_flatten(model.parameters())))
        self.assertTrue(all(v.dtype == mx.float32 for v in out32.values()))

    def test_attention_matches_unfused_reference(self):
        """Fused qkv + kernel-side GQA + packed full-sequence RoPE equal the
        original formulation (separate projections, repeated kv heads,
        rotate-half RoPE on patch tokens only)."""
        from mlx_vlm.models.sapiens2.backbone import (
            GroupedQueryAttention,
            RopePositionEmbedding,
        )

        D, heads, kv_heads, prefix, h, w = 64, 4, 2, 3, 4, 3
        attn = GroupedQueryAttention(D, heads, kv_heads)
        rope = RopePositionEmbedding(D, heads)
        x = mx.random.normal((2, prefix + h * w, D))
        out = attn(x, rope=rope(h, w, prefix=prefix))

        # reference
        hd = D // heads
        wq, wk, wv = mx.split(attn.wqkv.weight, [D, D + kv_heads * hd], axis=0)
        bq, bk, bv = mx.split(attn.wqkv.bias, [D, D + kv_heads * hd])
        B, N, _ = x.shape
        q = (x @ wq.T + bq).reshape(B, N, heads, hd).transpose(0, 2, 1, 3)
        k = (x @ wk.T + bk).reshape(B, N, kv_heads, hd).transpose(0, 2, 1, 3)
        v = (x @ wv.T + bv).reshape(B, N, kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = attn.q_norm(q), attn.k_norm(k)
        k = mx.repeat(k, heads // kv_heads, axis=1)
        v = mx.repeat(v, heads // kv_heads, axis=1)
        sin, cos = rope(h, w)  # patch tokens only, unpack to full width
        sin, cos = mx.tile(sin[:, 1], (1, 2)), mx.tile(cos[:, 0], (1, 2))

        def rot(t):
            t1, t2 = mx.split(t, 2, axis=-1)
            return mx.concatenate([-t2, t1], axis=-1)

        def apply(t):
            patch = t[:, :, prefix:]
            return mx.concatenate(
                [t[:, :, :prefix], patch * cos + rot(patch) * sin], axis=-2
            )

        ref = mx.fast.scaled_dot_product_attention(
            apply(q), apply(k), v, scale=hd**-0.5
        )
        ref = attn.gamma(attn.proj(ref.transpose(0, 2, 1, 3).reshape(B, N, D)))
        self.assertTrue(np.allclose(np.array(out), np.array(ref), atol=1e-5))

    def test_rope_tables_prefix_identity(self):
        from mlx_vlm.models.sapiens2.backbone import RopePositionEmbedding

        rope = RopePositionEmbedding(embed_dim=64, num_heads=4)
        sin, cos = rope(4, 3, prefix=3)
        self.assertEqual(sin.shape, (15, 2, 8))
        self.assertTrue(np.all(np.array(sin[:3]) == 0))
        self.assertTrue(np.all(np.array(cos[:3]) == 1))
        self.assertTrue(np.array_equal(np.array(sin[3:]), np.array(rope(4, 3)[0])))

    def test_backbone_casts_input_to_parameter_dtype(self):
        from mlx_vlm.models.sapiens2.sapiens2 import Model

        model = Model(self._tiny_config())
        model.apply(lambda p: p.astype(mx.bfloat16))
        out = model(mx.random.normal((1, 32, 24, 3)))  # float32 pixels
        self.assertEqual(out["last_hidden_state"].dtype, mx.bfloat16)

    def test_deconv_upsample_matches_conv_transpose(self):
        """The im2col/matmul deconv equals mx.conv_transpose2d(k4, s2, p1)."""
        from mlx_vlm.models.sapiens2.heads import DeconvUpsample

        layer = DeconvUpsample(6, 5)
        x = mx.random.normal((2, 7, 5, 6))
        ref = mx.conv_transpose2d(x, layer.weight, stride=2, padding=1)
        out = layer(x)
        self.assertEqual(out.shape, (2, 14, 10, 5))
        self.assertTrue(np.allclose(np.array(out), np.array(ref), atol=1e-4))
        # cached gemm weight is rebuilt when the parameter changes
        layer.update({"weight": mx.zeros_like(layer.weight)})
        self.assertTrue(np.all(np.array(layer(x)) == 0))

    def test_conv3x3_matches_conv2d(self):
        """The im2col 3x3 conv equals mx.conv2d on both the gemm path and the
        large-map fallback."""
        from mlx_vlm.models.sapiens2.heads import Conv3x3

        layer = Conv3x3(6, 4)
        x = mx.random.normal((2, 7, 5, 6))
        ref = mx.conv2d(x, layer.weight, padding=1) + layer.bias
        self.assertTrue(np.allclose(np.array(layer(x)), np.array(ref), atol=1e-4))
        layer.max_im2col_elements = 1  # force the conv fallback
        self.assertTrue(np.allclose(np.array(layer(x)), np.array(ref), atol=1e-4))

    def test_conv1x1_matches_conv2d(self):
        from mlx_vlm.models.sapiens2.heads import Conv1x1

        layer = Conv1x1(6, 4)
        x = mx.random.normal((2, 5, 3, 6))
        ref = mx.conv2d(x, layer.weight) + layer.bias
        self.assertTrue(np.allclose(np.array(layer(x)), np.array(ref), atol=1e-5))

    def test_udp_decode_batch_matches_reference_codec(self):
        """Device decode == the mmpose UDP/DARK codec (cv2 blur, numpy)."""
        from mlx_vlm.models.sapiens2.pose import udp_decode_batch

        try:
            import cv2
        except ImportError:  # pragma: no cover
            self.skipTest("opencv not installed")

        def reference_decode(hm, kernel=11):  # hm: (K, H, W)
            K, H, W = hm.shape
            flat = hm.reshape(K, -1)
            vals, idx = flat.max(axis=-1), flat.argmax(axis=-1)
            locs = np.stack([idx % W, idx // W], axis=-1).astype(np.float32)
            locs[vals <= 0] = -1
            border = (kernel - 1) // 2
            blurred = np.empty_like(hm)
            for k in range(K):
                origin_max = hm[k].max()
                dr = np.zeros((H + 2 * border, W + 2 * border), np.float32)
                dr[border:-border, border:-border] = hm[k]
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                blurred[k] = dr[border:-border, border:-border]
                blurred[k] *= origin_max / blurred[k].max()
            np.log(np.clip(blurred, 1e-3, 50.0), blurred)
            pad = np.pad(blurred, ((0, 0), (1, 1), (1, 1)), mode="edge")
            out = locs.copy()
            for k in range(K):
                px, py = int(locs[k, 0]) + 1, int(locs[k, 1]) + 1
                p = pad[k]
                i_, ix1, iy1, ix1y1 = (
                    p[py, px],
                    p[py, px + 1],
                    p[py + 1, px],
                    p[py + 1, px + 1],
                )
                ix1_y1_, ix1_, iy1_ = p[py - 1, px - 1], p[py, px - 1], p[py - 1, px]
                d = np.array([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)])
                dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
                dxy = 0.5 * (ix1y1 - ix1 - iy1 + 2 * i_ - ix1_ - iy1_ + ix1_y1_)
                hess = np.array([[dxx, dxy], [dxy, dyy]]) + np.finfo(
                    np.float32
                ).eps * np.eye(2)
                out[k] -= np.linalg.inv(hess) @ d
            return out, vals

        rng = np.random.default_rng(0)
        B, K, H, W = 2, 5, 64, 48
        yy, xx = np.mgrid[0:H, 0:W]
        hm = np.zeros((B, H, W, K), np.float32)
        for b in range(B):
            for k in range(K):
                cy, cx = rng.uniform(2, H - 3), rng.uniform(2, W - 3)
                hm[b, ..., k] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
        hm += rng.uniform(0, 0.05, hm.shape).astype(np.float32)
        hm[1, ..., 0] = 0.0  # no response: -1 sentinel, finite

        kps, scores = (
            np.array(a) for a in udp_decode_batch(mx.array(hm), (W - 1, H - 1))
        )
        self.assertTrue(np.isfinite(kps).all())
        self.assertTrue(np.array_equal(kps[1, 0], [-1, -1]))
        for b in range(B):
            ref_kps, ref_scores = reference_decode(hm[b].transpose(2, 0, 1))
            valid = ref_scores > 0
            self.assertTrue(np.allclose(kps[b][valid], ref_kps[valid], atol=1e-3))
            self.assertTrue(np.allclose(scores[b], ref_scores))

    def test_dense_predictor_outputs(self):
        """Dense tasks post-process on device and return MLX arrays at the
        input resolution."""
        image = self._image()
        seg = self._predictor("seg", self._deconv_head(), 5).infer(image)[
            "segmentation"
        ]
        self.assertIsInstance(seg, mx.array)
        self.assertEqual((seg.shape, seg.dtype), ((40, 30), mx.int32))
        self.assertTrue(seg.max() < 5)

        normals = self._predictor("normal", self._pixel_shuffle_head(), 3).infer(image)
        normals = normals["normals"]
        self.assertEqual(normals.shape, (40, 30, 3))
        self.assertTrue(mx.all(mx.linalg.norm(normals, axis=-1) < 1.01))

        matting = self._predictor("matting", self._pixel_shuffle_head(), 4).infer(image)
        self.assertEqual(matting["alphas"].shape, (40, 30))
        self.assertEqual(matting["foregrounds"].shape, (40, 30, 3))
        self.assertTrue(0 <= matting["alphas"].min() and matting["alphas"].max() <= 1)

    def test_pose_predictor_flip_test(self):
        """Pose inference batches the flipped crops; both modes run."""
        predictor = self._predictor("pose", self._deconv_head(), 4, flip_pairs=[[1, 2]])
        boxes = np.array([[0, 0, 29, 39], [5, 5, 25, 35]], np.float32)
        for flip in (True, False):
            out = predictor.infer(self._image(), boxes, flip_test=flip)
            self.assertIsInstance(out["keypoints"], mx.array)
            self.assertEqual(out["keypoints"].shape, (2, 4, 2))
            self.assertEqual(out["scores"].shape, (2, 4))
        # boxes may also be a nested list or an MLX array; None is the full image
        same = predictor.infer(self._image(), boxes.tolist(), flip_test=False)
        self.assertTrue(mx.array_equal(same["keypoints"], out["keypoints"]))
        full = predictor.infer(mx.array(self._image()), None, flip_test=False)
        self.assertEqual(full["boxes"].tolist(), [[0, 0, 29, 39]])

    def test_to_array_accepts_pil_numpy_and_mlx(self):
        """Images enter as PIL, numpy (also flipped views) or MLX arrays,
        without a numpy round trip on the MLX path."""
        from PIL import Image

        from mlx_vlm.models.sapiens2.generate import read_image
        from mlx_vlm.models.sapiens2.image import to_array

        image = self._image()
        want = image.tolist()
        self.assertEqual(to_array(Image.fromarray(image)).tolist(), want)
        self.assertEqual(to_array(image[:, ::-1])[:, ::-1].tolist(), want)
        got = read_image(Image.fromarray(image))
        self.assertEqual((got.shape, got.dtype), ((40, 30, 3), mx.uint8))
        self.assertIs(to_array(got), got)

    def _sync_trap(self):
        """Context that fails on any host <-> device synchronization."""
        from contextlib import ExitStack
        from unittest import mock

        def boom(*args, **kwargs):
            raise AssertionError("host sync during graph construction")

        stack = ExitStack()
        for name in ("eval", "async_eval", "synchronize"):
            stack.enter_context(mock.patch.object(mx, name, boom))
        for name in ("tolist", "item", "__bool__", "__float__", "__int__"):
            if hasattr(mx.array, name):
                stack.enter_context(mock.patch.object(mx.array, name, boom))
        return stack

    def test_infer_builds_graphs_without_host_sync(self):
        """Every task's ``infer`` only builds a graph: no eval, item or
        tolist on the way, including the first call (weight relayouts,
        RoPE / blur tables) and a second one with warm caches."""
        image = self._image()
        cases = [
            ("backbone", None, None, {}),
            ("seg", self._deconv_head(), 5, {}),
            ("normal", self._pixel_shuffle_head(), 3, {}),
            ("matting", self._pixel_shuffle_head(), 4, {}),
            ("pose", self._deconv_head(), 4, dict(flip_pairs=[[1, 2]])),
        ]
        for task, head, labels, kw in cases:
            predictor = self._predictor(task, head, labels, **kw)
            with self._sync_trap():
                first = predictor.infer(image)
                second = predictor.infer(mx.array(image))
            mx.eval(first, second)  # and the graphs are valid

    def test_stream_pipelines_frames_in_order(self):
        """``stream`` yields one output per frame, equal to ``infer`` on that
        frame, for any prefetch depth; pose boxes pair up with the frames."""
        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 255, (40, 30, 3), np.uint8) for _ in range(5)]
        seg = self._predictor("seg", self._deconv_head(), 5)
        want = [seg.infer(f)["segmentation"] for f in frames]
        for prefetch in (0, 1, 3):
            outs = list(seg.stream(iter(frames), prefetch=prefetch))
            self.assertEqual(len(outs), 5)
            got = [o["segmentation"] for o in outs]
            self.assertTrue(all(mx.array_equal(g, w) for g, w in zip(got, want)))
            # the input frame rides along for overlays
            self.assertTrue(mx.array_equal(outs[3]["frame"], mx.array(frames[3])))

        pose = self._predictor("pose", self._deconv_head(), 4, flip_pairs=[[1, 2]])
        boxes = [[[0, 0, 29, 39]], [[5, 5, 25, 35], [0, 0, 29, 39]]]
        outs = list(pose.stream(frames[:2], boxes=boxes, flip_test=False))
        self.assertEqual([o["keypoints"].shape[0] for o in outs], [1, 2])
        with self.assertRaises(ValueError):  # fewer boxes than frames
            list(pose.stream(frames, boxes=boxes))

    def test_read_video_frames(self):
        """Video frames stream in as RGB device arrays (OpenCV decode) and a
        path goes straight into ``stream``."""
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.sapiens2.generate import read_video_frames

        try:
            import cv2
        except ImportError:  # pragma: no cover
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.mp4")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
            )
            for bgr in ((0, 0, 255), (0, 255, 0), (255, 0, 0)):  # R, G, B in RGB
                writer.write(np.full((24, 32, 3), bgr, np.uint8))
            writer.release()
            frames = list(read_video_frames(path))
            if not frames:  # pragma: no cover
                self.skipTest("opencv cannot encode/decode mp4 here")
            self.assertEqual(len(frames), 3)
            self.assertEqual(
                (frames[0].shape, frames[0].dtype), ((24, 32, 3), mx.uint8)
            )
            # lossy codec: check the dominant channel, in RGB order
            dominant = [int(f.mean(axis=(0, 1)).argmax()) for f in frames]
            self.assertEqual(dominant, [0, 1, 2])

            seg = self._predictor("seg", self._deconv_head(), 5)
            outs = list(seg.stream(path))
            self.assertEqual(len(outs), 3)
            self.assertEqual(outs[0]["segmentation"].shape, (24, 32))
