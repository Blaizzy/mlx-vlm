"""DeepSeek V4 vision behavior and checkpoint conversion."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest
from PIL import Image

import mlx_vlm.models.deepseek_v4.convert as convert_module
from mlx_vlm.models.deepseek_v4.config import ModelConfig
from mlx_vlm.models.deepseek_v4.convert import (
    configure_parser,
    convert_deepseek_v4_vision,
)
from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model
from mlx_vlm.models.deepseek_v4.language import (
    MoEGate,
    create_image_attention_mask,
    get_image_visible,
)
from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
    IMAGE_PLACEHOLDER,
    DeepseekV4Processor,
    build_image_block,
    preprocess_image,
)
from mlx_vlm.models.deepseek_v4.vision import Aligner, ViT
from mlx_vlm.speculative.drafters.deepseek_v4_dspark import DeepseekV4DsparkDraftModel
from mlx_vlm.speculative.drafters.deepseek_v4_dspark.config import (
    DeepseekV4DsparkConfig,
)
from mlx_vlm.vision_cache import VisionFeatureCache

# Vision behavior


def tiny_vision_config(**kwargs):
    values = {
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "head_dim": 4,
        "o_groups": 2,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 2,
        "index_topk": 2,
        "hc_mult": 2,
        "vision_n_layers": 1,
        "vision_dim": 8,
        "vision_n_heads": 2,
        "vision_inter_dim": 12,
        "vision_patch_size": 2,
        "vision_downsample_ratio": 2,
    }
    values.update(kwargs)
    return ModelConfig(**values)


class TestDeepseekV4VisionTower(unittest.TestCase):
    def test_vit_and_aligner_preserve_published_shapes(self):
        config = tiny_vision_config()
        patches = mx.random.normal((12, 3, 2, 2))

        vision_output = ViT(config)(patches, n_h=3, n_w=4)
        aligned = Aligner(config)(vision_output, n_h=3, n_w=4)
        mx.eval(vision_output, aligned)

        self.assertEqual(vision_output.shape, (12, config.vision_dim))
        self.assertEqual(aligned.shape, (4, config.hidden_size))


class TestDeepseekV4VisionLanguage(unittest.TestCase):
    def test_image_visibility_matches_reference_span_rules(self):
        config = tiny_vision_config(sliding_window=3, vision_max_n_token=16)
        input_ids = mx.array([[1, 16, 17, 18, 19, 18, 20, 2]], dtype=mx.int32)

        left, right = get_image_visible(
            input_ids, config.vocab_size, config.vision_max_n_token
        )
        mask = create_image_attention_mask(
            input_ids,
            config.vocab_size,
            config.sliding_window,
            config.vision_max_n_token,
        )
        mx.eval(left, right, mask)

        self.assertEqual(left.tolist(), [[0, 0, 1, 2, 3, 4, 5, 0]])
        self.assertEqual(right.tolist(), [[0, 5, 4, 3, 2, 1, 0, 0]])
        self.assertEqual(
            mask.tolist(),
            [
                [
                    [
                        [True, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, True, False],
                        [True, True, True, True, True, True, True, False],
                        [False, True, True, True, True, True, True, False],
                        [False, True, True, True, True, True, True, False],
                        [False, True, True, True, True, True, True, False],
                        [False, True, True, True, True, True, True, False],
                        [False, False, False, False, False, True, True, True],
                    ]
                ]
            ],
        )

    def test_mixed_warm_text_and_cold_image_prefill_uses_row_aware_mask(self):
        class MixedCache:
            offset = mx.array([2, 0], dtype=mx.int32)

            def make_mask(self, length, **kwargs):
                del kwargs
                mask = mx.ones((2, 1, length, length + 2), dtype=mx.bool_)
                mask[1, :, :, :2] = False
                return mask

        class CaptureLayer:
            def __call__(self, hidden, mask, cache, input_ids, causal):
                del cache, input_ids
                self.causal = causal
                self.mask = mask
                return hidden

        config = tiny_vision_config(sliding_window=3, vision_max_n_token=16)
        model = Model(config)
        input_ids = mx.array(
            [[1, 2, 3, 4], [1, config.vocab_size, config.vocab_size + 4, 2]],
            dtype=mx.int32,
        )
        safe_ids = mx.where(input_ids < config.vocab_size, input_ids, 0)
        inputs_embeds = model.language_model.model.embed_tokens(safe_ids)
        layer = CaptureLayer()
        model.language_model.model.layers = [layer]

        result = model.language_model(
            input_ids,
            cache=[MixedCache()],
            inputs_embeds=inputs_embeds,
            skip_logits=True,
        )
        mx.eval(result.hidden_states if result.hidden_states is not None else [])

        self.assertFalse(layer.causal)
        self.assertTrue(bool(mx.all(layer.mask[0, :, :, :2]).item()))
        self.assertFalse(bool(mx.any(layer.mask[1, :, :, :2]).item()))
        self.assertTrue(
            mx.array_equal(
                layer.mask[1, :, :, 2:],
                create_image_attention_mask(
                    input_ids,
                    config.vocab_size,
                    config.sliding_window,
                    config.vision_max_n_token,
                )[1],
            ).item()
        )

    def test_non_hash_gate_selects_text_and_vision_biases(self):
        config = tiny_vision_config(
            num_hash_layers=0, scoring_func="sigmoid", routed_scaling_factor=1.0
        )
        gate = MoEGate(config, layer_idx=0)
        gate.weight = mx.zeros_like(gate.weight)
        gate.e_score_correction_bias = mx.array([10.0, 9.0, 0.0, 0.0])
        gate.bias_vl = mx.array([0.0, 0.0, 10.0, 9.0])
        hidden = mx.zeros((1, 2, config.hidden_size))
        input_ids = mx.array([[1, config.vocab_size + 2]], dtype=mx.int32)

        indices, weights = gate(hidden, input_ids)
        mx.eval(indices, weights)

        self.assertEqual(set(indices[0, 0].tolist()), {0, 1})
        self.assertEqual(set(indices[0, 1].tolist()), {2, 3})
        self.assertTrue(
            mx.allclose(
                weights, mx.full(weights.shape, 0.5, dtype=weights.dtype)
            ).item()
        )

    def test_image_prefill_is_not_chunked(self):
        model = Model(tiny_vision_config())
        text = mx.array([[1, 2]], dtype=mx.int32)
        image = mx.array([[1, model.config.vocab_size, 2]], dtype=mx.int32)

        self.assertTrue(model.language_model.chunked_prefill_policy(input_ids=text))
        self.assertFalse(model.language_model.chunked_prefill_policy(input_ids=image))
        self.assertTrue(
            model.language_model.chunked_prefill_policy(
                input_ids=text, draft_model=object()
            )
        )
        self.assertFalse(
            model.language_model.chunked_prefill_policy(
                input_ids=image, draft_model=object()
            )
        )

    def test_image_sentinels_are_rejected_during_decode(self):
        model = Model(tiny_vision_config())
        cache = model.make_cache()
        prefill = model(mx.array([[1]], dtype=mx.int32), cache=cache)
        mx.eval(prefill.logits)

        with self.assertRaisesRegex(ValueError, "one prefill"):
            model(mx.array([[model.config.vocab_size]], dtype=mx.int32), cache=cache)

    def test_sanitize_preserves_vision_gate_bias_name(self):
        model = Model(tiny_vision_config())
        weights = model.sanitize(
            {
                "layers.0.ffn.gate.bias": mx.zeros((4,)),
                "layers.0.ffn.gate.bias_vl": mx.ones((4,)),
            }
        )

        self.assertIn(
            "language_model.model.layers.0.ffn.gate.e_score_correction_bias", weights
        )
        self.assertIn("language_model.model.layers.0.ffn.gate.bias_vl", weights)

    def test_image_prefill_and_text_decode_cover_all_compression_ratios(self):
        processor = TestDeepseekV4ImageProcessor().make_processor(vocab_size=16)
        processed = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        image_kwargs = {
            key: value for key, value in processed.items() if key.startswith("image_")
        }

        for ratio in (0, 4, 128):
            with self.subTest(compress_ratio=ratio):
                model = Model(tiny_vision_config(compress_ratios=[ratio]))
                cache = model.make_cache()
                prefill = model(
                    processed["input_ids"],
                    pixel_values=processed["pixel_values"],
                    cache=cache,
                    **image_kwargs,
                )
                decode = model(mx.array([[2]], dtype=mx.int32), cache=cache)
                mx.eval(prefill.logits, decode.logits)

                self.assertEqual(
                    prefill.logits.shape,
                    (*processed["input_ids"].shape, model.config.vocab_size),
                )
                self.assertEqual(decode.logits.shape, (1, 1, model.config.vocab_size))

    def test_image_prefill_hidden_feeds_text_only_dspark(self):
        processor = TestDeepseekV4ImageProcessor().make_processor(vocab_size=16)
        processed = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        image_kwargs = {
            key: value for key, value in processed.items() if key.startswith("image_")
        }
        target = Model(tiny_vision_config())
        outputs = target(
            processed["input_ids"],
            pixel_values=processed["pixel_values"],
            capture_layer_ids=[0],
            **image_kwargs,
        )
        hidden = mx.concatenate(outputs.hidden_states, axis=-1)
        drafter = DeepseekV4DsparkDraftModel(
            DeepseekV4DsparkConfig(
                text_config=target.config,
                n_mtp_layers=1,
                target_layer_ids=[0],
                mask_token_id=1,
                markov_rank=4,
                block_size=3,
            )
        )
        draft_cache = drafter.reset(target)
        drafts = drafter.draft_block(
            2,
            hidden,
            draft_cache,
            block_size=3,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        mx.eval(drafts)

        self.assertFalse(drafter.stages[0].ffn.gate.vision)
        self.assertEqual(drafts.shape, (1, 2))


class ProcessorTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return 99 if token == IMAGE_PLACEHOLDER else self.unk_token_id

    def encode(self, text, **kwargs):
        del kwargs
        tokens = []
        for index, part in enumerate(text.split(IMAGE_PLACEHOLDER)):
            if part:
                tokens.append(10 + index)
            if index < text.count(IMAGE_PLACEHOLDER):
                tokens.append(99)
        return tokens

    def __call__(self, text, **kwargs):
        del kwargs
        rows = [text] if isinstance(text, str) else text
        return {"input_ids": [self.encode(row) for row in rows]}

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "|".join(message["content"] for message in messages)


class TestDeepseekV4ImageProcessor(unittest.TestCase):
    def make_processor(self, vocab_size=128):
        with patch.object(
            DeepseekV4Processor, "check_argument_for_proper_class", return_value=None
        ):
            return DeepseekV4Processor(
                ProcessorTokenizer(),
                vocab_size=vocab_size,
                vision_patch_size=2,
                vision_downsample_ratio=2,
                vision_max_n_token=64,
                vision_min_pixels=0,
                vision_max_wh_ratio=8,
            )

    def test_prompt_helper_preserves_openai_image_position(self):
        from mlx_vlm.prompt_utils import apply_chat_template

        processor = self.make_processor()
        rendered = apply_chat_template(
            processor,
            {"model_type": "deepseek_v4"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "after"},
                ],
            },
            num_images=1,
        )

        self.assertEqual(rendered, f"before{IMAGE_PLACEHOLDER}after")

    def test_extreme_landscape_respects_token_budget(self):
        image = Image.new("RGB", (4000, 100), (0, 0, 0))
        patches, _, _, n_llm_h, n_llm_w = preprocess_image(
            image,
            patch_size=14,
            downsample_ratio=3,
            max_n_token=384,
            min_pixels=147456,
            max_wh_ratio=8,
        )
        types, _ = build_image_block(n_llm_h, n_llm_w, start_pos=0)

        self.assertLessEqual(len(types), 384)
        self.assertEqual(patches.shape[1:], (3, 14, 14))

    def test_mixed_text_and_image_batch_runs_one_forward(self):
        processor = self.make_processor(vocab_size=16)
        output = processor(
            ["text only", f"before{IMAGE_PLACEHOLDER}after"],
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        # Keep batch size different from the attention-head count so a missing
        # singleton head axis cannot broadcast accidentally.
        model = Model(tiny_vision_config(num_attention_heads=4))
        result = model(
            output["input_ids"],
            pixel_values=output["pixel_values"],
            **{key: value for key, value in output.items() if key.startswith("image_")},
        )
        mx.eval(result.logits)

        self.assertEqual(
            result.logits.shape, (*output["input_ids"].shape, model.config.vocab_size)
        )

    def test_cached_packed_features_skip_vision_tower(self):
        processor = self.make_processor(vocab_size=16)
        output = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        model = Model(tiny_vision_config())
        cached = model.encode_images(
            output["pixel_values"],
            image_grid_hw=output["image_grid_hw"],
            image_permutations=output["image_permutations"],
        )
        mx.eval(*cached)

        with patch.object(
            model, "_encode_images", side_effect=AssertionError("cache miss")
        ):
            features = model.get_input_embeddings(
                output["input_ids"],
                output["pixel_values"],
                cached_image_features=cached,
                **{
                    key: value
                    for key, value in output.items()
                    if key.startswith("image_")
                },
            )
            mx.eval(features.inputs_embeds)

        self.assertEqual(features.inputs_embeds.shape[:2], output["input_ids"].shape)

    def test_server_vision_cache_reuses_packed_features(self):
        processor = self.make_processor(vocab_size=16)
        image = Image.new("RGB", (8, 4), (20, 40, 60))
        output = processor(
            f"before{IMAGE_PLACEHOLDER}after", images=[image], return_tensors="mlx"
        )
        model = Model(tiny_vision_config())
        vision_cache = VisionFeatureCache()
        image_kwargs = {
            key: value for key, value in output.items() if key.startswith("image_")
        }

        with patch.object(
            model, "_encode_images", wraps=model._encode_images
        ) as encode:
            first = model.get_input_embeddings(
                output["input_ids"],
                output["pixel_values"],
                vision_cache=vision_cache,
                _image_key=image,
                **image_kwargs,
            )
            second = model.get_input_embeddings(
                output["input_ids"],
                output["pixel_values"],
                vision_cache=vision_cache,
                _image_key=image,
                **image_kwargs,
            )
            mx.eval(first.inputs_embeds, second.inputs_embeds)

        self.assertEqual(encode.call_count, 1)
        self.assertEqual(len(vision_cache), 1)
        self.assertTrue(
            mx.array_equal(first.inputs_embeds, second.inputs_embeds).item()
        )


# Checkpoint conversion


def _write_shard(path, weights):
    mx.save_safetensors(str(path), weights, metadata={"format": "pt"})
    return path.name


def _make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "compress_ratios": [0],
        "o_groups": 2,
        "o_lora_rank": 2,
        "vision_n_layers": 1,
        "vision_dim": 4,
        "vision_n_heads": 1,
        "vision_inter_dim": 8,
        "vision_patch_size": 2,
        "quantization_config": {"quant_method": "fp8"},
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("{}")

    shard1 = {
        "model.embed_tokens.weight": mx.zeros((32, 32), dtype=mx.bfloat16),
        "model.vision.patch_embed.proj.weight": mx.ones((4, 12), dtype=mx.bfloat16),
        "model.image_start": mx.ones((32,), dtype=mx.bfloat16),
    }
    shard2 = {
        "model.layers.0.self_attn.wq_a.weight": mx.zeros((128, 128), dtype=mx.uint8),
        "model.layers.0.self_attn.wq_a.weight_scale_inv": mx.ones(
            (1, 1), dtype=mx.uint8
        ),
    }
    projection_names = ("gate_proj", "down_proj", "up_proj")
    for expert, shard in ((0, shard1), (1, shard2)):
        for projection in projection_names:
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            shard[f"{prefix}.weight"] = mx.zeros((32, 16), dtype=mx.int8)
            shard[f"{prefix}.weight_scale_inv"] = mx.ones((32, 1), dtype=mx.uint8)
    mtp = {"mtp.layers.0.weight": mx.ones((2, 2), dtype=mx.bfloat16)}

    shard_names = [
        _write_shard(source / "model-00001-of-00003.safetensors", shard1),
        _write_shard(source / "model-00002-of-00003.safetensors", shard2),
        _write_shard(source / "model-00003-of-00003.safetensors", mtp),
    ]
    weight_map = {}
    for shard_name, weights in zip(shard_names, (shard1, shard2, mtp)):
        weight_map.update({key: shard_name for key in weights})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123456}, "weight_map": weight_map})
    )
    return source


def test_streaming_converter_maps_and_packs_mixed_checkpoint(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "output"

    convert_deepseek_v4_vision(source, output)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["source_total_size"] == 123456
    assert len(set(index["weight_map"].values())) == 1
    assert not any(key.startswith("mtp.") for key in index["weight_map"])
    weights = mx.load(str(next(output.glob("model-*.safetensors"))))

    assert "vision.patch_embed.proj.weight" in weights
    assert weights["vision.patch_embed.proj.weight"].dtype == mx.bfloat16
    assert "language_model.model.embed_tokens.weight" in weights
    expert_prefix = "language_model.model.layers.0.ffn.switch_mlp"
    for projection in ("gate_proj", "down_proj", "up_proj"):
        assert weights[f"{expert_prefix}.{projection}.weight"].shape == (2, 32, 4)
        assert weights[f"{expert_prefix}.{projection}.scales"].shape == (2, 32, 1)
    attention = "language_model.model.layers.0.attn.wq_a"
    assert weights[f"{attention}.weight"].shape == (128, 32)
    assert weights[f"{attention}.scales"].shape == (128, 4)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"][f"{expert_prefix}.gate_proj"] == {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }
    assert config["quantization"][attention] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert (output / "tokenizer.json").exists()


def test_streaming_converter_refuses_nonempty_output(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep")

    try:
        convert_deepseek_v4_vision(source, output)
    except ValueError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("Expected non-empty output to be rejected")


def test_dedicated_converter_resolves_source_and_extracts_dspark(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    output = tmp_path / "output"
    drafter = tmp_path / "drafter"
    calls = {}

    monkeypatch.setattr(
        convert_module,
        "get_model_path",
        lambda model, revision=None: calls.update(model=model, revision=revision)
        or source,
    )

    def fake_convert(model_path, output_path):
        calls.update(model_path=model_path, output_path=Path(output_path))
        Path(output_path).mkdir()
        return Path(output_path)

    class Splitter:
        def split(self, model_path, output_path):
            calls.update(dspark_source=model_path, dspark_output=output_path)

    from mlx_vlm.speculative.drafters import mtp_split

    monkeypatch.setattr(convert_module, "convert_deepseek_v4_vision", fake_convert)
    monkeypatch.setattr(mtp_split, "detect_mtp_splitter", lambda path: Splitter())
    monkeypatch.setattr(
        convert_module,
        "create_model_card",
        lambda path, source_id: calls.update(card=(path, source_id)),
    )
    monkeypatch.setattr(
        convert_module,
        "upload_to_hub",
        lambda path, repo: calls.update(upload=(path, repo)),
    )

    result = convert_module.convert(
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        output,
        revision="test-revision",
        mtp=True,
        mtp_output=drafter,
        upload_repo="mlx-community/test-model",
    )

    assert result == output
    assert calls["model"] == "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    assert calls["revision"] == "test-revision"
    assert calls["model_path"] == source
    assert calls["output_path"] == output
    assert calls["dspark_source"] == str(source)
    assert calls["dspark_output"] == str(drafter)
    assert calls["card"] == (output, "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
    assert calls["upload"] == (output, "mlx-community/test-model")


def test_dedicated_converter_parser_uses_model_specific_options():
    args = configure_parser().parse_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            "--mlx-path",
            "converted",
            "--mtp",
        ]
    )

    assert args.model == "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    assert args.output_path == "converted"
    assert args.mtp is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
