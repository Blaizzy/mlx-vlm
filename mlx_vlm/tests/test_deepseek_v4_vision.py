import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
from PIL import Image

from mlx_vlm.models.deepseek_v4.config import ModelConfig
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


class TestDeepseekV4VisionConfig(unittest.TestCase):
    def test_text_only_config_keeps_vision_disabled(self):
        config = ModelConfig()

        self.assertEqual(config.vision_n_layers, 0)

    def test_published_vision_fields_are_loaded(self):
        config = ModelConfig.from_dict(
            {
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_n_heads": 16,
                "vision_inter_dim": 2816,
                "vision_patch_size": 14,
                "vision_rope_theta": 10000,
                "vision_downsample_ratio": 3,
                "vision_max_n_token": 384,
                "vision_min_pixels": 147456,
                "vision_max_wh_ratio": 8,
            }
        )

        self.assertEqual(config.vision_n_layers, 32)
        self.assertEqual(config.vision_dim, 1024)
        self.assertEqual(config.vision_n_heads, 16)
        self.assertEqual(config.vision_inter_dim, 2816)
        self.assertEqual(config.vision_patch_size, 14)
        self.assertEqual(config.vision_rope_theta, 10000)
        self.assertEqual(config.vision_downsample_ratio, 3)
        self.assertEqual(config.vision_max_n_token, 384)
        self.assertEqual(config.vision_min_pixels, 147456)
        self.assertEqual(config.vision_max_wh_ratio, 8)


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
    def test_vit_uses_official_ffn_parameter_namespace(self):
        parameters = dict(tree_flatten(ViT(tiny_vision_config()).parameters()))

        self.assertIn("blocks.0.ffn.w1.weight", parameters)
        self.assertIn("blocks.0.ffn.w2.weight", parameters)
        self.assertFalse(any(".mlp." in key for key in parameters))

    def test_vit_and_aligner_preserve_published_shapes(self):
        config = tiny_vision_config()
        patches = mx.random.normal((12, 3, 2, 2))

        vision_output = ViT(config)(patches, n_h=3, n_w=4)
        aligned = Aligner(config)(vision_output, n_h=3, n_w=4)
        mx.eval(vision_output, aligned)

        self.assertEqual(vision_output.shape, (12, config.vision_dim))
        self.assertEqual(aligned.shape, (4, config.hidden_size))

    def test_model_replaces_sentinels_after_safe_embedding(self):
        config = tiny_vision_config()
        model = Model(config)
        model.image_start = mx.full((config.hidden_size,), 1.0)
        model.image_pad = mx.full((config.hidden_size,), 2.0)
        model.image_newline = mx.full((config.hidden_size,), 3.0)
        model.image_end = mx.full((config.hidden_size,), 4.0)
        types = mx.array([0, 2, 2, 4], dtype=mx.int32)
        input_ids = mx.array(
            [[1, *(config.vocab_size + types).tolist(), 2]], dtype=mx.int32
        )
        patches = mx.random.normal((8, 3, 2, 2))

        features = model.get_input_embeddings(
            input_ids,
            patches,
            image_grid_hw=mx.array([[2, 4]], dtype=mx.int32),
            image_sample_indices=mx.array([0], dtype=mx.int32),
            image_offsets=mx.array([1], dtype=mx.int32),
            image_types=types,
            image_type_offsets=mx.array([0, 4], dtype=mx.int32),
            image_permutations=mx.array([1, 0], dtype=mx.int32),
        ).inputs_embeds
        expected_images = model.encode_image(patches, 2, 4)[[1, 0]]
        mx.eval(features, expected_images)

        self.assertTrue(mx.allclose(features[0, 1], model.image_start).item())
        self.assertTrue(mx.allclose(features[0, 2:4], expected_images).item())
        self.assertTrue(mx.allclose(features[0, 4], model.image_end).item())
        self.assertTrue(
            mx.allclose(
                features[0, [0, 5]],
                model.language_model.model.embed_tokens(mx.array([1, 2])),
            ).item()
        )

    def test_text_only_model_does_not_construct_vision_modules(self):
        model = Model(tiny_vision_config(vision_n_layers=0))

        self.assertFalse(hasattr(model, "vision"))
        output = model.get_input_embeddings(mx.array([[1, 2]], dtype=mx.int32))
        self.assertEqual(output.inputs_embeds.shape, (1, 2, 8))


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
                    [True, False, False, False, False, False, False, False],
                    [True, True, True, True, True, True, True, False],
                    [True, True, True, True, True, True, True, False],
                    [False, True, True, True, True, True, True, False],
                    [False, True, True, True, True, True, True, False],
                    [False, True, True, True, True, True, True, False],
                    [False, True, True, True, True, True, True, False],
                    [False, False, False, False, False, True, True, True],
                ]
            ],
        )

    def test_hash_gate_uses_score_routing_for_image_tokens(self):
        config = tiny_vision_config(scoring_func="sigmoid")
        gate = MoEGate(config, layer_idx=0)
        gate.weight = mx.zeros_like(gate.weight)
        gate.tid2eid = mx.zeros_like(gate.tid2eid)
        gate.tid2eid[1] = mx.array([0, 1])
        gate.bias_vl = mx.array([0.0, 0.0, 10.0, 9.0])
        hidden = mx.zeros((1, 2, config.hidden_size))
        input_ids = mx.array([[1, config.vocab_size + 2]], dtype=mx.int32)

        indices, weights = gate(hidden, input_ids)
        mx.eval(indices, weights)

        self.assertEqual(indices[0, 0].tolist(), [0, 1])
        self.assertEqual(set(indices[0, 1].tolist()), {2, 3})
        self.assertTrue(
            mx.allclose(
                weights, mx.full(weights.shape, 0.75, dtype=weights.dtype)
            ).item()
        )

    def test_non_hash_gate_selects_text_and_vision_biases(self):
        config = tiny_vision_config(
            num_hash_layers=0,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
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

    def test_text_only_gate_keeps_original_parameters(self):
        config = tiny_vision_config(vision_n_layers=0)
        hash_gate = MoEGate(config, layer_idx=0)
        score_gate = MoEGate(config, layer_idx=config.num_hash_layers)

        self.assertFalse(hasattr(hash_gate, "bias_vl"))
        self.assertFalse(hasattr(hash_gate, "e_score_correction_bias"))
        self.assertTrue(hasattr(score_gate, "e_score_correction_bias"))
        self.assertFalse(hasattr(score_gate, "bias_vl"))

    def test_image_prefill_is_not_chunked(self):
        model = Model(tiny_vision_config())
        text = mx.array([[1, 2]], dtype=mx.int32)
        image = mx.array([[1, model.config.vocab_size, 2]], dtype=mx.int32)

        self.assertTrue(model.language_model.chunked_prefill_policy(input_ids=text))
        self.assertFalse(model.language_model.chunked_prefill_policy(input_ids=image))
        self.assertFalse(
            model.language_model.chunked_prefill_policy(
                input_ids=text, draft_model=object()
            )
        )

    def test_image_sentinels_are_rejected_during_decode(self):
        model = Model(tiny_vision_config())
        cache = model.make_cache()
        prefill = model(mx.array([[1]], dtype=mx.int32), cache=cache)
        mx.eval(prefill.logits)

        with self.assertRaisesRegex(ValueError, "one prefill"):
            model(
                mx.array([[model.config.vocab_size]], dtype=mx.int32),
                cache=cache,
            )

    def test_sanitize_preserves_vision_gate_bias_name(self):
        model = Model(tiny_vision_config())
        weights = model.sanitize(
            {
                "layers.0.ffn.gate.bias": mx.zeros((4,)),
                "layers.0.ffn.gate.bias_vl": mx.ones((4,)),
            }
        )

        self.assertIn(
            "language_model.model.layers.0.ffn.gate.e_score_correction_bias",
            weights,
        )
        self.assertIn(
            "language_model.model.layers.0.ffn.gate.bias_vl",
            weights,
        )

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

    def test_image_prompt_supports_multiple_decode_steps(self):
        processor = TestDeepseekV4ImageProcessor().make_processor(vocab_size=16)
        processed = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        model = Model(tiny_vision_config())
        cache = model.make_cache()
        output = model(
            processed["input_ids"],
            pixel_values=processed["pixel_values"],
            cache=cache,
            **{
                key: value
                for key, value in processed.items()
                if key.startswith("image_")
            },
        )
        for token in (2, 3, 4):
            output = model(mx.array([[token]], dtype=mx.int32), cache=cache)
            mx.eval(output.logits)

        self.assertEqual(output.logits.shape, (1, 1, model.config.vocab_size))

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
            DeepseekV4Processor,
            "check_argument_for_proper_class",
            return_value=None,
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

    def test_build_image_block_matches_n_layout(self):
        types, perm = build_image_block(2, 3, start_pos=0)

        self.assertEqual(
            types.tolist(),
            [1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4],
        )
        self.assertEqual(perm.tolist(), [0, 3, 1, 4, 2, 5])

    def test_openai_content_blocks_become_ordered_placeholders(self):
        processor = self.make_processor()
        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        )

        self.assertEqual(rendered, f"before{IMAGE_PLACEHOLDER}after")

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

    def test_processor_packs_patches_and_expands_sentinels(self):
        processor = self.make_processor()
        image = Image.new("RGB", (8, 4), (255, 0, 127))

        output = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[image],
            return_tensors="mlx",
        )

        self.assertEqual(output["pixel_values"].shape, (8, 3, 2, 2))
        self.assertEqual(output["image_grid_hw"].tolist(), [[2, 4]])
        self.assertEqual(output["image_sample_indices"].tolist(), [0])
        self.assertEqual(output["image_offsets"].tolist(), [1])
        self.assertEqual(output["image_type_offsets"].tolist(), [0, 12])
        self.assertEqual(
            output["image_types"].tolist(),
            [1, 1, 0, 2, 1, 2, 1, 3, 1, 1, 1, 4],
        )
        self.assertEqual(output["image_permutations"].tolist(), [0, 1])
        self.assertEqual(
            output["input_ids"][0, 1:13].tolist(),
            (128 + output["image_types"]).tolist(),
        )
        self.assertLessEqual(float(output["pixel_values"].max().item()), 1.0)
        self.assertGreaterEqual(float(output["pixel_values"].min().item()), -1.0)

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

    def test_portrait_and_landscape_resize_constraints(self):
        for size in ((120, 1600), (1600, 120), (512, 512)):
            with self.subTest(size=size):
                patches, n_h, n_w, n_llm_h, n_llm_w = preprocess_image(
                    Image.new("RGB", size, (0, 0, 0)),
                    patch_size=14,
                    downsample_ratio=3,
                    max_n_token=384,
                    min_pixels=147456,
                    max_wh_ratio=8,
                )
                types, _ = build_image_block(n_llm_h, n_llm_w, start_pos=0)

                if size[0] >= size[1]:
                    self.assertLessEqual(n_w / n_h, 8)
                self.assertLessEqual(len(types), 384)
                self.assertEqual(patches.shape[0], n_h * n_w)

    def test_packed_processor_output_feeds_model_wrapper(self):
        processor = self.make_processor(vocab_size=16)
        output = processor(
            f"before{IMAGE_PLACEHOLDER}after",
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        model = Model(tiny_vision_config())

        features = model.get_input_embeddings(
            output["input_ids"],
            output["pixel_values"],
            **{key: value for key, value in output.items() if key.startswith("image_")},
        )
        mx.eval(features.inputs_embeds)

        self.assertEqual(
            features.inputs_embeds.shape,
            (*output["input_ids"].shape, model.config.hidden_size),
        )

    def test_flat_images_are_allocated_across_batch_samples(self):
        processor = self.make_processor(vocab_size=16)
        image = Image.new("RGB", (8, 4), (20, 40, 60))
        output = processor(
            [
                f"{IMAGE_PLACEHOLDER}after",
                f"before{IMAGE_PLACEHOLDER}",
            ],
            images=[image, image],
            return_tensors="mlx",
        )

        self.assertEqual(output["image_sample_indices"].tolist(), [0, 1])
        for image_idx, sample_idx in enumerate([0, 1]):
            type_start = int(output["image_type_offsets"][image_idx].item())
            type_end = int(output["image_type_offsets"][image_idx + 1].item())
            offset = int(output["image_offsets"][image_idx].item())
            types = output["image_types"][type_start:type_end]
            self.assertEqual(
                output["input_ids"][sample_idx, offset : offset + len(types)].tolist(),
                (16 + types).tolist(),
            )

    def test_multiple_images_preserve_offsets_within_one_prompt(self):
        processor = self.make_processor(vocab_size=16)
        output = processor(
            f"a{IMAGE_PLACEHOLDER}b{IMAGE_PLACEHOLDER}c",
            images=[
                Image.new("RGB", (8, 4), (20, 40, 60)),
                Image.new("RGB", (4, 8), (60, 40, 20)),
            ],
            return_tensors="mlx",
        )

        self.assertEqual(output["image_sample_indices"].tolist(), [0, 0])
        self.assertEqual(len(output["image_offsets"]), 2)
        self.assertLess(
            int(output["image_offsets"][0].item()),
            int(output["image_offsets"][1].item()),
        )
        model = Model(tiny_vision_config())
        features = model.get_input_embeddings(
            output["input_ids"],
            output["pixel_values"],
            **{key: value for key, value in output.items() if key.startswith("image_")},
        )
        mx.eval(features.inputs_embeds)
        self.assertEqual(features.inputs_embeds.shape[:2], output["input_ids"].shape)

    def test_mixed_text_and_image_batch_runs_one_forward(self):
        processor = self.make_processor(vocab_size=16)
        output = processor(
            ["text only", f"before{IMAGE_PLACEHOLDER}after"],
            images=[Image.new("RGB", (8, 4), (20, 40, 60))],
            return_tensors="mlx",
        )
        model = Model(tiny_vision_config())
        result = model(
            output["input_ids"],
            pixel_values=output["pixel_values"],
            **{key: value for key, value in output.items() if key.startswith("image_")},
        )
        mx.eval(result.logits)

        self.assertEqual(
            result.logits.shape,
            (*output["input_ids"].shape, model.config.vocab_size),
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


if __name__ == "__main__":
    unittest.main()
