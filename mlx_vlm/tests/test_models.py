import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):

    def language_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(model.model_type, model_type)
        self.assertEqual(len(model.layers), num_layers)

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs, cache = model(inputs)
            self.assertEqual(outputs.shape, (batch_size, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs, cache = model(
                mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache
            )
            self.assertEqual(outputs.shape, (batch_size, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

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
    ):
        self.assertEqual(vision_tower.model_type, model_type)

        batch_size = 1

        input_tensor = mx.random.uniform(
            shape=(batch_size, image_size[0], image_size[1], num_channels)
        )

        # Perform a forward pass
        *_, hidden_states = vision_tower(input_tensor, output_hidden_states=True)
        # Check the output tensor shape
        self.assertEqual(
            hidden_states[vision_feature_layer][-1][-1].shape, (vision_hidden_size,)
        )

    def test_nanoLlava(self):
        from mlx_vlm.models import nanoLlava

        text_config = nanoLlava.TextConfig(
            model_type="qwen2",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = nanoLlava.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        args = nanoLlava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava-qwen2",
            auto_map={
                "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
            },
            hidden_size=1024,
            mm_hidden_size=1152,
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_projector_type="mlp2x_gelu",
            ignore_index=-100,
            image_token_index=-200,
            vocab_size=151936,
        )

        model = nanoLlava.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.mm_projector,
            args.vision_config.hidden_size,
            args.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower.vision_tower,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )

    def test_llava(self):
        from mlx_vlm.models import llava

        text_config = llava.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        args = llava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            args.vision_config.hidden_size,
            args.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )

    def test_idefics2(self):
        from mlx_vlm.models import idefics2

        text_config = idefics2.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=8,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = idefics2.VisionConfig(
            model_type="idefics2",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=980,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        perceiver_config = idefics2.PerceiverConfig(
            model_type="idefics2Perceiver",
            resampler_n_latents=64,
            resampler_depth=3,
            resampler_n_heads=16,
            resampler_head_dim=96,
            num_key_value_heads=4,
        )

        args = idefics2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            perceiver_config=perceiver_config,
            model_type="idefics2",
            ignore_index=-100,
            image_token_index=32001,
        )

        model = idefics2.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )


if __name__ == "__main__":
    unittest.main()
