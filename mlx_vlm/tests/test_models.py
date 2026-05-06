import inspect
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


class TestModels(unittest.TestCase):
    def language_test_runner(self, model, model_type, vocab_size, num_layers):
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
        for t in [mx.float32, mx.float16]:
            vision_tower.update(
                tree_map(lambda p: p.astype(t), vision_tower.parameters())
            )
            self.assertEqual(vision_tower.model_type, model_type)

            if model_type == "llama4_vision_model":
                vision_hidden_size = kwargs.pop(
                    "projector_output_dim", vision_hidden_size
                )

            batch_size = kwargs.pop("batch_size", 1)
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
                if model_type == "llama4_vision_model":
                    self.assertEqual(hidden_states.shape[1], vision_hidden_size)
                else:
                    self.assertEqual(hidden_states.shape[1], vision_hidden_size)
            else:
                self.assertEqual(hidden_states.shape[-1], vision_hidden_size)

            self.assertEqual(hidden_states.dtype, t)

    def test_llava_bunny(self):
        from mlx_vlm.models import llava_bunny

        text_config = llava_bunny.TextConfig(
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

        vision_config = llava_bunny.VisionConfig(
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

        config = llava_bunny.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava-qwen2",
            auto_map={
                "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
            },
            hidden_size=1024,
            mm_hidden_size=1152,
            mm_projector_type="mlp2x_gelu",
            ignore_index=-100,
            image_token_index=-200,
            vocab_size=151936,
        )

        model = llava_bunny.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.mm_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llava_next(self):
        from mlx_vlm.models import llava_next

        text_config = llava_next.TextConfig(
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

        vision_config = llava_next.VisionConfig(
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

        config = llava_next.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava_next.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
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

        config = llava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
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

        config = idefics2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            perceiver_config=perceiver_config,
            model_type="idefics2",
            ignore_index=-100,
            image_token_index=32001,
        )

        model = idefics2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_idefics3(self):
        from mlx_vlm.models import idefics3

        text_config = idefics3.TextConfig(
            model_type="idefics3",
            hidden_size=2048,
            num_hidden_layers=24,
            intermediate_size=8192,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=49155,
            num_key_value_heads=8,
            rope_theta=273768.0,
            rope_traditional=False,
        )

        vision_config = idefics3.VisionConfig(
            model_type="idefics3",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = idefics3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="idefics3",
            ignore_index=-100,
            image_token_id=49153,
        )

        model = idefics3.Model(config)

        head_dim = (
            config.text_config.hidden_size // config.text_config.num_attention_heads
        )
        expected_kv_width = config.text_config.num_key_value_heads * head_dim
        self.assertEqual(
            model.language_model.layers[0].self_attn.k_proj.weight.shape,
            (expected_kv_width, config.text_config.hidden_size),
        )
        self.assertEqual(
            model.language_model.layers[0].self_attn.v_proj.weight.shape,
            (expected_kv_width, config.text_config.hidden_size),
        )

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_smolvlm_text_config_infers_heads_from_head_dim(self):
        from mlx_vlm.models import smolvlm

        text_config = smolvlm.TextConfig.from_dict(
            {
                "hidden_size": 2048,
                "head_dim": 64,
                "intermediate_size": 8192,
                "num_hidden_layers": 24,
                "rms_norm_eps": 1e-5,
                "vocab_size": 49280,
            }
        )

        self.assertEqual(text_config.num_attention_heads, 32)
        self.assertEqual(text_config.num_key_value_heads, 32)

    def test_smolvlm_vision_config_infers_500m_defaults(self):
        from mlx_vlm.models import smolvlm

        vision_config = smolvlm.VisionConfig.from_dict(
            {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "patch_size": 16,
                "image_size": 512,
                "model_type": "smolvlm_vision",
            }
        )

        self.assertEqual(vision_config.num_hidden_layers, 12)
        self.assertEqual(vision_config.intermediate_size, 3072)

    def test_internvl_chat(self):
        from mlx_vlm.models import internvl_chat

        test_config = internvl_chat.TextConfig(
            model_type="qwen2",
            hidden_size=3584,
            num_hidden_layers=5,
            intermediate_size=18944,
            num_attention_heads=28,
            rms_norm_eps=1e-6,
            max_window_layers=16,
            use_sliding_window=False,
            vocab_size=151674,
            num_key_value_heads=4,
            rope_theta=1000000.0,
            rope_scaling={"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"},
            hidden_act="silu",
            max_position_embeddings=32768,
        )

        vision_config = internvl_chat.VisionConfig(
            model_type="intern_vit_6b",
            num_hidden_layers=5,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = internvl_chat.ModelConfig(
            text_config=test_config,
            vision_config=vision_config,
            model_type="internvl_chat",
            ignore_index=-100,
            image_token_index=151667,
        )

        model = internvl_chat.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_paligemma(self):
        from mlx_vlm.models import paligemma

        text_config = paligemma.TextConfig(
            model_type="gemma",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
            num_key_value_heads=1,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = paligemma.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=2048,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = paligemma.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="paligemma",
            ignore_index=-100,
            image_token_index=257152,
            hidden_size=2048,
            vocab_size=257216,
        )

        model = paligemma.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_paligemma_from_dict_defaults_bidirectional_attention(self):
        from mlx_vlm.models import paligemma

        config = paligemma.ModelConfig.from_dict(
            {
                "model_type": "paligemma",
                "hidden_size": 2048,
                "projection_dim": 2048,
                "text_config": {
                    "model_type": "gemma2",
                    "hidden_size": 2048,
                    "num_hidden_layers": 2,
                    "intermediate_size": 8192,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "vocab_size": 256000,
                    "head_dim": 256,
                    "query_pre_attn_scalar": 256,
                    "attn_logit_softcapping": 50.0,
                    "final_logit_softcapping": 30.0,
                    "hidden_act": "gelu_pytorch_tanh",
                },
                "vision_config": {
                    "model_type": "siglip_vision_model",
                    "num_hidden_layers": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_attention_heads": 16,
                    "image_size": 896,
                    "patch_size": 14,
                },
            }
        )

        self.assertTrue(config.text_config.use_bidirectional_attention)
        self.assertEqual(config.text_config.hidden_activation, "gelu_pytorch_tanh")
        self.assertEqual(config.text_config.num_image_tokens, 4096)
        self.assertEqual(config.vision_config.projection_dim, 2048)

    def test_multi_modality(self):
        from mlx_vlm.models import multi_modality

        text_config = multi_modality.TextConfig(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=24,
            intermediate_size=5632,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = multi_modality.VisionConfig(
            model_type="vision",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-5,
            params={},
        )

        projector_config = multi_modality.ProjectorConfig(
            cls="MlpProjector",
            model_type="projector",
            params={
                "depth": 2,
                "input_dim": 1024,
                "n_embed": 2048,
                "projector_type": "mlp_gelu",
            },
        )

        config = multi_modality.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="multi_modality",
            ignore_index=-100,
            image_token_index=100015,
            vocab_size=32000,
        )

        model = multi_modality.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.aligner,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_phi3_v(self):
        from mlx_vlm.models import phi3_v

        text_config = phi3_v.TextConfig()

        vision_config = phi3_v.VisionConfig(
            model_type="phi3_v",
            image_dim_out=1024,
            model_name="openai/clip-vit-large-patch14-336",
            name="clip_vision_model",
            num_img_tokens=144,
        )

        # Use smaller model dimensions for CI memory constraints
        config = phi3_v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            **{
                "hidden_size": 256,
                "intermediate_size": 512,
                "max_position_embeddings": 4096,
                "model_type": "phi3_v",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "original_max_position_embeddings": 4096,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "long_factor": [1.0] * 32,
                    "short_factor": [1.0] * 32,
                    "type": "su",
                },
                "rope_theta": 10000.0,
                "vocab_size": 1000,
            },
        )

        model = phi3_v.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_mistral3(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            head_dim=128,
            hidden_size=5120,
            intermediate_size=32768,
            max_position_embeddings=131072,
            model_type="mistral",
            num_attention_heads=32,
            num_hidden_layers=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            rope_theta=1000000000.0,
            vocab_size=131072,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=False,
            layer_types=["full_attention"] * 40,
            use_qk_norm=False,
        )

        vision_config = mistral3.VisionConfig(
            model_type="pixtral",
            hidden_size=1024,
            num_hidden_layers=24,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            rms_norm_eps=1e-6,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mistral3",
        )

        model = mistral3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_ministral3(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            head_dim=128,
            hidden_size=3072,
            intermediate_size=9216,
            max_position_embeddings=262144,
            model_type="ministral3",
            num_attention_heads=32,
            num_hidden_layers=26,
            rms_norm_eps=1e-05,
            rope_parameters={
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 16.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "type": "yarn",
            },
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=True,
            vocab_size=131072,
        )

        vision_config = mistral3.VisionConfig(
            head_dim=64,
            hidden_size=1024,
            image_size=1540,
            intermediate_size=4096,
            model_type="pixtral",
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=24,
            patch_size=14,
            rope_theta=10000.0,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="pixtral",
        )
        model = mistral3.Model(config)
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )
        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_pixtral(self):
        from mlx_vlm.models import pixtral

        text_config = pixtral.TextConfig(
            model_type="mistral",
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

        vision_config = pixtral.VisionConfig(
            model_type="pixtral",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            rms_norm_eps=1e-6,
        )

        config = pixtral.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="pixtral",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = pixtral.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        pixel_values = mx.random.uniform(shape=(2, 56, 56, 3))
        image_sizes = mx.array([[28, 42], [56, 56]])

        full_hidden, _ = model.vision_tower(pixel_values, output_hidden_states=True)
        sized_hidden, _ = model.vision_tower(
            pixel_values, output_hidden_states=True, image_sizes=image_sizes
        )

        expected_full_tokens = 2 * (56 // 14) * (56 // 14)
        expected_sized_tokens = (28 // 14) * (42 // 14) + (56 // 14) * (56 // 14)

        self.assertEqual(full_hidden.shape[1], expected_full_tokens)
        self.assertEqual(sized_hidden.shape[1], expected_sized_tokens)

    def test_qwen2_vl(self):
        from mlx_vlm.models import qwen2_vl

        text_config = qwen2_vl.TextConfig(
            model_type="qwen2_vl",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=37,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=152064,
            num_key_value_heads=4,
            max_position_embeddings=512,
            rope_theta=10000,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
            tie_word_embeddings=False,
        )

        vision_config = qwen2_vl.VisionConfig(
            model_type="qwen2_vl",
            depth=2,
            embed_dim=32,
            hidden_size=32,
            image_size=224,
            num_heads=4,
            patch_size=14,
            mlp_ratio=4,
            in_channels=3,
            spatial_merge_size=1,
            temporal_patch_size=2,
        )

        config = qwen2_vl.ModelConfig(
            model_type="qwen2_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=151655,
            vocab_size=32000,
        )

        model = qwen2_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.in_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=-1,
            grid_thw=mx.ones((1, 3)),  # image temporals shape (num_images, 3)
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )

    def test_qwen2_5_vl(self):
        from mlx_vlm.models import qwen2_5_vl

        text_config = qwen2_5_vl.TextConfig(
            model_type="qwen2_5_vl",
            hidden_size=1280,
            num_hidden_layers=32,
            intermediate_size=3420,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=16,
            max_position_embeddings=128000,
            rope_theta=1000000.0,
            rope_traditional=False,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
            tie_word_embeddings=True,
        )

        vision_config = qwen2_5_vl.VisionConfig(
            model_type="qwen2_5_vl",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            image_size=384,
            vocab_size=32000,
            mlp_ratio=4.0,
            in_channels=3,
            layer_norm_eps=1e-6,
            spatial_patch_size=14,
            spatial_merge_size=2,
            tokens_per_second=2,
            temporal_patch_size=2,
            window_size=112,
            patch_size=14,
            fullatt_block_indexes=[7, 15, 23, 31],
        )

        config = qwen2_5_vl.ModelConfig(
            model_type="qwen2_5_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=32000,
        )

        model = qwen2_5_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )

    def test_dots_ocr(self):
        from mlx_vlm.models import dots_ocr

        text_config = dots_ocr.TextConfig(
            model_type="dots_ocr",
            vocab_size=256,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            attention_bias=True,
            tie_word_embeddings=False,
        )

        vision_config = dots_ocr.VisionConfig(
            model_type="dots_vit",
            embed_dim=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=1,
            use_bias=False,
        )

        config = dots_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="dots_ocr",
            image_token_id=10,
            video_token_id=11,
            vocab_size=256,
        )

        model = dots_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        pixel_values = mx.random.uniform(shape=(4, 3 * 14 * 14), dtype=mx.float32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        vision_features = model.vision_tower(pixel_values, image_grid_thw)
        self.assertEqual(vision_features.shape, (1, 64))

        input_ids = mx.array([[1, config.image_token_id, 2]], dtype=mx.int32)
        embeddings = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, 64))

    def test_qwen3_vl(self):
        from mlx_vlm.models import qwen3_vl

        text_config = qwen3_vl.TextConfig(
            model_type="qwen3_vl_text",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            head_dim=32,
            vocab_size=10_000,
            rope_theta=1000,
            max_position_embeddings=1000,
            tie_word_embeddings=False,
            norm_topk_prob=True,
            rope_scaling={"rope_type": "mrope", "mrope_section": [8, 6, 6]},
        )

        vision_config = qwen3_vl.VisionConfig(
            model_type="qwen3_vl",
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            out_hidden_size=128,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=144,
            deepstack_visual_indexes=[],
        )

        config = qwen3_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="qwen3_vl",
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=10_000,
        )

        model = qwen3_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Test vision model with proper input format
        # Input shape: (total_patches, channels, temporal_patch_size, patch_size, patch_size)
        # For grid_thw = [1, 28, 28], we have 1*28*28 = 784 patches
        grid_thw = mx.array([[1, 28, 28]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor
        pixel_values = mx.random.uniform(
            shape=(
                num_patches,
                config.vision_config.in_channels,
                config.vision_config.temporal_patch_size,
                config.vision_config.patch_size,
                config.vision_config.patch_size,
            )
        )

        # Forward pass
        hidden_states, _ = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have 28/2 * 28/2 = 14*14 = 196 patches
        expected_patches = (
            grid_thw[0, 1] // config.vision_config.spatial_merge_size
        ) * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

        # Multi-image batch: per-sample slicing in `_deepstack_process` must
        # avoid the (N,D)/(M,D) broadcast crash and write through to output.
        self._run_deepstack_multi_image_assertions(
            model.language_model.model._deepstack_process
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )

    def test_qwen3_5_model_config(self):
        from mlx_vlm.models import qwen3_5, qwen3_5_moe

        quantization = {
            "group_size": 128,
            "bits": 4,
            "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                "group_size": 128,
                "bits": 6,
            },
            "model.visual.blocks.0.attn.qkv": False,
            "lm_head": False,
        }

        for model_module in (qwen3_5, qwen3_5_moe):
            with self.subTest(model_type=model_module.__name__):
                config = model_module.ModelConfig.from_dict(
                    {
                        "model_type": model_module.__name__.rsplit(".", 1)[-1],
                        "text_config": {},
                        "vision_config": {},
                        "quantization": quantization,
                        "quantization_config": quantization,
                    }
                )

                self.assertIn(
                    "language_model.model.layers.0.linear_attn.in_proj_qkv",
                    config.quantization,
                )
                self.assertEqual(
                    config.quantization[
                        "language_model.model.layers.0.linear_attn.in_proj_qkv"
                    ],
                    {"group_size": 128, "bits": 6},
                )
                self.assertNotIn(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv",
                    config.quantization,
                )
                self.assertIn("vision_tower.blocks.0.attn.qkv", config.quantization)
                self.assertIn("language_model.lm_head", config.quantization)
                self.assertIs(config.quantization, config.quantization_config)

    def test_qwen3_vl_moe(self):
        from mlx_vlm.models import qwen3_vl_moe

        text_config = qwen3_vl_moe.TextConfig(
            model_type="qwen3_vl_moe_text",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            head_dim=32,
            vocab_size=10_000,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            num_experts_per_tok=2,
            num_experts=4,
            moe_intermediate_size=128,
            rope_theta=1000,
            max_position_embeddings=1000,
            tie_word_embeddings=False,
            norm_topk_prob=True,
            rope_scaling={"rope_type": "mrope", "mrope_section": [8, 6, 6]},
        )

        vision_config = qwen3_vl_moe.VisionConfig(
            model_type="qwen3_vl_moe",
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            out_hidden_size=128,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=144,
            deepstack_visual_indexes=[],
        )

        config = qwen3_vl_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="qwen3_vl_moe",
            image_token_id=151655,
            video_token_id=151656,
            vocab_size=10_000,
        )

        model = qwen3_vl_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Test vision model with proper input format
        # Input shape: (total_patches, channels, temporal_patch_size, patch_size, patch_size)
        # For grid_thw = [1, 28, 28], we have 1*28*28 = 784 patches
        grid_thw = mx.array([[1, 28, 28]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor
        pixel_values = mx.random.uniform(
            shape=(
                num_patches,
                config.vision_config.in_channels,
                config.vision_config.temporal_patch_size,
                config.vision_config.patch_size,
                config.vision_config.patch_size,
            )
        )

        # Forward pass
        hidden_states, _ = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have 28/2 * 28/2 = 14*14 = 196 patches
        expected_patches = (
            grid_thw[0, 1] // config.vision_config.spatial_merge_size
        ) * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

        # Multi-image batch
        self._run_deepstack_multi_image_assertions(
            model.language_model.model._deepstack_process
        )

        # Decode-step RoPE offset must come from cache._idx, not offset.item().
        self._assert_mrope_decode_uses_cache_idx(
            model.language_model, config.text_config.hidden_size
        )

    def _run_deepstack_multi_image_assertions(self, deepstack_fn):
        """Shared assertions for qwen3_vl / qwen3_vl_moe `_deepstack_process`.

        Exercises the multi-image batch path: sample 0 has 2 visual tokens,
        sample 1 has 3 visual tokens. Pre-PR-1055 this crashed with a
        ``Shapes (N,D)/(M,D) cannot be broadcast`` because every sample saw
        the full (5,H) ``visual_embeds`` instead of its per-sample slice.
        """
        H = 4
        # hidden_states: distinct nonzero values per (batch, position) so
        # we can check element-wise where the scatter-add landed.
        base = mx.arange(2 * 6 * H, dtype=mx.float32).reshape(2, 6, H)
        # mask sample 0 -> visuals at rows {1, 3} (2 visuals)
        # mask sample 1 -> visuals at rows {0, 2, 4} (3 visuals)
        visual_pos_masks = mx.array(
            [
                [False, True, False, True, False, False],
                [True, False, True, False, True, False],
            ]
        )
        # 5 distinct visual embed rows (2 for sample 0 + 3 for sample 1)
        visual_embeds = mx.arange(5 * H, dtype=mx.float32).reshape(5, H) + 100.0

        out = deepstack_fn(base, visual_pos_masks, visual_embeds)
        self.assertEqual(out.shape, base.shape)

        out_l = out.tolist()
        base_l = base.tolist()
        emb_l = visual_embeds.tolist()

        # Sample 0: rows 1 and 3 received visual_embeds[0] and [1]
        self.assertEqual(
            out_l[0][1],
            [base_l[0][1][i] + emb_l[0][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[0][3],
            [base_l[0][3][i] + emb_l[1][i] for i in range(H)],
        )
        # Sample 0: untouched rows
        for r in (0, 2, 4, 5):
            self.assertEqual(out_l[0][r], base_l[0][r])

        # Sample 1: rows 0, 2, 4 received visual_embeds[2], [3], [4]
        self.assertEqual(
            out_l[1][0],
            [base_l[1][0][i] + emb_l[2][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[1][2],
            [base_l[1][2][i] + emb_l[3][i] for i in range(H)],
        )
        self.assertEqual(
            out_l[1][4],
            [base_l[1][4][i] + emb_l[4][i] for i in range(H)],
        )
        # Sample 1: untouched rows
        for r in (1, 3, 5):
            self.assertEqual(out_l[1][r], base_l[1][r])

        # Empty-mask sample passes through unchanged.
        empty_masks = mx.array([[False, False, False]])
        empty_hidden = mx.ones((1, 3, H))
        empty_out = deepstack_fn(empty_hidden, empty_masks, mx.zeros((0, H)))
        self.assertEqual(empty_out.tolist(), empty_hidden.tolist())

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
        # MRoPE shape: (3, batch, seq).
        self.assertEqual(tuple(position_ids.shape), (3, 1, 1))
        # Decode position == cache._idx (10), not cache.offset[0].item() (3).
        self.assertEqual(position_ids[0, 0, 0].item(), 10)

    def test_glm4v_moe(self):
        from mlx_vlm.models import glm4v_moe

        text_config = glm4v_moe.TextConfig(
            model_type="glm4v_moe",
            vocab_size=257152,
            hidden_size=4096,
            intermediate_size=10944,
            max_position_embeddings=8192,
            moe_intermediate_size=13696,
            norm_topk_prob=False,
            num_attention_heads=32,
            n_group=1,
            head_dim=128,
            topk_group=1,
            n_shared_experts=2,
            n_routed_experts=16,
            routed_scaling_factor=1.0,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            num_hidden_layers=5,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_theta=5000000,
            use_qk_norm=True,
            attention_bias=False,
            partial_rotary_factor=0.5,
            rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12]},
            tie_word_embeddings=False,
        )

        vision_config = glm4v_moe.VisionConfig(
            model_type="glm4v_moe",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            patch_size=14,
            window_size=112,
            image_size=336,
            in_channels=3,
            rms_norm_eps=1e-05,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            initializer_range=0.02,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        config = glm4v_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v_moe",
            vocab_size=257152,
            ignore_index=-100,
            image_token_index=151363,
            image_token_id=151363,
            video_token_index=151364,
            video_token_id=151364,
            vision_start_token_id=151339,
            vision_end_token_id=151340,
            hidden_size=2048,
            pad_token_id=0,
        )

        model = glm4v_moe.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_glm4v(self):
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v",
        )

        vision_config = glm4v.VisionConfig(
            model_type="glm4v",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            out_hidden_size=1536,
            num_heads=16,
            patch_size=14,
            window_size=112,
            image_size=336,
            in_channels=3,
            rms_norm_eps=1e-05,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            initializer_range=0.02,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        config = glm4v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v",
            vocab_size=257152,
            ignore_index=-100,
            image_token_index=151363,
            image_token_id=151363,
            video_token_index=151364,
            video_token_id=151364,
            vision_start_token_id=151339,
            vision_end_token_id=151340,
            hidden_size=2048,
            pad_token_id=0,
        )

        model = glm4v.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.in_channels,
            (140, 1176),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 10, 14]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_lfm2_vl(self):
        from mlx_vlm.models import lfm2_vl

        text_config = lfm2_vl.TextConfig(
            layer_types=[
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ],
        )
        vision_config = lfm2_vl.VisionConfig()
        config = lfm2_vl.ModelConfig(
            text_config=text_config, vision_config=vision_config
        )
        model = lfm2_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # TODO: Add vision test runner for lfm2_vl
        # Rewrite inputs to be defined by the test classes

    def test_mllama(self):
        from mlx_vlm.models import mllama

        vision_config = mllama.VisionConfig(
            image_size=50,
            patch_size=14,
            num_channels=3,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=10,
            num_attention_heads=16,
            max_num_tiles=4,
            max_aspect_ratio_id=8,
            num_global_layers=8,
            norm_eps=1e-5,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            vision_output_dim=7680,
            intermediate_layers_indices=[3, 7, 15, 23, 30],
        )

        text_config = mllama.TextConfig(
            model_type="mllama",
            hidden_size=4096,
            num_hidden_layers=10,
            intermediate_size=14336,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
        )

        model_config = mllama.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mllama",
            ignore_index=-100,
            image_token_index=128256,
            vision_feature_select_strategy="default",
            vision_feature_layer=-2,
            vocab_size=32000,
        )

        # Create the model
        model = mllama.Model(model_config)

        # Create dummy input data
        batch_size = 1
        seq_length = 5
        num_tiles = 4
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        pixel_values = mx.random.normal((batch_size, 1, num_tiles, 3, 50, 50))
        mask = mx.ones((batch_size, seq_length))
        aspect_ratio_ids = mx.zeros((batch_size, 1), dtype=mx.int32)
        aspect_ratio_mask = mx.ones((batch_size, 1, num_tiles))
        cross_attention_mask = mx.ones((batch_size, seq_length, 1, num_tiles))

        # Forward pass
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
        )

        # Check output shape
        expected_shape = (batch_size, seq_length, model_config.vocab_size)
        self.assertEqual(output.logits.shape, expected_shape)

    def test_molmo(self):
        from mlx_vlm.models import molmo

        text_config = molmo.TextConfig()
        vision_config = molmo.VisionConfig()
        config = molmo.ModelConfig(text_config=text_config, vision_config=vision_config)
        model = molmo.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.n_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.d_model,
            config.vision_config.num_channels,
            (576, 588),
        )

    def test_molmo2(self):
        from mlx_vlm.models import molmo2

        text_config = molmo2.TextConfig(
            model_type="molmo2",
            hidden_size=256,  # Reduced for testing
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=151936,
            additional_vocab_size=128,
            hidden_act="silu",
            layer_norm_eps=1e-6,
            rope_theta=5000000.0,
            use_qk_norm=True,
        )

        vit_config = molmo2.config.VitConfig(
            model_type="molmo2",
            hidden_size=128,  # Reduced for testing
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            image_patch_size=14,
            image_num_pos=729,  # 27x27
            image_default_input_size=[378, 378],
        )

        adapter_config = molmo2.config.AdapterConfig(
            model_type="molmo2",
            hidden_size=128,  # Match vit_config
            intermediate_size=256,
            text_hidden_size=256,  # Match text_config
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            vit_layers=[-1, -2],  # Use last two layers
        )

        vision_config = molmo2.VisionConfig(
            vit_config=vit_config,
            adapter_config=adapter_config,
        )

        config = molmo2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="molmo2",
        )
        model = molmo2.Model(config)

        # Test language model
        # Note: vocab_size in logits is base vocab only, additional tokens are handled separately
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_florence2(self):
        from mlx_vlm.models import florence2

        text_config = florence2.TextConfig()
        vision_config = florence2.VisionConfig(drop_path_rate=0.0)
        config = florence2.ModelConfig(
            text_config=text_config, vision_config=vision_config
        )
        model = florence2.Model(config)

        # Create dummy data
        batch_size = 1
        seq_length = 590
        # Create dummy text inputs
        inputs_embeds = mx.zeros((batch_size, seq_length, config.text_config.d_model))

        # Create dummy masks and embeddings
        decoder_inputs_embeds = mx.zeros((batch_size, 1, config.text_config.d_model))

        # Forward pass
        output = model.language_model(
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )

        # Check output shape matches the example shape
        self.assertEqual(
            output.logits.shape, (batch_size, 1, config.text_config.vocab_size)
        )
        self.assertEqual(
            output.encoder_outputs.shape,
            (batch_size, seq_length, config.text_config.d_model),
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.dim_embed[-1],
            config.vision_config.in_chans,
            config.vision_config.image_size,
            channel_first=True,
        )

    def test_deepseek_vl_v2(self):
        from mlx_vlm.models import deepseek_vl_v2

        text_config = deepseek_vl_v2.TextConfig(model_type="text")
        vision_config = deepseek_vl_v2.VisionConfig(model_type="vision")
        projector_config = deepseek_vl_v2.ProjectorConfig()
        config = deepseek_vl_v2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="deepseek_v2",
        )
        model = deepseek_vl_v2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision,
            config.vision_config.model_type,
            config.vision_config.width,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_aya_vision(self):
        from mlx_vlm.models import aya_vision

        text_config = aya_vision.TextConfig(model_type="aya_vision")
        vision_config = aya_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=1152,
            num_attention_heads=16,
            patch_size=14,
            num_hidden_layers=27,
        )
        config = aya_vision.ModelConfig(
            model_type="aya_vision",
            text_config=text_config,
            vision_config=vision_config,
        )
        model = aya_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llama4(self):
        from mlx_vlm.models import llama4

        text_config = llama4.TextConfig(
            model_type="llama4_text",
            hidden_size=5120,
            num_hidden_layers=3,
            intermediate_size=8192,
            intermediate_size_mlp=16384,
            num_attention_heads=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            vocab_size=32000,
            attention_chunk_size=8192,
            attention_dropout=0.0,
            head_dim=128,
            hidden_act="silu",
            attention_bias=False,
        )
        vision_config = llama4.VisionConfig(
            model_type="llama4_vision_model",
            image_size=336,
            patch_size=14,
            num_channels=3,
            num_hidden_layers=3,
            hidden_size=1408,
            intermediate_size=5632,
            num_attention_heads=16,
            norm_eps=1e-05,
            initializer_range=0.02,
            pixel_shuffle_ratio=0.5,
            projector_input_dim=4096,
            projector_output_dim=4096,
            projector_dropout=0.0,
            vision_output_dim=4096,
            rope_theta=10000,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        config = llama4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llama4",
        )
        model = llama4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            channel_first=True,
            projector_output_dim=config.vision_config.projector_output_dim,
        )

    def test_kimi_vl(self):
        pass

        from mlx_vlm.models import kimi_vl

        text_config = kimi_vl.TextConfig()
        vision_config = kimi_vl.VisionConfig()
        config = kimi_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="kimi_vl",
        )
        model = kimi_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.patch_size, config.vision_config.patch_size),
            grid_thw=mx.array(
                [[20, 28], [22, 28]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
            batch_size=1176,
            vision_feature_layer=-1,
        )

    def test_gemma3(self):
        from mlx_vlm.models import gemma3

        text_config = gemma3.TextConfig(
            model_type="gemma3",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
        )
        vision_config = gemma3.VisionConfig(
            model_type="gemma3",
            image_size=224,
            patch_size=14,
            num_channels=3,
            num_hidden_layers=18,
            hidden_size=2048,
            intermediate_size=16384,
            num_attention_heads=8,
        )
        config = gemma3.ModelConfig(
            text_config=text_config, vision_config=vision_config, model_type="gemma3"
        )
        model = gemma3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_gemma4(self):
        import tempfile
        from pathlib import Path

        from mlx_lm.utils import quantize_model

        from mlx_vlm.models import gemma4
        from mlx_vlm.utils import load_model, save_config, save_weights

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            rms_norm_eps=1e-6,
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=8,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=3,
            final_logit_softcapping=30.0,
        )
        vision_config = gemma4.VisionConfig(
            model_type="gemma4_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            patch_size=16,
            pooling_kernel_size=2,
            default_output_length=4,
            position_embedding_size=64,
            use_clipped_linears=False,
        )
        config = gemma4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="gemma4",
            vocab_size=64,
            image_token_id=63,
        )
        model = gemma4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,  # num_channels
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Full model forward: text-only (no image)
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image tokens
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

        # Quantized save/load regression for per-layer projection.
        quant_model = gemma4.Model(config)

        def quantize_per_layer_projection(path: str, _module: nn.Module):
            return path == "language_model.model.per_layer_model_projection"

        quant_model, quantized_config = quantize_model(
            quant_model,
            {
                "model_type": "gemma4",
                "vocab_size": config.vocab_size,
                "image_token_id": config.image_token_id,
                "audio_config": None,
                "text_config": vars(text_config).copy(),
                "vision_config": vars(vision_config).copy(),
            },
            group_size=32,
            bits=4,
            quant_predicate=quantize_per_layer_projection,
        )
        self.assertTrue(
            hasattr(
                quant_model.language_model.model.per_layer_model_projection, "scales"
            )
        )
        quantized_config["quantization"][
            "language_model.model.per_layer_model_projection"
        ] = {
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        }

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir)
            save_weights(model_path, quant_model)
            save_config(quantized_config, model_path / "config.json")
            loaded = load_model(model_path)

        self.assertTrue(
            hasattr(loaded.language_model.model.per_layer_model_projection, "scales")
        )
        logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32)).logits
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, config.vocab_size))

        # Full model forward: text + audio tokens
        audio_config = gemma4.AudioConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            conv_kernel_size=3,
            attention_chunk_size=4,
            attention_context_left=5,
            attention_context_right=0,
            subsampling_conv_channels=(8, 4),
            output_proj_dims=32,
        )
        config_with_audio = gemma4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            model_type="gemma4",
            vocab_size=64,
            image_token_id=63,
            audio_token_id=62,
        )
        model_audio = gemma4.Model(config_with_audio)
        aud_id = config_with_audio.audio_token_id
        input_ids_audio = mx.array([[0, aud_id, aud_id, aud_id, aud_id, 1]])
        audio_features = mx.random.normal((1, 64, 128))
        audio_mask = mx.zeros((1, 64), dtype=mx.bool_)
        output = model_audio(
            input_ids_audio, audio_features=audio_features, audio_mask=audio_mask
        )
        self.assertEqual(
            output.logits.shape,
            (1, 6, config_with_audio.text_config.vocab_size),
        )

    def test_gemma4_moe(self):
        """Gemma 4 MoE variant: MoE, K-eq-V, no per-layer inputs."""
        from mlx_vlm.models import gemma4

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=5,
            final_logit_softcapping=30.0,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=16,
        )
        vision_config = gemma4.VisionConfig(
            model_type="gemma4_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            patch_size=16,
            pooling_kernel_size=2,
            default_output_length=4,
            position_embedding_size=64,
            use_clipped_linears=False,
        )
        config = gemma4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="gemma4",
            vocab_size=64,
            image_token_id=63,
        )
        model = gemma4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Verify MoE layers are created
        for layer in model.language_model.model.layers:
            self.assertTrue(layer.enable_moe)
            self.assertIsNotNone(layer.router)
            self.assertIsNotNone(layer.experts)

        # Verify K-eq-V on full attention layers
        for layer in model.language_model.model.layers:
            if layer.layer_type == "full_attention":
                self.assertTrue(layer.self_attn.use_k_eq_v)
                self.assertFalse(hasattr(layer.self_attn, "v_proj"))
            else:
                self.assertFalse(layer.self_attn.use_k_eq_v)

        # Verify layer_scalar exists on all layers
        for layer in model.language_model.model.layers:
            self.assertIsNotNone(layer.layer_scalar)

        # Full model forward: text-only
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

    def test_gemma4_attention_snapshots_cache_offset(self):
        """Gemma 4 Attention must snapshot cache.offset to prevent in-place
        mutation aliasing under batched caches where cache.offset is an
        mx.array. Without the snapshot, cache.update_and_fetch would mutate
        the local offset variable between K-rope and Q-rope, producing a
        one-position shift and a deterministic decode loop. See the equivalent
        defense in mlx_lm/models/gemma4_text.py (offset = mx.array(cache.offset)).
        """
        from mlx_vlm.models.gemma4 import language

        text_config = language.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            rms_norm_eps=1e-6,
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=8,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=3,
            final_logit_softcapping=30.0,
        )
        attn = language.Attention(text_config, layer_idx=0)

        # Stub cache mirroring the BatchRotatingKVCache shape: cache.offset
        # is an mx.array, update_and_fetch advances it in place via +=.
        class _StubMxArrayCache:
            def __init__(self, start):
                self.offset = mx.array([start])
                self.state = (
                    mx.zeros((1, 1, 0, 16)),
                    mx.zeros((1, 1, 0, 16)),
                )
                self.max_size = 2048

            def update_and_fetch(self, keys, values):
                self.offset += keys.shape[-2]
                new_keys = mx.concatenate([self.state[0], keys], axis=-2)
                new_values = mx.concatenate([self.state[1], values], axis=-2)
                self.state = (new_keys, new_values)
                return new_keys, new_values

        cache = _StubMxArrayCache(start=21)
        cache_offset_id = id(cache.offset)

        rope_ids = []
        rope_values = []
        original_rope = attn.rope

        def _recording_rope(x, offset=None):
            rope_ids.append(id(offset) if offset is not None else None)
            rope_values.append(offset.tolist() if hasattr(offset, "tolist") else offset)
            return original_rope(x, offset=offset)

        attn.rope = _recording_rope

        x = mx.random.uniform(shape=(1, 1, text_config.hidden_size))
        output = attn(x, mask=None, cache=cache)
        mx.eval(output)

        # Both K-rope and Q-rope must fire.
        self.assertGreaterEqual(len(rope_ids), 2)

        # The offset object passed to rope must not alias cache.offset;
        # otherwise cache.update_and_fetch would mutate it between K-rope
        # and Q-rope.
        for i, oid in enumerate(rope_ids):
            self.assertNotEqual(
                oid,
                cache_offset_id,
                f"rope call #{i} aliased cache.offset instead of snapshotting",
            )

        # Stub advanced in place, confirming the mx.array mutation path.
        self.assertEqual(cache.offset.tolist(), [22])

        # Both rope calls must see the same pre-update value.
        self.assertEqual(rope_values[0], [21])
        self.assertEqual(rope_values[1], [21])
        self.assertEqual(rope_values[0], rope_values[1])

    def test_gemma4_dense(self):
        """Gemma 4 dense variant: K-eq-V, no per-layer inputs, no MoE."""
        from mlx_vlm.models import gemma4

        text_config = gemma4.TextConfig(
            model_type="gemma4_text",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            sliding_window=32,
            sliding_window_pattern=5,
            final_logit_softcapping=30.0,
            attention_k_eq_v=True,
            enable_moe_block=False,
        )
        vision_config = gemma4.VisionConfig(
            model_type="gemma4_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            patch_size=16,
            pooling_kernel_size=2,
            default_output_length=4,
            position_embedding_size=64,
            use_clipped_linears=False,
        )
        config = gemma4.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="gemma4",
            vocab_size=64,
            image_token_id=63,
        )
        model = gemma4.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            3,
            (64, 64),
            vision_feature_layer=0,
            channel_first=True,
        )

        # Verify NO MoE layers
        for layer in model.language_model.model.layers:
            self.assertFalse(layer.enable_moe)

        # Verify K-eq-V on full attention layers
        for layer in model.language_model.model.layers:
            if layer.layer_type == "full_attention":
                self.assertTrue(layer.self_attn.use_k_eq_v)
                self.assertFalse(hasattr(layer.self_attn, "v_proj"))
            else:
                self.assertFalse(layer.self_attn.use_k_eq_v)

        # Verify layer_scalar exists on all layers
        for layer in model.language_model.model.layers:
            self.assertIsNotNone(layer.layer_scalar)

        # Full model forward: text-only
        input_ids = mx.array([[0, 1, 2, 3]])
        output = model(input_ids)
        self.assertEqual(output.logits.shape, (1, 4, config.text_config.vocab_size))

        # Full model forward: text + image
        img_id = config.image_token_id
        input_ids_with_img = mx.array([[0, img_id, img_id, img_id, img_id, 1]])
        pixel_values = mx.random.uniform(shape=(1, 3, 64, 64))
        output = model(input_ids_with_img, pixel_values=pixel_values)
        self.assertEqual(output.logits.shape, (1, 6, config.text_config.vocab_size))

    def test_deepseekocr(self):
        from mlx_vlm.models import deepseekocr

        text_config = deepseekocr.TextConfig()
        vision_config = deepseekocr.VisionConfig(model_type="vision")
        projector_config = deepseekocr.ProjectorConfig(projector_type="linear")
        config = deepseekocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            model_type="deepseekocr",
        )
        model = deepseekocr.Model(config)
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # TODO: Add test for vision model. Ensure I can pass input type and shapes.

    def test_jina_vlm(self):
        from mlx_vlm.models import jina_vlm

        text_config = jina_vlm.TextConfig(
            model_type="jina_vlm",
            hidden_size=2048,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            additional_vocab_size=128,
            intermediate_size=6144,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            use_qk_norm=True,
        )

        vision_config = jina_vlm.VisionConfig(
            model_type="jina_vlm",
            hidden_size=1152,
            num_hidden_layers=4,
            num_attention_heads=16,
            head_dim=72,
            patch_size=14,
            image_size=378,
            num_channels=3,
            intermediate_size=4304,
            vit_layers=(-2, -4),
            output_size=2048,
            pooling_h=2,
            pooling_w=2,
            connector_hidden_size=6144,
        )

        config = jina_vlm.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="jina_vlm",
            vocab_size=151936,
        )

        model = jina_vlm.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Vision model expects patchified input from processor, skip standard test
        # Test basic forward pass with patchified input instead
        batch_size = 1
        n_patches = (
            config.vision_config.image_size // config.vision_config.patch_size
        ) ** 2
        patch_dim = (
            config.vision_config.patch_size**2 * config.vision_config.num_channels
        )
        pixel_values = mx.random.uniform(shape=(batch_size, n_patches, patch_dim))
        output, hidden_states = model.vision_model(pixel_values)
        # Check output shape matches hidden size
        self.assertEqual(output.shape[-1], config.vision_config.hidden_size)
        self.assertEqual(len(hidden_states), config.vision_config.num_hidden_layers + 1)

    def test_hunyuan_vl(self):
        from mlx_vlm.models import hunyuan_vl

        text_config = hunyuan_vl.TextConfig(
            model_type="hunyuan_vl",
            vocab_size=120818,
            org_vocab_size=120818,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            attention_head_dim=128,
            intermediate_size=3584,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            use_qk_norm=True,
            rope_theta=10000.0,
            rope_scaling={
                "alpha": 1000.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "type": "xdrope",
                "xdrope_section": [16, 16, 16, 16],
            },
            max_position_embeddings=32768,
            rms_norm_eps=1e-5,
            norm_type="rms",
            tie_word_embeddings=True,
            use_cache=True,
            initializer_range=0.02,
            routed_scaling_factor=1.0,
            bos_token_id=120000,
            eos_token_id=120020,
            eod_token_id=120020,
            pad_token_id=-1,
            pad_id=120002,
        )

        vision_config = hunyuan_vl.VisionConfig(
            model_type="hunyuan_vl",
            hidden_size=1152,
            out_hidden_size=1024,
            num_hidden_layers=5,
            num_attention_heads=16,
            intermediate_size=4304,
            patch_size=16,
            num_channels=3,
            spatial_merge_size=2,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rms_norm_eps=1e-5,
            interpolate_mode="bilinear",
            cat_extra_token=1,
            img_max_token_num=4096,
            max_vit_seq_len=16384,
            add_patchemb_bias=True,
            max_image_size=2048,
            hidden_act="gelu",
        )

        config = hunyuan_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="hunyuan_vl",
            image_token_id=120120,
            image_start_token_id=120118,
            image_end_token_id=120119,
            image_newline_token_id=120121,
            bos_token_id=120000,
            eos_token_id=120020,
            pad_token_id=-1,
            pad_id=120002,
            vocab_size=120818,
            org_vocab_size=120818,
            tie_word_embeddings=True,
        )

        model = hunyuan_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.num_channels,
            (1080, 768),
            vision_feature_layer=-1,
            grid_thw=mx.array(
                [[1, 18, 60]], dtype=mx.int64
            ),  # image temporals shape (num_images, 3)
        )

    def test_ernie4_5_moe_vl(self):
        from mlx_vlm.models import ernie4_5_moe_vl

        # Config based on baidu/ERNIE-4.5-VL-28B-A3B-Thinking (scaled down for testing)
        text_config = ernie4_5_moe_vl.TextConfig(
            model_type="ernie",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rope_theta=500000,
            rope_scaling={"type": "default", "mrope_section": [12, 12, 8]},
            tie_word_embeddings=True,
            moe_num_experts=[8, 8],
            moe_intermediate_size=[128, 64],
            moe_k=2,
            moe_layer_start_index=[1, 1],
            moe_layer_end_index=[4, 3],
            moe_num_shared_experts=1,
            use_bias=False,
        )

        vision_config = ernie4_5_moe_vl.VisionConfig(
            model_type="DFNRope_vision_transformer",
            depth=4,
            embed_dim=128,
            hidden_size=128,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            in_channels=3,
        )

        config = ernie4_5_moe_vl.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            pixel_hidden_size=128,
            spatial_conv_size=2,
            temporal_conv_size=2,
            vocab_size=1000,
        )

        model = ernie4_5_moe_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_minicpmo(self):
        from mlx_vlm.models import minicpmo

        text_config = minicpmo.TextConfig(
            model_type="minicpmo",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=4,
            head_dim=16,
            rope_theta=10000.0,
            max_position_embeddings=2048,
        )
        vision_config = minicpmo.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=28,
            patch_size=14,
        )
        setattr(vision_config, "spatial_merge_size", 1)
        config = minicpmo.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            query_num=4,
        )
        setattr(config, "image_token_id", 1)
        setattr(config, "video_token_id", 2)
        setattr(config, "vision_start_token_id", 3)
        model = minicpmo.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=0,
        )

    def test_phi4mm(self):
        from mlx_vlm.models import phi4mm

        config = phi4mm.ModelConfig(
            text_config=phi4mm.TextConfig(
                model_type="phi4mm",
                max_position_embeddings=2048,
            ),
            vision_config=phi4mm.VisionConfig(
                model_type="siglip2_vision_model",
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_hidden_layers=2,
                patch_size=14,
                image_size=28,
                num_channels=3,
                layer_norm_eps=1e-6,
            ),
            model_type="phi4mm",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            mm_hidden_size=32,
            image_token_index=-200,
            audio_token_index=200,
            audio_processor={
                "config": {
                    "attention_dim": 32,
                    "attention_heads": 4,
                    "num_blocks": 2,
                    "linear_units": 64,
                    "input_size": 80,
                    "time_reduction": 8,
                    "kernel_size": 3,
                    "conv_channels": 32,
                    "ext_pw_out_channel": 32,
                    "depthwise_seperable_out_channel": 32,
                }
            },
        )
        model = phi4mm.Model(config)

        # Language model
        self.assertEqual(model.model_type, config.model_type)
        self.assertEqual(len(model.layers), config.num_hidden_layers)

        batch_size = 1
        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            logits = outputs.logits
            self.assertEqual(logits.shape, (batch_size, 2, config.vocab_size))
            self.assertEqual(logits.dtype, t)

        # Vision tower: SigLIP2 with NaFlex
        # Input: (B, num_patches, patch_dim) where patch_dim = P*P*C
        patch_size = config.vision_config.patch_size
        num_channels = config.vision_config.num_channels
        img_size = config.vision_config.image_size
        patch_dim = patch_size * patch_size * num_channels
        num_patches = (img_size // patch_size) ** 2
        pixel_values = mx.random.uniform(shape=(1, num_patches, patch_dim))
        features = model.vision_tower(pixel_values)
        self.assertEqual(features.shape[-1], config.vision_config.hidden_size)

        # MM projector: projects vision features to LLM hidden size
        projected = model.apply_mm_projector(features)
        self.assertEqual(projected.shape[-1], config.hidden_size)

        # Audio encoder: Conformer
        # Input: (B, T, 80) mel spectrogram features
        audio_input = mx.random.uniform(shape=(1, 100, 80))
        audio_mask = mx.ones((1, 100))
        encoded_audio, _ = model.audio_encoder(audio_input, audio_mask)
        audio_config = getattr(config, "_audio_config")
        self.assertEqual(encoded_audio.shape[-1], audio_config.attention_dim)

        # Audio projection: projects audio features to LLM hidden size
        audio_projected = model.audio_projection(encoded_audio, mode="speech")
        self.assertEqual(audio_projected.shape[-1], config.hidden_size)

    def test_glm_ocr(self):
        from mlx_vlm.models import glm_ocr

        text_config = glm_ocr.TextConfig(
            model_type="glm_ocr_text",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-5,
            max_position_embeddings=1000,
            rope_parameters={
                "rope_type": "default",
                "mrope_section": [8, 12, 12],
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000,
            },
        )

        vision_config = glm_ocr.VisionConfig(
            model_type="glm_ocr_vision",
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            patch_size=14,
            in_channels=3,
            out_hidden_size=128,
            spatial_merge_size=2,
            temporal_patch_size=2,
            rms_norm_eps=1e-5,
        )

        config = glm_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm_ocr",
            image_token_id=999,
            video_token_id=998,
            vocab_size=1000,
        )

        model = glm_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Test vision model with proper input format
        # grid_thw format: [temporal, height/patch, width/patch]
        # For grid_thw = [2, 14, 14], we have 2*14*14 = 392 patches
        # Height/width must be divisible by spatial_merge_size (2)
        grid_thw = mx.array([[2, 14, 14]], dtype=mx.int64)
        num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])

        # Create input tensor - flat array that gets reshaped internally
        # Shape: (num_patches * in_channels * temporal_patch_size * patch_size * patch_size)
        total_elements = (
            num_patches
            * config.vision_config.in_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size
            * config.vision_config.patch_size
        )
        pixel_values = mx.random.uniform(shape=(total_elements,))

        # Forward pass
        hidden_states = model.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Check output shape
        # After spatial merge (2x2), we should have:
        # temporal * (height/spatial_merge) * (width/spatial_merge)
        # = 2 * (14/2) * (14/2) = 2 * 7 * 7 = 98 patches
        expected_patches = int(
            grid_thw[0, 0]
            * (grid_thw[0, 1] // config.vision_config.spatial_merge_size)
            * (grid_thw[0, 2] // config.vision_config.spatial_merge_size)
        )
        self.assertEqual(hidden_states.shape[0], expected_patches)
        self.assertEqual(hidden_states.shape[1], config.vision_config.out_hidden_size)

    def test_phi4_siglip(self):
        from mlx_vlm.models import phi4_siglip

        text_config = phi4_siglip.TextConfig(
            model_type="phi4-siglip",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            partial_rotary_factor=1.0,
        )

        vision_config = phi4_siglip.VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            patch_size=14,
            num_patches=256,
        )

        config = phi4_siglip.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="phi4-siglip",
            mm_hidden_size=16,
        )

        model = phi4_siglip.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_moondream3(self):
        from mlx_vlm.models import moondream3

        text_config = moondream3.TextConfig(
            model_type="moondream3",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=256,
            rope_theta=1500000.0,
            rope_dim=8,
            rms_norm_eps=1e-5,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_start_layer=2,
            attention_bias=True,
            prefix_attn=5,
        )

        vision_config = moondream3.VisionConfig(
            model_type="moondream3_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=14,
            crop_size=28,
            in_channels=3,
            proj_inner_dim=64,
            proj_out_dim=64,
            attention_bias=True,
            layer_norm_eps=1e-6,
        )

        config = moondream3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="moondream3",
        )

        model = moondream3.Model(config)

        # Language model test
        self.language_test_runner(
            model.language_model,
            text_config.model_type,
            text_config.vocab_size,
            text_config.num_hidden_layers,
        )

        # Vision encoder test
        batch_size = 1
        crop_size = vision_config.crop_size
        pixel_values = mx.random.uniform(shape=(batch_size, crop_size, crop_size, 3))
        features = model.vision.encoder(pixel_values)
        grid_size = crop_size // vision_config.patch_size
        num_patches = grid_size * grid_size
        self.assertEqual(
            features.shape, (batch_size, num_patches, vision_config.hidden_size)
        )

        # Vision projection: concat global+local -> project
        combined = mx.concatenate([features, features], axis=-1)
        projected = model.vision.proj_mlp(combined)
        self.assertEqual(
            projected.shape,
            (batch_size, num_patches, vision_config.proj_out_dim),
        )

        # Full model forward with vision
        # Input: BOS + num_patches placeholders + 2 text tokens
        input_ids = mx.zeros((1, 1 + num_patches + 2), dtype=mx.int32)
        input_ids[0, -2:] = mx.array([1, 2])
        outputs = model(
            input_ids,
            pixel_values=pixel_values,
            num_crops=[1],
            crop_layouts=[(1, 1)],
        )
        self.assertEqual(outputs.logits.shape[-1], text_config.vocab_size)

    def test_mistral4(self):
        from mlx_vlm.models import mistral3

        text_config = mistral3.TextConfig(
            model_type="mistral4",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=256,
            max_position_embeddings=1024,
            rope_traditional=False,
            rope_parameters={
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "factor": 4.0,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 512,
            },
            tie_word_embeddings=False,
            attention_bias=False,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            first_k_dense_replace=0,
        )

        vision_config = mistral3.VisionConfig(
            model_type="pixtral",
            hidden_size=1024,
            num_hidden_layers=2,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            num_channels=3,
        )

        config = mistral3.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="mistral3",
        )

        model = mistral3.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_falcon_ocr(self):
        from mlx_vlm.models import falcon_ocr

        text_config = falcon_ocr.TextConfig(
            model_type="falcon_ocr",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
            vocab_size=256,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

        vision_config = falcon_ocr.VisionConfig(
            model_type="falcon_ocr",
            spatial_patch_size=4,
            temporal_patch_size=1,
            channel_size=3,
        )

        config = falcon_ocr.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="falcon_ocr",
            vocab_size=256,
            img_id=227,
            image_cls_token_id=244,
            img_end_id=230,
        )

        model = falcon_ocr.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_falcon_perception(self):
        from mlx_vlm.models import falcon_perception

        text_config = falcon_perception.TextConfig(
            model_type="falcon_perception",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
            vocab_size=256,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

        vision_config = falcon_perception.VisionConfig(
            model_type="falcon_perception",
            spatial_patch_size=4,
            temporal_patch_size=1,
            channel_size=3,
        )

        config = falcon_perception.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="falcon_perception",
            vocab_size=256,
            img_id=227,
            image_cls_token_id=244,
            img_end_id=230,
            coord_token_id=240,
            size_token_id=241,
            seg_token_id=262,
            coord_enc_dim=64,
            coord_dec_dim=128,
            coord_out_dim=256,
            size_enc_dim=64,
            size_dec_dim=128,
            size_out_dim=256,
            do_segmentation=False,
            segm_out_dim=64,
            num_segm_layers=1,
        )

        model = falcon_perception.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite_vision(self):
        from mlx_vlm.models import granite_vision

        text_config = granite_vision.TextConfig(
            model_type="granite",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=300000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=8.0,
        )

        vision_config = granite_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=56,
            patch_size=14,
        )

        config = granite_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite_vision",
            vision_feature_layer=[-2, -1],
            vision_feature_select_strategy="full",
        )

        model = granite_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite4_vision(self):
        from mlx_vlm.models import granite4_vision

        text_config = granite4_vision.TextConfig(
            model_type="granitemoehybrid",
            hidden_size=64,
            intermediate_size=128,
            shared_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=10000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=10.0,
        )

        vision_config = granite4_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=48,
            patch_size=16,
        )

        config = granite4_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite4_vision",
            deepstack_layer_map=[[-1, 0]],
            use_spatial_sampling=False,
            downsample_rate="3/3",
            use_image_newline_parameter=False,
        )

        model = granite4_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

    def test_granite4_1_vision(self):
        from mlx_vlm.models import granite4_vision

        text_config = granite4_vision.TextConfig(
            model_type="granite",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=10000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=10.0,
        )

        vision_config = granite4_vision.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=48,
            patch_size=16,
        )

        config = granite4_vision.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="granite4_vision",
            deepstack_layer_map=[[-1, 0]],
            use_spatial_sampling=False,
            downsample_rate="3/3",
            use_image_newline_parameter=False,
        )

        model = granite4_vision.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=0,
        )

    def test_youtu_vl(self):
        from mlx_vlm.models import youtu_vl

        text_config = youtu_vl.TextConfig(
            model_type="youtu_vl",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1024,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=32,
            rope_theta=500000.0,
            rope_interleave=True,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )
        vision_config = youtu_vl.VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=128,
            out_hidden_size=256,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            window_size=64,
            fullatt_block_indexes=[1],
        )
        config = youtu_vl.ModelConfig(
            model_type="youtu_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=100,
            video_token_id=101,
            vocab_size=1024,
        )
        model = youtu_vl.Model(config)

        # Language model: MLA with absorb — fp32/fp16 forward + cached decode check
        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        # Vision tower takes packed patches + spatial_shapes:
        #   pixel_values: (num_patches, patch_size**2 * channels)
        #   spatial_shapes: (batch, 2) — (h_patches, w_patches)
        patch_dim = vision_config.patch_size**2 * vision_config.num_channels
        h_p, w_p = 4, 4
        num_patches = h_p * w_p
        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.out_hidden_size,
            config.vision_config.num_channels,
            (num_patches, patch_dim),
            vision_feature_layer=-1,
            spatial_shapes=mx.array([[h_p, w_p]], dtype=mx.int32),
        )

        # sanitize splits kv_b_proj per-head into embed_q (k) + unembed_out (v)
        H, nope, v_head = 4, 32, 32
        kv_rank = text_config.kv_lora_rank
        w = mx.arange(H * (nope + v_head) * kv_rank, dtype=mx.float32).reshape(
            H * (nope + v_head), kv_rank
        )
        sanitized = model.sanitize(
            {
                "model.layers.0.self_attn.kv_b_proj.weight": w,
                "lm_head.weight": mx.zeros((1, 1)),  # tied; must be dropped
            }
        )
        prefix = "language_model.model.layers.0.self_attn"
        self.assertNotIn(f"{prefix}.kv_b_proj.weight", sanitized)
        self.assertNotIn("language_model.lm_head.weight", sanitized)
        self.assertEqual(
            sanitized[f"{prefix}.embed_q.weight"].shape, (H, kv_rank, nope)
        )
        self.assertEqual(
            sanitized[f"{prefix}.unembed_out.weight"].shape, (H, v_head, kv_rank)
        )


class TestGetInputEmbeddings(unittest.TestCase):
    """Test that all models with get_input_embeddings return InputEmbeddingsFeatures."""

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

    def test_llava_input_embeddings(self):
        from mlx_vlm.models import llava

        model = llava.Model(
            llava.ModelConfig(
                text_config=llava.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=llava.VisionConfig(
                    model_type="clip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava")

    def test_llava_bunny_input_embeddings(self):
        from mlx_vlm.models import llava_bunny

        model = llava_bunny.Model(
            llava_bunny.ModelConfig(
                text_config=llava_bunny.TextConfig(
                    model_type="qwen2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                ),
                vision_config=llava_bunny.VisionConfig(
                    model_type="siglip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava-qwen2",
                auto_map={
                    "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                    "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
                },
                hidden_size=16,
                mm_hidden_size=16,
                image_token_index=-200,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava_bunny")

    def test_llava_next_input_embeddings(self):
        from mlx_vlm.models import llava_next

        model = llava_next.Model(
            llava_next.ModelConfig(
                text_config=llava_next.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=llava_next.VisionConfig(
                    model_type="clip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="llava",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "llava_next")

    def test_qwen2_vl_input_embeddings(self):
        from mlx_vlm.models import qwen2_vl

        model = qwen2_vl.Model(
            qwen2_vl.ModelConfig(
                text_config=qwen2_vl.TextConfig(
                    model_type="qwen2_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen2_vl.VisionConfig(
                    model_type="qwen2_vl",
                    depth=1,
                    embed_dim=16,
                    hidden_size=16,
                    num_heads=2,
                    image_size=28,
                    patch_size=14,
                    in_channels=3,
                ),
                model_type="qwen2_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen2_vl")

    def test_qwen2_5_vl_input_embeddings(self):
        from mlx_vlm.models import qwen2_5_vl

        model = qwen2_5_vl.Model(
            qwen2_5_vl.ModelConfig(
                text_config=qwen2_5_vl.TextConfig(
                    model_type="qwen2_5_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen2_5_vl.VisionConfig(
                    model_type="qwen2_5_vl",
                    depth=1,
                    hidden_size=16,
                    num_heads=2,
                    out_hidden_size=16,
                    image_size=28,
                    patch_size=14,
                    in_channels=3,
                    fullatt_block_indexes=[0],
                    window_size=14,
                ),
                model_type="qwen2_5_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen2_5_vl")

    def test_qwen3_vl_input_embeddings(self):
        from mlx_vlm.models import qwen3_vl

        model = qwen3_vl.Model(
            qwen3_vl.ModelConfig(
                text_config=qwen3_vl.TextConfig(
                    model_type="qwen3_vl_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    head_dim=8,
                    rope_theta=1000.0,
                    max_position_embeddings=1000,
                    rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
                ),
                vision_config=qwen3_vl.VisionConfig(
                    model_type="qwen3_vl",
                    depth=1,
                    hidden_size=16,
                    num_heads=2,
                    out_hidden_size=16,
                    patch_size=14,
                    in_channels=3,
                    num_position_embeddings=4,
                    deepstack_visual_indexes=[],
                ),
                model_type="qwen3_vl",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_vl")

    def test_paligemma_input_embeddings(self):
        from mlx_vlm.models import paligemma

        model = paligemma.Model(
            paligemma.ModelConfig(
                text_config=paligemma.TextConfig(
                    model_type="gemma",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-6,
                ),
                vision_config=paligemma.VisionConfig(
                    model_type="siglip_vision_model",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                    projection_dim=16,
                ),
                model_type="paligemma",
                image_token_index=31,
                hidden_size=16,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "paligemma")

    def test_idefics2_input_embeddings(self):
        from mlx_vlm.models import idefics2

        model = idefics2.Model(
            idefics2.ModelConfig(
                text_config=idefics2.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=idefics2.VisionConfig(
                    model_type="idefics2",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                perceiver_config=idefics2.PerceiverConfig(
                    model_type="idefics2Perceiver",
                    resampler_n_latents=2,
                    resampler_depth=1,
                    resampler_n_heads=2,
                    resampler_head_dim=8,
                    num_key_value_heads=2,
                ),
                model_type="idefics2",
                image_token_index=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "idefics2")

    def test_idefics3_input_embeddings(self):
        from mlx_vlm.models import idefics3

        model = idefics3.Model(
            idefics3.ModelConfig(
                text_config=idefics3.TextConfig(
                    model_type="idefics3",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=idefics3.VisionConfig(
                    model_type="idefics3",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="idefics3",
                image_token_id=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "idefics3")

    def test_gemma3_input_embeddings(self):
        from mlx_vlm.models import gemma3

        model = gemma3.Model(
            gemma3.ModelConfig(
                text_config=gemma3.TextConfig(
                    model_type="gemma3",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-6,
                    num_key_value_heads=2,
                    head_dim=8,
                ),
                vision_config=gemma3.VisionConfig(
                    model_type="gemma3",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="gemma3",
                hidden_size=16,
                pad_token_id=0,
                image_token_index=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma3")

    def test_pixtral_input_embeddings(self):
        from mlx_vlm.models import pixtral

        model = pixtral.Model(
            pixtral.ModelConfig(
                text_config=pixtral.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                ),
                vision_config=pixtral.VisionConfig(
                    model_type="pixtral",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="pixtral",
                image_token_index=31,
                vocab_size=32,
                vision_feature_layer=-1,
            )
        )
        self._check_returns_input_embeddings_features(model, "pixtral")

    def test_mistral3_input_embeddings(self):
        from mlx_vlm.models import mistral3

        model = mistral3.Model(
            mistral3.ModelConfig(
                text_config=mistral3.TextConfig(
                    model_type="mistral",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    head_dim=8,
                    layer_types=["full_attention"],
                ),
                vision_config=mistral3.VisionConfig(
                    model_type="pixtral",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="mistral3",
            )
        )
        self._check_returns_input_embeddings_features(model, "mistral3")

    def test_multi_modality_input_embeddings(self):
        from mlx_vlm.models import multi_modality

        model = multi_modality.Model(
            multi_modality.ModelConfig(
                text_config=multi_modality.TextConfig(
                    model_type="llama",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-6,
                ),
                vision_config=multi_modality.VisionConfig(
                    model_type="vision",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                    params={},
                ),
                projector_config=multi_modality.ProjectorConfig(
                    cls="MlpProjector",
                    model_type="projector",
                    params={
                        "depth": 1,
                        "input_dim": 16,
                        "n_embed": 16,
                        "projector_type": "mlp_gelu",
                    },
                ),
                model_type="multi_modality",
                image_token_index=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "multi_modality")

    def test_aya_vision_input_embeddings(self):
        from mlx_vlm.models import aya_vision

        model = aya_vision.Model(
            aya_vision.ModelConfig(
                text_config=aya_vision.TextConfig(
                    model_type="aya_vision",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                ),
                vision_config=aya_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=16,
                    num_attention_heads=2,
                    patch_size=14,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    image_size=28,
                ),
                model_type="aya_vision",
            )
        )
        self._check_returns_input_embeddings_features(model, "aya_vision")

    def test_deepseek_vl_v2_input_embeddings(self):
        from mlx_vlm.models import deepseek_vl_v2

        model = deepseek_vl_v2.Model(
            deepseek_vl_v2.ModelConfig(
                text_config=deepseek_vl_v2.TextConfig(
                    model_type="deepseek_v2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    kv_lora_rank=8,
                    q_lora_rank=16,
                    qk_rope_head_dim=8,
                    v_head_dim=8,
                    qk_nope_head_dim=0,
                    moe_intermediate_size=16,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    num_experts_per_tok=1,
                ),
                vision_config=deepseek_vl_v2.VisionConfig(
                    model_type="vision",
                    layers=1,
                    width=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                ),
                projector_config=deepseek_vl_v2.ProjectorConfig(
                    projector_type="downsample_mlp_gelu",
                    input_dim=16,
                    n_embed=16,
                ),
                model_type="deepseek_vl_v2",
            )
        )
        self._check_returns_input_embeddings_features(model, "deepseek_vl_v2")

    def test_deepseekocr_input_embeddings(self):
        from mlx_vlm.models import deepseekocr

        model = deepseekocr.Model(
            deepseekocr.ModelConfig(
                text_config=deepseekocr.TextConfig(
                    model_type="deepseek_v2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    kv_lora_rank=8,
                    q_lora_rank=16,
                    qk_rope_head_dim=0,
                    v_head_dim=8,
                    qk_nope_head_dim=0,
                    moe_intermediate_size=16,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    num_experts_per_tok=1,
                ),
                vision_config=deepseekocr.VisionConfig(
                    model_type="vision",
                    layers=1,
                    width=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                ),
                projector_config=deepseekocr.ProjectorConfig(
                    projector_type="linear",
                    input_dim=16,
                    n_embed=16,
                ),
                model_type="deepseekocr",
            )
        )
        self._check_returns_input_embeddings_features(model, "deepseekocr")

    def test_falcon_ocr_input_embeddings(self):
        from mlx_vlm.models import falcon_ocr

        model = falcon_ocr.Model(
            falcon_ocr.ModelConfig(
                text_config=falcon_ocr.TextConfig(
                    model_type="falcon_ocr",
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    head_dim=16,
                    num_key_value_heads=2,
                    vocab_size=256,
                    intermediate_size=128,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=512,
                    rope_theta=10000.0,
                    tie_word_embeddings=False,
                ),
                vision_config=falcon_ocr.VisionConfig(
                    model_type="falcon_ocr",
                    spatial_patch_size=4,
                    temporal_patch_size=1,
                    channel_size=3,
                ),
                model_type="falcon_ocr",
                vocab_size=256,
                img_id=227,
                image_cls_token_id=244,
                img_end_id=230,
            )
        )
        self._check_returns_input_embeddings_features(model, "falcon_ocr")

    def test_falcon_perception_input_embeddings(self):
        from mlx_vlm.models import falcon_perception

        model = falcon_perception.Model(
            falcon_perception.ModelConfig(
                text_config=falcon_perception.TextConfig(
                    model_type="falcon_perception",
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    head_dim=16,
                    num_key_value_heads=2,
                    vocab_size=256,
                    intermediate_size=128,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=512,
                    rope_theta=10000.0,
                    tie_word_embeddings=False,
                ),
                vision_config=falcon_perception.VisionConfig(
                    model_type="falcon_perception",
                    spatial_patch_size=4,
                    temporal_patch_size=1,
                    channel_size=3,
                ),
                model_type="falcon_perception",
                vocab_size=256,
                img_id=227,
                image_cls_token_id=244,
                img_end_id=230,
                coord_token_id=240,
                size_token_id=241,
                seg_token_id=262,
                coord_enc_dim=64,
                coord_dec_dim=128,
                coord_out_dim=256,
                size_enc_dim=64,
                size_dec_dim=128,
                size_out_dim=256,
                do_segmentation=False,
                segm_out_dim=64,
                num_segm_layers=1,
            )
        )
        self._check_returns_input_embeddings_features(model, "falcon_perception")

    def test_fastvlm_input_embeddings(self):
        from mlx_vlm.models import fastvlm

        model = fastvlm.Model(
            fastvlm.ModelConfig(
                text_config=fastvlm.TextConfig(
                    model_type="fastvlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                ),
                vision_config=fastvlm.VisionConfig(
                    model_type="llava_qwen2",
                    hidden_size=16,
                ),
                model_type="llava_qwen2",
                mm_hidden_size=16,
            )
        )
        self._check_returns_input_embeddings_features(model, "fastvlm")

    def test_florence2_input_embeddings(self):
        from mlx_vlm.models import florence2

        model = florence2.Model(
            florence2.ModelConfig(
                vision_config=florence2.VisionConfig(
                    model_type="davit",
                    hidden_size=16,
                    in_chans=3,
                    depths=[1],
                    dim_embed=[16],
                    num_heads=[2],
                    num_groups=[2],
                    patch_size=[7],
                    patch_stride=[4],
                    patch_padding=[3],
                    patch_prenorm=[False],
                ),
                text_config=florence2.TextConfig(
                    d_model=16,
                    encoder_attention_heads=2,
                    decoder_attention_heads=2,
                    encoder_ffn_dim=32,
                    decoder_ffn_dim=32,
                    encoder_layers=1,
                    decoder_layers=1,
                    vocab_size=32,
                ),
                model_type="florence2",
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "florence2")

    def test_gemma3n_input_embeddings(self):
        from mlx_vlm.models import gemma3n

        model = gemma3n.Model(
            gemma3n.ModelConfig(
                text_config=gemma3n.TextConfig(
                    model_type="gemma3n",
                    hidden_size=16,
                    num_hidden_layers=2,
                    intermediate_size=[32, 32],
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    vocab_size_per_layer_input=32,
                    hidden_size_per_layer_input=16,
                    altup_num_inputs=2,
                    laurel_rank=8,
                    layer_types=["sliding_attention", "full_attention"],
                    sliding_window_pattern=1,
                ),
                vision_config=gemma3n.VisionConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=4,
                    vocab_offset=32,
                ),
                audio_config=gemma3n.AudioConfig(
                    vocab_size=4,
                    vocab_offset=36,
                    hidden_size=16,
                ),
                model_type="gemma3n",
                hidden_size=16,
                vocab_size=40,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma3n")

    def test_gemma4_input_embeddings(self):
        from mlx_vlm.models import gemma4

        model = gemma4.Model(
            gemma4.ModelConfig(
                text_config=gemma4.TextConfig(
                    model_type="gemma4_text",
                    hidden_size=16,
                    num_hidden_layers=2,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    global_head_dim=8,
                    vocab_size=32,
                    vocab_size_per_layer_input=32,
                    hidden_size_per_layer_input=8,
                    num_kv_shared_layers=0,
                    sliding_window=32,
                    sliding_window_pattern=1,
                ),
                vision_config=gemma4.VisionConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    patch_size=16,
                    pooling_kernel_size=2,
                    default_output_length=4,
                    position_embedding_size=64,
                    use_clipped_linears=False,
                ),
                model_type="gemma4",
                hidden_size=16,
                vocab_size=32,
                image_token_id=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "gemma4")

    def test_glm4v_input_embeddings(self):
        from mlx_vlm.models import glm4v

        model = glm4v.Model(
            glm4v.ModelConfig(
                text_config=glm4v.TextConfig(
                    model_type="glm4v_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                ),
                vision_config=glm4v.VisionConfig(
                    model_type="glm4v_vision",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    patch_size=14,
                    out_hidden_size=16,
                ),
                model_type="glm4v",
            )
        )
        self._check_returns_input_embeddings_features(model, "glm4v")

    def test_glm4v_moe_input_embeddings(self):
        from mlx_vlm.models import glm4v_moe

        model = glm4v_moe.Model(
            glm4v_moe.ModelConfig(
                text_config=glm4v_moe.TextConfig(
                    model_type="glm4v_text",
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    max_position_embeddings=128,
                    moe_intermediate_size=16,
                    norm_topk_prob=True,
                    num_attention_heads=2,
                    n_group=1,
                    head_dim=8,
                    topk_group=1,
                    n_shared_experts=1,
                    n_routed_experts=2,
                    routed_scaling_factor=1.0,
                    num_experts_per_tok=1,
                    first_k_dense_replace=0,
                    num_hidden_layers=1,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-5,
                    use_qk_norm=False,
                    attention_bias=False,
                    partial_rotary_factor=0.5,
                    rope_theta=10000.0,
                ),
                vision_config=glm4v_moe.VisionConfig(
                    model_type="glm4v_moe",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    patch_size=14,
                    out_hidden_size=16,
                ),
                model_type="glm4v_moe",
            )
        )
        self._check_returns_input_embeddings_features(model, "glm4v_moe")

    def test_hunyuan_vl_input_embeddings(self):
        from mlx_vlm.models import hunyuan_vl

        model = hunyuan_vl.Model(
            hunyuan_vl.ModelConfig(
                text_config=hunyuan_vl.TextConfig(
                    model_type="hunyuan_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                    attention_head_dim=8,
                ),
                vision_config=hunyuan_vl.VisionConfig(
                    model_type="hunyuan_vl",
                    hidden_size=16,
                    out_hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                ),
                model_type="hunyuan_vl",
            )
        )
        self._check_returns_input_embeddings_features(model, "hunyuan_vl")

    def test_jina_vlm_input_embeddings(self):
        from mlx_vlm.models import jina_vlm

        model = jina_vlm.Model(
            jina_vlm.ModelConfig(
                text_config=jina_vlm.TextConfig(
                    model_type="jina_vlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    intermediate_size=32,
                ),
                vision_config=jina_vlm.VisionConfig(
                    model_type="jina_vlm",
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    head_dim=8,
                    intermediate_size=32,
                    output_size=16,
                    connector_hidden_size=32,
                ),
                model_type="jina_vlm",
            )
        )
        self._check_returns_input_embeddings_features(model, "jina_vlm")

    def test_lfm2_vl_input_embeddings(self):
        from mlx_vlm.models import lfm2_vl

        model = lfm2_vl.Model(
            lfm2_vl.ModelConfig(
                text_config=lfm2_vl.TextConfig(
                    model_type="lfm2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    layer_types=["full_attention"],
                    block_dim=16,
                    block_ff_dim=32,
                    conv_dim=16,
                    conv_dim_out=16,
                ),
                vision_config=lfm2_vl.VisionConfig(
                    model_type="lfm2_vl",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="lfm2-vl",
                projector_hidden_size=16,
            )
        )
        self._check_returns_input_embeddings_features(model, "lfm2_vl")

    def test_molmo2_input_embeddings(self):
        from mlx_vlm.models import molmo2

        model = molmo2.Model(
            molmo2.ModelConfig(
                text_config=molmo2.TextConfig(
                    model_type="molmo2",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=32,
                    additional_vocab_size=4,
                ),
                vision_config=molmo2.VisionConfig(
                    vit_config=molmo2.VitConfig(
                        model_type="molmo2",
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                    ),
                    adapter_config=molmo2.AdapterConfig(
                        model_type="molmo2",
                        hidden_size=16,
                        intermediate_size=32,
                        text_hidden_size=16,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                        vit_layers=[-1],
                    ),
                ),
                model_type="molmo2",
            )
        )
        self._check_returns_input_embeddings_features(model, "molmo2")

    def test_paddleocr_vl_input_embeddings(self):
        from mlx_vlm.models import paddleocr_vl

        model = paddleocr_vl.Model(
            paddleocr_vl.ModelConfig(
                text_config=paddleocr_vl.TextConfig(
                    model_type="paddleocr_vl",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    head_dim=8,
                ),
                vision_config=paddleocr_vl.VisionConfig(
                    model_type="paddleocr_vl",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                ),
                model_type="paddleocr_vl",
            )
        )
        self._check_returns_input_embeddings_features(model, "paddleocr_vl")

    def test_phi3_v_input_embeddings(self):
        from mlx_vlm.models import phi3_v
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = phi3_v.Model(
            phi3_v.ModelConfig(
                text_config=phi3_v.TextConfig(),
                vision_config=phi3_v.VisionConfig(
                    model_type="phi3_v",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="phi3_v",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=32,
            )
        )
        # phi3_v uses 'inputs' as positional arg instead of 'input_ids'
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

        num_image_tokens = int((1 * 1 + 1) * 144 + 1 + (1 + 1) * 12)
        mm_input_ids = mx.array([[1, 2] + [-1] * num_image_tokens + [3]])
        image_sizes = mx.array([[336, 336]], dtype=mx.int64)

        model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))

        pixel_values = mx.random.normal(shape=(1, 2, 3, 336, 336))
        result_bf16 = model.get_input_embeddings(
            mm_input_ids, pixel_values=pixel_values, image_sizes=image_sizes
        )
        self.assertIsInstance(result_bf16, InputEmbeddingsFeatures)
        self.assertIsNotNone(result_bf16.inputs_embeds)
        self.assertEqual(result_bf16.inputs_embeds.dtype, mx.bfloat16)

    def test_phi4mm_input_embeddings(self):
        from mlx_vlm.models import phi4mm

        model = phi4mm.Model(
            phi4mm.ModelConfig(
                text_config=phi4mm.TextConfig(
                    model_type="phi4mm",
                    max_position_embeddings=2048,
                ),
                vision_config=phi4mm.VisionConfig(
                    model_type="siglip2_vision_model",
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    patch_size=14,
                    image_size=28,
                    num_channels=3,
                    layer_norm_eps=1e-6,
                ),
                model_type="phi4mm",
                vocab_size=256,
                hidden_size=64,
                num_hidden_layers=2,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                rms_norm_eps=1e-5,
                mm_hidden_size=32,
                image_token_index=-200,
                audio_token_index=200,
                audio_processor={
                    "config": {
                        "attention_dim": 32,
                        "attention_heads": 4,
                        "num_blocks": 2,
                        "linear_units": 64,
                        "input_size": 80,
                        "time_reduction": 8,
                        "kernel_size": 3,
                        "conv_channels": 32,
                        "ext_pw_out_channel": 32,
                        "depthwise_seperable_out_channel": 32,
                    }
                },
            )
        )
        self._check_returns_input_embeddings_features(model, "phi4mm")

    def test_qwen3_vl_moe_input_embeddings(self):
        from mlx_vlm.models import qwen3_vl_moe

        model = qwen3_vl_moe.Model(
            qwen3_vl_moe.ModelConfig(
                text_config=qwen3_vl_moe.TextConfig(
                    model_type="qwen3_vl_moe",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_experts=2,
                    num_experts_per_tok=1,
                    decoder_sparse_step=1,
                    mlp_only_layers=[],
                    moe_intermediate_size=16,
                    rms_norm_eps=1e-5,
                    vocab_size=32,
                    num_key_value_heads=2,
                    head_dim=8,
                    rope_theta=10000.0,
                    max_position_embeddings=128,
                ),
                vision_config=qwen3_vl_moe.VisionConfig(
                    model_type="qwen3_vl_moe",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    out_hidden_size=16,
                    num_heads=2,
                ),
                model_type="qwen3_vl_moe",
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_vl_moe")

    def test_qwen3_omni_moe_input_embeddings(self):
        from mlx_vlm.models import qwen3_omni_moe

        text_config = qwen3_omni_moe.TextConfig(
            num_hidden_layers=1,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            moe_intermediate_size=16,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            max_position_embeddings=128,
        )
        model = qwen3_omni_moe.Model(
            qwen3_omni_moe.ModelConfig(
                thinker_config=qwen3_omni_moe.ThinkerConfig(
                    text_config=text_config,
                    vision_config=qwen3_omni_moe.VisionConfig(
                        model_type="qwen3_omni_moe_vision_encoder",
                        depth=1,
                        hidden_size=16,
                        intermediate_size=32,
                        out_hidden_size=16,
                        num_heads=2,
                    ),
                    audio_config=qwen3_omni_moe.AudioConfig(
                        d_model=16,
                        encoder_layers=1,
                        encoder_attention_heads=2,
                        encoder_ffn_dim=32,
                        num_hidden_layers=1,
                        output_dim=16,
                    ),
                ),
                talker_config=qwen3_omni_moe.TalkerConfig(
                    text_config=text_config,
                    code_predictor_config=qwen3_omni_moe.CodePredictorConfig(
                        num_hidden_layers=1,
                        hidden_size=16,
                        intermediate_size=32,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                    ),
                    thinker_hidden_size=16,
                ),
                code2wav_config=qwen3_omni_moe.Code2WavConfig(),
                model_type="qwen3_omni_moe",
                enable_audio_output=False,
            )
        )
        self._check_returns_input_embeddings_features(model, "qwen3_omni_moe")

    def test_internvl_chat_input_embeddings(self):
        from mlx_vlm.models import internvl_chat

        model = internvl_chat.Model(
            internvl_chat.ModelConfig(
                text_config=internvl_chat.TextConfig(
                    model_type="qwen2",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    max_window_layers=1,
                    hidden_act="silu",
                ),
                vision_config=internvl_chat.VisionConfig(
                    model_type="intern_vit_6b",
                    num_hidden_layers=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                    num_channels=3,
                ),
                model_type="internvl_chat",
                image_token_index=31,
            )
        )
        self._check_returns_input_embeddings_features(model, "internvl_chat")

    def test_glm_ocr_input_embeddings(self):
        from mlx_vlm.models import glm_ocr

        model = glm_ocr.Model(
            glm_ocr.ModelConfig(
                text_config=glm_ocr.TextConfig(
                    model_type="glm_ocr_text",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    vocab_size=32,
                    num_key_value_heads=2,
                    head_dim=8,
                    rms_norm_eps=1e-5,
                    max_position_embeddings=1000,
                    rope_parameters={
                        "rope_type": "default",
                        "mrope_section": [2, 3, 3],
                        "partial_rotary_factor": 1.0,
                        "rope_theta": 10000,
                    },
                ),
                vision_config=glm_ocr.VisionConfig(
                    model_type="glm_ocr_vision",
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    out_hidden_size=16,
                    patch_size=14,
                    in_channels=3,
                    rms_norm_eps=1e-5,
                ),
                model_type="glm_ocr",
                image_token_id=31,
                vocab_size=32,
            )
        )
        self._check_returns_input_embeddings_features(model, "glm_ocr")

    def test_phi4_siglip_input_embeddings(self):
        from mlx_vlm.models import phi4_siglip
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = phi4_siglip.Model(
            phi4_siglip.ModelConfig(
                text_config=phi4_siglip.TextConfig(
                    model_type="phi4-siglip",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=phi4_siglip.VisionConfig(
                    model_type="siglip2_vision_model",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    patch_size=14,
                    num_patches=256,
                ),
                model_type="phi4-siglip",
                mm_hidden_size=16,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

    def test_moondream3_input_embeddings(self):
        from mlx_vlm.models import moondream3
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = moondream3.Model(
            moondream3.ModelConfig(
                text_config=moondream3.TextConfig(
                    model_type="moondream3",
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=16,
                    vocab_size=256,
                    rope_dim=8,
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=32,
                    moe_start_layer=2,
                ),
                vision_config=moondream3.VisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    patch_size=14,
                    crop_size=28,
                    proj_inner_dim=64,
                    proj_out_dim=64,
                ),
                model_type="moondream3",
            )
        )
        # Text-only
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)
        self.assertIsNone(result.attention_mask_4d)

        # With vision: should return prefix attention mask
        num_patches = (28 // 14) ** 2  # 4 patches
        input_ids_vis = mx.zeros((1, 1 + num_patches + 2), dtype=mx.int32)
        pixel_values = mx.random.uniform(shape=(1, 28, 28, 3))
        result_vis = model.get_input_embeddings(
            input_ids_vis,
            pixel_values=pixel_values,
            num_crops=[1],
            crop_layouts=[(1, 1)],
        )
        self.assertIsInstance(result_vis, InputEmbeddingsFeatures)
        self.assertIsNotNone(result_vis.inputs_embeds)
        self.assertIsNotNone(result_vis.attention_mask_4d)

    def test_granite_vision_input_embeddings(self):
        from mlx_vlm.models import granite_vision
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = granite_vision.Model(
            granite_vision.ModelConfig(
                text_config=granite_vision.TextConfig(
                    model_type="granite",
                    hidden_size=16,
                    num_hidden_layers=1,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=granite_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                model_type="granite_vision",
                vision_feature_layer=-1,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)

    def test_granite4_vision_input_embeddings(self):
        from mlx_vlm.models import granite4_vision
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        model = granite4_vision.Model(
            granite4_vision.ModelConfig(
                text_config=granite4_vision.TextConfig(
                    model_type="granitemoehybrid",
                    hidden_size=64,
                    num_hidden_layers=1,
                    intermediate_size=128,
                    shared_intermediate_size=128,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    vocab_size=32,
                    rms_norm_eps=1e-5,
                ),
                vision_config=granite4_vision.VisionConfig(
                    model_type="siglip_vision_model",
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=16,
                ),
                model_type="granite4_vision",
                deepstack_layer_map=[[-1, 0]],
                use_spatial_sampling=False,
                downsample_rate="2/2",
                use_image_newline_parameter=False,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(input_ids)
        self.assertIsInstance(result, InputEmbeddingsFeatures)
        self.assertIsNotNone(result.inputs_embeds)


class TestChunkedPrefillRoPE(unittest.TestCase):
    """Test chunked prefill RoPE position ID generation for vision-language models."""

    def test_ernie_chunked_prefill_rope(self):
        """Test ERNIE 4.5 MoE VL chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import ernie4_5_moe_vl

        text_config = ernie4_5_moe_vl.TextConfig(
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
        )
        vision_config = ernie4_5_moe_vl.VisionConfig(
            embed_dim=256,
            hidden_size=256,
            num_heads=8,
            patch_size=14,
            spatial_merge_size=2,
        )
        model_config = ernie4_5_moe_vl.ModelConfig(
            model_type="ernie4_5_moe_vl",
            hidden_size=256,
            vision_config=vision_config,
            text_config=text_config,
            im_patch_id=100,
            image_token_id=100,
            video_token_id=101,
            image_start_token_id=99,
            vision_start_token_id=99,
        )
        lm = ernie4_5_moe_vl.LanguageModel(text_config, model_config)

        input_ids = mx.array([[1, 2, 3, 99, 100, 100, 100, 100, 5, 6, 7]])
        image_grid_thw = mx.array([[1, 4, 4]])
        position_ids, _ = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[1], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        full_input = [1, 2, 3, 99, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10]
        chunked_input = full_input[:8]
        chunked_input_ids = mx.array([chunked_input])
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[1], len(chunked_input))

        # Position IDs have correct 3D shape for MRoPE
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 1)  # batch size
        self.assertEqual(position_ids.shape[2], 3)  # 3D positions (T, H, W)

    def test_glm4v_chunked_prefill_rope(self):
        """Test GLM4V chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v_text",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=256,
        )
        vision_config = glm4v.VisionConfig(
            model_type="glm4v_vision",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        model_config = glm4v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v",
            vocab_size=64,
            image_token_id=61,
            image_token_index=61,
            video_token_id=62,
            video_token_index=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
            pad_token_id=0,
        )
        lm = glm4v.LanguageModel(text_config, model_config)

        input_ids = mx.array([[10, 60, 61, 11, 12, 13, 14, 15]], dtype=mx.int32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        position_ids, rope_deltas = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[2], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        chunked_input_ids = input_ids[:, :4]
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[2], chunked_input_ids.shape[1])

        # Position IDs have expected MRoPE shape
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 3)  # MRoPE dimensions
        self.assertEqual(position_ids.shape[1], 1)  # batch size

        # Regression guard: full-length mask with chunked inputs should not fail
        full_mask = mx.ones((1, input_ids.shape[1]), dtype=mx.int32)
        lm._position_ids = position_ids
        lm._rope_deltas = rope_deltas
        outputs = lm(chunked_input_ids, mask=full_mask, image_grid_thw=image_grid_thw)
        self.assertEqual(
            outputs.logits.shape,
            (1, chunked_input_ids.shape[1], text_config.vocab_size),
        )

    def test_glm4v_get_rope_index_per_row_deltas(self):
        from mlx_vlm.models import glm4v

        text_config = glm4v.TextConfig(
            model_type="glm4v_text",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=256,
        )
        vision_config = glm4v.VisionConfig(
            model_type="glm4v_vision",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        lm = glm4v.LanguageModel(
            text_config,
            glm4v.ModelConfig(
                text_config=text_config,
                vision_config=vision_config,
                model_type="glm4v",
                vocab_size=64,
                image_token_id=61,
                image_token_index=61,
                video_token_id=62,
                video_token_index=62,
                vision_start_token_id=60,
                vision_end_token_id=59,
                pad_token_id=0,
            ),
        )

        input_ids = mx.array(
            [
                [10, 60, 61, 61, 61, 61, 11, 12],
                [10, 11, 12, 13, 14, 15, 16, 17],
            ],
            dtype=mx.int32,
        )
        _, rope_deltas = lm.get_rope_index(
            input_ids, mx.array([[1, 4, 4]], dtype=mx.int32)
        )
        self.assertEqual(rope_deltas.shape, (2, 1))
        self.assertEqual(rope_deltas[1, 0].item(), 0)
        self.assertNotEqual(rope_deltas[0, 0].item(), rope_deltas[1, 0].item())

        input_ids = mx.array(
            [[0, 0, 10, 11, 12, 13], [10, 11, 12, 13, 14, 15]], dtype=mx.int32
        )
        attention_mask = mx.array(
            [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=mx.int32
        )
        position_ids, rope_deltas = lm.get_rope_index(
            input_ids, image_grid_thw=None, attention_mask=attention_mask
        )
        self.assertEqual(rope_deltas.shape, (2, 1))
        self.assertEqual(position_ids.shape, (3, 2, 6))
        self.assertEqual(rope_deltas[0, 0].item(), -2)
        self.assertEqual(rope_deltas[1, 0].item(), 0)

    def test_glm4v_moe_chunked_prefill_rope(self):
        """Test GLM4V-MoE chunked prefill RoPE position ID generation."""
        from mlx_vlm.models import glm4v_moe

        text_config = glm4v_moe.TextConfig(
            model_type="glm4v_moe_text",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            max_position_embeddings=256,
            moe_intermediate_size=16,
            norm_topk_prob=True,
            num_attention_heads=2,
            n_group=1,
            head_dim=8,
            topk_group=1,
            n_shared_experts=1,
            n_routed_experts=2,
            routed_scaling_factor=1.0,
            num_experts_per_tok=1,
            first_k_dense_replace=0,
            num_hidden_layers=1,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            use_qk_norm=False,
            attention_bias=False,
            partial_rotary_factor=0.5,
            rope_theta=10000.0,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            tie_word_embeddings=False,
        )
        vision_config = glm4v_moe.VisionConfig(
            model_type="glm4v_moe",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=14,
            out_hidden_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            image_size=28,
        )
        model_config = glm4v_moe.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm4v_moe",
            vocab_size=64,
            image_token_id=61,
            image_token_index=61,
            video_token_id=62,
            video_token_index=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
            pad_token_id=0,
        )
        lm = glm4v_moe.LanguageModel(text_config, model_config)

        input_ids = mx.array([[10, 60, 61, 11, 12, 13, 14, 15]], dtype=mx.int32)
        image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)
        position_ids, rope_deltas = lm.get_rope_index(input_ids, image_grid_thw)

        # Position IDs length matches input sequence length
        self.assertEqual(position_ids.shape[2], input_ids.shape[1])

        # Chunked input position IDs match partial sequence
        chunked_input_ids = input_ids[:, :4]
        chunked_position_ids, _ = lm.get_rope_index(chunked_input_ids, image_grid_thw)
        self.assertEqual(chunked_position_ids.shape[2], chunked_input_ids.shape[1])

        # Position IDs have expected MRoPE shape
        self.assertEqual(len(position_ids.shape), 3)
        self.assertEqual(position_ids.shape[0], 3)  # MRoPE dimensions
        self.assertEqual(position_ids.shape[1], 1)  # batch size

        # Regression guard: full-length mask with chunked inputs should not fail
        full_mask = mx.ones((1, input_ids.shape[1]), dtype=mx.int32)
        lm._position_ids = position_ids
        lm._rope_deltas = rope_deltas
        outputs = lm(chunked_input_ids, mask=full_mask, image_grid_thw=image_grid_thw)
        self.assertEqual(
            outputs.logits.shape,
            (1, chunked_input_ids.shape[1], text_config.vocab_size),
        )


class TestMiniCPMO(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        from mlx_vlm.models import minicpmo

        text_config = minicpmo.TextConfig(
            model_type="minicpmo",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=4,
            head_dim=16,
            rope_theta=10000.0,
            max_position_embeddings=2048,
        )
        vision_config = minicpmo.VisionConfig(
            model_type="siglip_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=28,
            patch_size=14,
        )
        return minicpmo.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            query_num=4,
        )

    def test_minicpmo_config_from_root_fields(self):
        from mlx_vlm.models import minicpmo

        cfg = {
            "model_type": "minicpmo",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960,
            "query_num": 64,
            "vision_config": {
                "model_type": "siglip",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_channels": 3,
                "image_size": 448,
                "patch_size": 14,
            },
        }
        model_config = minicpmo.ModelConfig.from_dict(cfg)
        self.assertEqual(model_config.text_config.hidden_size, 4096)
        self.assertEqual(model_config.vision_config.model_type, "siglip_vision_model")
        self.assertEqual(model_config.query_num, 64)

    def test_minicpmo_sanitize_key_mapping_and_qkv_split(self):
        from mlx_vlm.models import minicpmo

        model = minicpmo.Model(self._tiny_config())
        weights = {
            "llm.model.embed_tokens.weight": mx.zeros((10, 10)),
            "llm.lm_head.weight": mx.zeros((10, 10)),
            "vpm.embeddings.patch_embedding.weight": mx.zeros((8, 3, 14, 14)),
            "resampler.attn.in_proj_weight": mx.zeros((192, 64)),
            "resampler.attn.in_proj_bias": mx.zeros((192,)),
            "apm.conv1.weight": mx.zeros((1, 1)),
        }

        sanitized = model.sanitize(weights)
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)
        self.assertIn("vision_tower.embeddings.patch_embedding.weight", sanitized)
        self.assertNotIn("apm.conv1.weight", sanitized)

        self.assertIn("resampler.attn.q_proj.weight", sanitized)
        self.assertIn("resampler.attn.k_proj.weight", sanitized)
        self.assertIn("resampler.attn.v_proj.weight", sanitized)
        self.assertIn("resampler.attn.q_proj.bias", sanitized)
        self.assertIn("resampler.attn.k_proj.bias", sanitized)
        self.assertIn("resampler.attn.v_proj.bias", sanitized)

    def test_minicpmo_sanitize_audio_conv_layout(self):
        from mlx_vlm.models import minicpmo

        model = minicpmo.Model(self._tiny_config())
        weights = {
            "apm.conv1.weight": mx.zeros((8, 80, 3)),
            "apm.conv2.weight": mx.zeros((8, 8, 3)),
        }

        sanitized = model.sanitize(weights)
        self.assertEqual(sanitized["audio_tower.conv1.weight"].shape, (8, 3, 80))
        self.assertEqual(sanitized["audio_tower.conv2.weight"].shape, (8, 3, 8))


class TestPhi4MM(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        from mlx_vlm.models.phi4mm.config import ModelConfig, TextConfig, VisionConfig

        text_config = TextConfig(
            model_type="phi4mm",
            max_position_embeddings=2048,
        )
        vision_config = VisionConfig(
            model_type="siglip2_vision_model",
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            patch_size=14,
            image_size=28,
            num_channels=3,
            layer_norm_eps=1e-6,
        )
        return ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="phi4mm",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            mm_hidden_size=32,
            image_token_index=-200,
            audio_token_index=200,
            vision_lora={"r": 4, "lora_alpha": 8},
            speech_lora={"r": 4, "lora_alpha": 8},
            audio_processor={
                "config": {
                    "attention_dim": 32,
                    "attention_heads": 4,
                    "num_blocks": 2,
                    "linear_units": 64,
                    "input_size": 80,
                    "time_reduction": 8,
                    "kernel_size": 3,
                    "conv_channels": 32,
                    "ext_pw_out_channel": 32,
                    "depthwise_seperable_out_channel": 32,
                }
            },
        )

    def test_phi4mm_sanitize_lora_keys(self):
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        hidden = config.hidden_size
        qkv_size = config.num_attention_heads * (
            hidden // config.num_attention_heads
        ) + 2 * config.num_key_value_heads * (hidden // config.num_attention_heads)
        lora_r = 4

        weights = {
            # Base layer weight
            "model.layers.0.self_attn.qkv_proj.base_layer.weight": mx.zeros(
                (qkv_size, hidden)
            ),
            # Vision LoRA
            "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight": mx.zeros(
                (lora_r, hidden)
            ),
            "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight": mx.zeros(
                (qkv_size, lora_r)
            ),
            # Speech LoRA
            "model.layers.0.self_attn.qkv_proj.lora_A.speech.weight": mx.zeros(
                (lora_r, hidden)
            ),
            "model.layers.0.self_attn.qkv_proj.lora_B.speech.weight": mx.zeros(
                (qkv_size, lora_r)
            ),
            # Embed tokens
            "model.embed_tokens.weight": mx.zeros((config.vocab_size, hidden)),
        }

        sanitized = model.sanitize(weights)

        # Base layer should be merged with vision LoRA by default
        self.assertIn(
            "language_model.model.layers.0.self_attn.qkv_proj.weight", sanitized
        )
        # LoRA keys should not appear in sanitized output
        for k in sanitized:
            self.assertNotIn("lora_A", k)
            self.assertNotIn("lora_B", k)
            self.assertNotIn("base_layer", k)

        # Speech LoRA should be stored for runtime switching
        self.assertTrue(len(model._speech_lora_a) > 0)
        self.assertTrue(len(model._speech_lora_b) > 0)
        self.assertTrue(len(model._base_weights) > 0)

    def test_phi4mm_quant_predicate_skips_multimodal(self):
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        predicate = model.quant_predicate

        # Language model layers should be quantized
        self.assertTrue(
            predicate(
                "language_model.model.layers.0.self_attn.qkv_proj", nn.Linear(4, 4)
            )
        )

        # Multimodal modules should NOT be quantized
        self.assertFalse(
            predicate("audio_encoder.encoders.0.attn.linear_q", nn.Linear(4, 4))
        )
        self.assertFalse(predicate("audio_projection.speech.proj_0", nn.Linear(4, 4)))
        self.assertFalse(predicate("mm_projector.0", nn.Linear(4, 4)))
        self.assertFalse(
            predicate("vision_tower.vision_tower.encoder.layers.0", nn.Linear(4, 4))
        )

    def test_phi4mm_quant_predicate_clears_lora(self):
        """Accessing quant_predicate should merge LoRAs and clear stored copies."""
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)

        hidden = config.hidden_size
        head_dim = hidden // config.num_attention_heads
        qkv_size = (
            config.num_attention_heads * head_dim
            + 2 * config.num_key_value_heads * head_dim
        )
        key = "language_model.model.layers.0.self_attn.qkv_proj.weight"
        lora_r = 4

        # Set up LoRA weights with correct shapes
        model._base_weights = {key: mx.ones((qkv_size, hidden))}
        model._speech_lora_a = {key: mx.zeros((lora_r, hidden))}
        model._speech_lora_b = {key: mx.zeros((qkv_size, lora_r))}
        model._speech_lora_scale = 1.0
        model._vision_lora_a = {key: mx.zeros((lora_r, hidden))}
        model._vision_lora_b = {key: mx.zeros((qkv_size, lora_r))}
        model._vision_lora_scale = 1.0
        model._active_lora = "vision"

        # Accessing the property triggers merge and clears LoRA dicts
        _ = model.quant_predicate

        self.assertEqual(len(model._base_weights), 0)
        self.assertEqual(len(model._speech_lora_a), 0)
        self.assertEqual(len(model._speech_lora_b), 0)
        self.assertEqual(len(model._vision_lora_a), 0)
        self.assertEqual(len(model._vision_lora_b), 0)

    def test_phi4mm_set_modality_skips_when_no_lora(self):
        """set_modality should no-op when _base_weights is empty (quantized model)."""
        from mlx_vlm.models import phi4mm

        config = self._tiny_config()
        model = phi4mm.Model(config)
        model._base_weights = {}

        # Should not raise even with modality flags set
        model.set_modality(has_image=True, has_audio=True)


class TestSam3(unittest.TestCase):
    # ─── SAM3 Tests ────────────────────────────────────────────

    def test_sam3_config(self):
        """Config parses the nested detector/tracker structure."""
        from mlx_vlm.models import sam3

        config = sam3.ModelConfig()
        self.assertEqual(config.model_type, "sam3_video")
        self.assertEqual(
            config.detector_config.vision_config.backbone_config.hidden_size, 1024
        )
        self.assertEqual(config.detector_config.text_config.hidden_size, 1024)
        self.assertEqual(config.detector_config.detr_encoder_config.num_layers, 6)
        self.assertEqual(config.detector_config.detr_decoder_config.num_queries, 200)
        self.assertEqual(config.tracker_config.memory_attention_num_layers, 4)

    def test_sam3_vision_encoder(self):
        """ViT backbone + FPN produce correct shapes."""
        from mlx_vlm.models.sam3.config import VisionEncoderConfig, ViTConfig
        from mlx_vlm.models.sam3.vision import VisionEncoder

        vit_cfg = ViTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            image_size=112,
            patch_size=14,
            window_size=4,
            global_attn_indexes=[1],
            pretrain_image_size=56,
        )
        vision_cfg = VisionEncoderConfig(
            backbone_config=vit_cfg,
            fpn_hidden_size=32,
            scale_factors=[2.0, 1.0],
        )
        encoder = VisionEncoder(vision_cfg)

        x = mx.random.normal((1, 112, 112, 3))
        fpn_out = encoder(x)
        self.assertIsInstance(fpn_out, list)
        self.assertEqual(len(fpn_out), 2)
        # 1x scale = 8x8 (112/14), 2x scale = 16x16
        self.assertEqual(fpn_out[1].shape, (1, 8, 8, 32))
        self.assertEqual(fpn_out[0].shape, (1, 16, 16, 32))

    def test_sam3_text_encoder(self):
        """CLIP text encoder produces correct shapes."""
        from mlx_vlm.models.sam3.config import TextEncoderConfig
        from mlx_vlm.models.sam3.text_encoder import TextEncoder

        cfg = TextEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=16,
            projection_dim=32,
        )
        encoder = TextEncoder(cfg, d_model=32)

        input_ids = mx.array([[1, 2, 3, 4, 0, 0]])
        out = encoder(input_ids)
        self.assertEqual(out.shape, (1, 6, 64))

    def test_sam3_detr_encoder(self):
        """DETR encoder with text cross-attention."""
        from mlx_vlm.models.sam3.config import DETREncoderConfig
        from mlx_vlm.models.sam3.encoder import DETREncoder

        cfg = DETREncoderConfig(
            hidden_size=64, num_layers=2, num_attention_heads=2, intermediate_size=128
        )
        encoder = DETREncoder(cfg)

        src = mx.random.normal((1, 16, 64))
        pos = mx.random.normal((1, 16, 64))
        text = mx.random.normal((1, 4, 64))

        out = encoder(src, pos, text)
        self.assertEqual(out.shape, (1, 16, 64))

    def test_sam3_detr_decoder(self):
        """DETR decoder with box refinement and presence token."""
        from mlx_vlm.models.sam3.config import DETRDecoderConfig
        from mlx_vlm.models.sam3.decoder import DETRDecoder

        cfg = DETRDecoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=2,
            num_queries=10,
            intermediate_size=128,
        )
        decoder = DETRDecoder(cfg)

        memory = mx.random.normal((1, 16, 64))
        text = mx.random.normal((1, 4, 64))
        pos = mx.random.normal((1, 16, 64))

        hs, boxes, presence = decoder(memory, text, pos, spatial_shape=(4, 4))
        self.assertEqual(hs.shape, (2, 1, 10, 64))  # (L, B, Q, D)
        self.assertEqual(boxes.shape, (2, 1, 10, 4))  # (L, B, Q, 4)
        self.assertEqual(presence.shape, (2, 1, 1))  # (L, B, 1)

    def test_sam3_dot_product_scoring(self):
        """DotProductScoring with scale and clamp."""
        from mlx_vlm.models.sam3.segmentation import DotProductScoring

        scorer = DotProductScoring(64)
        hs = mx.random.normal((2, 1, 10, 64))  # (L, B, Q, D)
        text = mx.random.normal((1, 4, 64))
        mask = mx.array([[1, 1, 1, 0]])

        scores = scorer(hs, text, mask)
        self.assertEqual(scores.shape, (2, 1, 10, 1))
        # Scores should be clamped to [-12, 12]
        scores_np = scores.tolist()
        for layer in scores_np:
            for batch in layer:
                for query in batch:
                    for val in query:
                        self.assertGreaterEqual(val, -12.0)
                        self.assertLessEqual(val, 12.0)

    def test_sam3_mask_decoder(self):
        """Mask decoder produces correct mask resolution."""
        from mlx_vlm.models.sam3.config import DetectorMaskDecoderConfig
        from mlx_vlm.models.sam3.segmentation import MaskDecoder

        cfg = DetectorMaskDecoderConfig(hidden_size=32, num_upsampling_stages=2)
        decoder = MaskDecoder(cfg)

        queries = mx.random.normal((1, 10, 32))
        # 2 FPN levels: 8x8, 4x4
        features = [mx.random.normal((1, 8, 8, 32)), mx.random.normal((1, 4, 4, 32))]
        encoder_hs = mx.random.normal((1, 16, 32))

        out = decoder(queries, features, encoder_hidden_states=encoder_hs)
        self.assertIn("pred_masks", out)
        self.assertIn("semantic_seg", out)
        self.assertEqual(out["pred_masks"].shape[0], 1)
        self.assertEqual(out["pred_masks"].shape[1], 10)

    def test_sam3_full_model(self):
        """Full SAM3 model instantiation and forward pass."""
        from mlx_vlm.models import sam3

        config = sam3.ModelConfig(
            detector_config=sam3.DetectorConfig(
                vision_config=sam3.VisionEncoderConfig(
                    backbone_config=sam3.ViTConfig(
                        hidden_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        intermediate_size=128,
                        image_size=112,
                        patch_size=14,
                        window_size=4,
                        global_attn_indexes=[1],
                        pretrain_image_size=56,
                    ),
                    fpn_hidden_size=32,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                ),
                text_config=sam3.TextEncoderConfig(
                    hidden_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    intermediate_size=128,
                    vocab_size=100,
                    max_position_embeddings=16,
                    projection_dim=32,
                ),
                detr_encoder_config=sam3.DETREncoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                ),
                detr_decoder_config=sam3.DETRDecoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    num_queries=10,
                    intermediate_size=64,
                ),
                geometry_encoder_config=sam3.GeometryEncoderConfig(
                    hidden_size=32,
                    num_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                ),
                mask_decoder_config=sam3.DetectorMaskDecoderConfig(
                    hidden_size=32,
                    num_upsampling_stages=3,
                ),
            ),
        )

        model = sam3.Model(config)

        pixel_values = mx.random.normal((1, 112, 112, 3))
        input_ids = mx.array([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        attention_mask = mx.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        outputs = model.detect(pixel_values, input_ids, attention_mask)
        mx.eval(outputs)

        self.assertIn("pred_logits", outputs)
        self.assertIn("pred_boxes", outputs)
        self.assertIn("pred_masks", outputs)
        self.assertIn("presence_logits", outputs)
        self.assertEqual(outputs["pred_logits"].shape, (1, 10))
        self.assertEqual(outputs["pred_boxes"].shape, (1, 10, 4))

    def test_sam3_sanitize(self):
        """Sanitize transposes conv weights correctly."""
        from mlx_vlm.models.sam3.sam3 import Model

        weights = {
            "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight": mx.zeros(
                (64, 3, 14, 14)
            ),
            "detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.0.weight": mx.zeros(
                (128, 64, 2, 2)
            ),
            "tracker_model.memory_temporal_positional_encoding": mx.zeros(
                (7, 1, 1, 32)
            ),
            "detector_model.detr_encoder.layers.0.self_attn.q_proj.weight": mx.zeros(
                (64, 64)
            ),
        }

        sanitized = Model.sanitize(weights)

        # Conv2d: (out, in, H, W) -> (out, H, W, in)
        self.assertEqual(
            sanitized[
                "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
            ].shape,
            (64, 14, 14, 3),
        )
        # ConvTranspose2d: (in, out, H, W) -> (out, H, W, in)
        self.assertEqual(
            sanitized[
                "detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.0.weight"
            ].shape,
            (64, 2, 2, 128),
        )
        # Non-conv 4D param: unchanged
        self.assertEqual(
            sanitized["tracker_model.memory_temporal_positional_encoding"].shape,
            (7, 1, 1, 32),
        )
        # 2D weight: unchanged
        self.assertEqual(
            sanitized[
                "detector_model.detr_encoder.layers.0.self_attn.q_proj.weight"
            ].shape,
            (64, 64),
        )

    def test_sam3_quant_predicate(self):
        """quant_predicate skips convs, small embeddings, and odd dimensions."""
        from mlx_vlm.models.sam3.sam3 import Model

        class FakeModule:
            def __init__(self, shape):
                self.weight = mx.zeros(shape)

        # Should quantize: large linear in DETR
        self.assertTrue(
            Model.quant_predicate(
                "detector_model.detr_encoder.layers.0.self_attn.q_proj",
                FakeModule((256, 256)),
            )
        )
        # Should skip: conv layers
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.vision_encoder.neck.fpn_layers.0.proj1",
                FakeModule((256, 256)),
            )
        )
        # Should skip: small embedding
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.detr_decoder.query_embed", FakeModule((200, 256))
            )
        )
        # Should quantize: vision encoder linear (not skipped for better compression)
        self.assertTrue(
            Model.quant_predicate(
                "detector_model.vision_encoder.backbone.layers.0.attention.q_proj",
                FakeModule((1024, 1024)),
            )
        )
        # Should skip: patch_embeddings (conv)
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection",
                FakeModule((1024, 1024)),
            )
        )
        # Should skip: odd dimension
        self.assertFalse(
            Model.quant_predicate(
                "detector_model.geometry_encoder.boxes_pos_enc_project",
                FakeModule((256, 258)),
            )
        )

    def test_sam3_position_encoding(self):
        """Sinusoidal position encoding and 2D RoPE produce correct shapes."""
        from mlx_vlm.models.sam3.position import (
            PositionEmbeddingSine,
            compute_axial_cis,
        )

        pos_enc = PositionEmbeddingSine(num_pos_feats=32)
        x = mx.random.normal((1, 8, 8, 64))
        pos = pos_enc(x)
        self.assertEqual(pos.shape, (1, 8, 8, 64))

        cos, sin = compute_axial_cis(64, 8, 8)
        self.assertEqual(cos.shape, (64, 64))
        self.assertEqual(sin.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
