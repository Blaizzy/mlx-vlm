import inspect
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
            if model_type in ["qwen2_5_vl", "glm4v_moe", "glm4v", "hunyuan_vl"]:
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
        from types import SimpleNamespace

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

        # Regression check: runtime image token id from tokenizer should be used

        dummy_model = SimpleNamespace(config=model.config)
        dummy_model.config.media_placeholder_token_id = 163605
        dummy_model.config.image_token_index = 163605

        input_ids = mx.array([[11, 163592, 12, 163592, 13]], dtype=mx.int32)
        inputs_embeds = mx.zeros((1, 5, 8), dtype=mx.float32)
        image_features = mx.ones((2, 8), dtype=mx.float32)

        with self.assertRaises(ValueError):
            kimi_vl.Model._prepare_inputs_for_multimodal(
                dummy_model, image_features, inputs_embeds, input_ids
            )

        merged = kimi_vl.Model._prepare_inputs_for_multimodal(
            dummy_model,
            image_features,
            inputs_embeds,
            input_ids,
            image_token_id=163592,
        )
        self.assertEqual(merged.shape, inputs_embeds.shape)
        self.assertTrue(mx.all(merged[0, 1] == 1).item())
        self.assertTrue(mx.all(merged[0, 3] == 1).item())

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


if __name__ == "__main__":
    unittest.main()
