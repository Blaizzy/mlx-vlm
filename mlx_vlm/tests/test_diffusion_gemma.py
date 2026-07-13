import contextlib
import importlib
import io
import json
import unittest
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
from mlx_vlm.utils import StoppingCriteria


def tiny_config_dict():
    return {
        "model_type": "diffusion_gemma",
        "canvas_length": 3,
        "image_token_id": 258880,
        "text_config": {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_experts": 4,
            "top_k_experts": 2,
            "use_bidirectional_attention": None,
            "final_logit_softcapping": 30.0,
        },
        "vision_config": None,
        "generation_config": {
            "max_denoising_steps": 1,
            "sampler_config": {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": 0.1,
            },
            "linear_temperature_schedule_config": {
                "_cls_name": "LinearTemperatureScheduleConfig",
                "t_min": 0.4,
                "t_max": 0.8,
            },
        },
    }


class FakeTokenizer:
    all_special_ids = []
    eos_token_ids = [999999]

    def __init__(self):
        self.stopping_criteria = StoppingCriteria([999999], self)

    def decode(self, tokens, **kwargs):
        return "".join(chr(65 + (int(token) % 26)) for token in tokens)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.detokenizer = NaiveStreamingDetokenizer(self.tokenizer)


class RecordingEncoder:
    def __init__(self, inner):
        self.inner = inner
        self.input_lengths = []
        self.attention_masks = []
        self.mm_token_type_ids = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, input_ids, *args, **kwargs):
        self.input_lengths.append(input_ids.shape[1])
        self.attention_masks.append(kwargs.get("attention_mask"))
        self.mm_token_type_ids.append(kwargs.get("mm_token_type_ids"))
        return self.inner(input_ids, *args, **kwargs)


class TinyDiffusionGemma4Tokenizer:
    image_token = "<image>"
    image_token_id = 60
    video_token = "<video>"
    video_token_id = 61
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    unk_token_id = 2
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.additional_special_tokens = []
        self.stc_token = None
        self.etc_token = None
        self.escape_token = None
        self.soc_token = None
        self.eoc_token = None
        self.special_tokens = {
            self.image_token: self.image_token_id,
            self.video_token: self.video_token_id,
            self.boi_token: 62,
            self.eoi_token: 63,
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }

    @property
    def all_special_ids(self):
        additional_ids = [
            self.convert_tokens_to_ids(token)
            for token in self.additional_special_tokens
        ]
        attr_ids = [
            self.convert_tokens_to_ids(token)
            for token in (
                self.stc_token,
                self.etc_token,
                self.escape_token,
                self.soc_token,
                self.eoc_token,
            )
            if token is not None
        ]
        return additional_ids + attr_ids

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, self.unk_token_id)

    def add_special_tokens(self, tokens):
        for token in tokens.get("additional_special_tokens", []):
            self.special_tokens[token] = self.video_token_id

    def __call__(self, text=None, **kwargs):
        del kwargs
        if isinstance(text, str):
            text = [text]
        rows = [self._encode(prompt) for prompt in text]
        max_len = max(len(row) for row in rows)
        input_ids = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        attention_mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _encode(self, text):
        ids = []
        i = 0
        specials = sorted(self.special_tokens, key=len, reverse=True)
        while i < len(text):
            for token in specials:
                if text.startswith(token, i):
                    ids.append(self.special_tokens[token])
                    i += len(token)
                    break
            else:
                if not text[i].isspace():
                    ids.append(10)
                i += 1
        return ids


class TinyVideoProcessor:
    model_input_names = ["pixel_values_videos"]

    def __call__(self, videos, fps=None):
        del videos, fps
        return {
            "pixel_values_videos": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "num_frames_per_video": [2],
            "num_soft_tokens_per_frame": [1],
            "frame_timestamps": [[0.0, 1.0]],
        }


class TinyImageProcessor:
    model_input_names = ["pixel_values"]

    def fetch_images(self, images):
        return images

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        return {"pixel_values": np.stack(images).astype(np.float32)}, [1] * len(images)


def tiny_diffusion_gemma_processor(image_processor=None):
    from mlx_vlm.models.diffusion_gemma import DiffusionGemma4Processor

    tokenizer = TinyDiffusionGemma4Tokenizer()
    processor = DiffusionGemma4Processor.__new__(DiffusionGemma4Processor)
    processor.image_processor = image_processor
    processor.tokenizer = tokenizer
    processor.video_processor = TinyVideoProcessor()
    processor.feature_extractor = None
    processor.image_seq_length = 280
    processor.audio_seq_length = 750
    processor.audio_ms_per_token = 40
    processor.image_token_id = tokenizer.image_token_id
    processor.image_token = tokenizer.image_token
    processor.video_token_id = tokenizer.video_token_id
    processor.video_token = tokenizer.video_token
    processor.boi_token = tokenizer.boi_token
    processor.eoi_token = tokenizer.eoi_token
    processor.audio_token_id = None
    processor.audio_token = ""
    processor.full_audio_sequence = None
    processor.full_image_sequence = ""
    return processor


class TestDiffusionGemma4(unittest.TestCase):
    def test_model_resolves_from_config(self):
        from mlx_vlm.utils import get_model_and_args

        arch, model_type = get_model_and_args(tiny_config_dict())

        self.assertEqual(model_type, "diffusion_gemma")
        self.assertEqual(arch.Model.__name__, "Model")

    def test_load_config_preserves_generation_config_for_model_config(self):
        from mlx_vlm.models.diffusion_gemma import ModelConfig
        from mlx_vlm.utils import load_config

        config = tiny_config_dict()
        generation_config = {
            "max_denoising_steps": 48,
            "sampler_config": {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": 0.1,
            },
        }

        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(config))
            Path(tmpdir, "generation_config.json").write_text(
                json.dumps(generation_config)
            )

            loaded = load_config(Path(tmpdir))

        self.assertEqual(loaded["model_type"], "diffusion_gemma")
        self.assertEqual(loaded["generation_config"], generation_config)
        self.assertEqual(
            ModelConfig.from_dict(loaded).generation_config,
            generation_config,
        )

    def test_auto_processor_loads_multimodal_processor(self):
        from transformers import AutoProcessor

        from mlx_vlm.models.diffusion_gemma import DiffusionGemma4Processor
        from mlx_vlm.models.gemma4.processing_gemma4 import (
            Gemma4ImageProcessor,
            Gemma4VideoProcessor,
        )

        tokenizer = TinyDiffusionGemma4Tokenizer()
        tokenizer.chat_template = None

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "diffusion_gemma"}),
                encoding="utf-8",
            )
            (model_dir / "processor_config.json").write_text(
                json.dumps(
                    {
                        "audio_ms_per_token": 40,
                        "audio_seq_length": 750,
                        "image_processor": {
                            "do_normalize": False,
                            "image_processor_type": "Gemma4ImageProcessor",
                            "max_soft_tokens": 140,
                            "patch_size": 16,
                            "pooling_kernel_size": 3,
                        },
                        "image_seq_length": 140,
                        "processor_class": "DiffusionGemma4Processor",
                        "video_processor": {
                            "max_soft_tokens": 70,
                            "num_frames": 8,
                            "video_processor_type": "Gemma4VideoProcessor",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "transformers.processing_utils.ProcessorMixin."
                    "check_argument_for_proper_class",
                    return_value=None,
                ),
            ):
                processor = AutoProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(processor, DiffusionGemma4Processor)
        self.assertIsInstance(processor.image_processor, Gemma4ImageProcessor)
        self.assertIsInstance(processor.video_processor, Gemma4VideoProcessor)
        self.assertEqual(processor.image_processor.max_soft_tokens, 140)
        self.assertEqual(processor.video_processor.num_frames, 8)
        self.assertEqual(
            DiffusionGemma4Processor.get_attributes(),
            ["image_processor", "tokenizer", "video_processor"],
        )

    def test_forward_shape(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)

        out = model(
            input_ids=mx.array([[2, 3, 4]]),
            canvas_ids=mx.array([[5, 6, 7]]),
        )
        mx.eval(out.logits)

        self.assertEqual(out.logits.shape, (1, 3, 64))
        self.assertEqual(len(model.make_cache()), 2)

    def test_video_token_type_ids_drive_vision_embeddings_without_video_token_id(self):
        from mlx_vlm.models.diffusion_gemma.language import EncoderModel

        class DummyDecoder:
            embed_scale = 1.0

            def embed_tokens(self, input_ids):
                ids = input_ids.astype(mx.float32)
                return mx.stack([ids, ids + 1000], axis=-1)

        class DummyEncoder:
            config = SimpleNamespace(image_token_id=60, video_token_id=None)
            text_config = SimpleNamespace(pad_token_id=0)
            decoder = DummyDecoder()

            def get_image_features(self, pixel_values):
                del pixel_values
                return mx.array([[[101.0, 102.0], [201.0, 202.0]]])

        input_ids = mx.array([[10, 61, 61, 11]], dtype=mx.int32)
        mm_token_type_ids = mx.array([[0, 2, 2, 0]], dtype=mx.int32)
        pixel_values = mx.zeros((2, 3, 4, 4), dtype=mx.float32)

        embeddings = EncoderModel._embed_inputs(
            DummyEncoder(),
            input_ids,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
        )
        mx.eval(embeddings)

        self.assertEqual(embeddings[0, 1].tolist(), [101.0, 102.0])
        self.assertEqual(embeddings[0, 2].tolist(), [201.0, 202.0])

    def test_self_conditioning_soft_embeddings_use_embedding_scale(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        decoder = model.model.decoder
        captured = {}

        class CaptureSelfConditioning:
            def __call__(self, inputs_embeds, self_conditioning_signal):
                captured["self_conditioning_signal"] = self_conditioning_signal
                return inputs_embeds

        decoder.self_conditioning = CaptureSelfConditioning()
        self_conditioning_logits = mx.zeros((1, 1, config.text_config.vocab_size))
        decoder._embed_canvas(
            mx.array([[2]], dtype=mx.int32),
            self_conditioning_logits,
        )

        expected = (
            mx.softmax(self_conditioning_logits, axis=-1, precise=True)
            @ decoder.embed_tokens.weight
        ) * decoder.embed_scale
        max_diff = mx.max(mx.abs(captured["self_conditioning_signal"] - expected))
        mx.eval(max_diff)

        self.assertLess(float(max_diff.item()), 1e-6)

    def test_precomputed_self_conditioning_embeddings_match_logits_path(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        decoder = model.model.decoder
        decoder.embed_tokens.weight = decoder.embed_tokens.weight.astype(mx.bfloat16)
        input_ids = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
        canvas_ids = mx.array([[6, 7, 8]], dtype=mx.int32)
        self_conditioning_logits = mx.linspace(
            -0.5,
            0.5,
            config.text_config.vocab_size * canvas_ids.shape[-1],
        ).reshape(1, canvas_ids.shape[-1], -1)
        stored_self_conditioning_logits = self_conditioning_logits.astype(
            decoder.embed_tokens.weight.dtype
        )

        self_conditioning_embeddings = model.diffusion_self_conditioning(
            self_conditioning_logits,
            model.diffusion_prepare_self_conditioning(),
        ).astype(decoder.embed_tokens.weight.dtype)
        logits_output = model(
            input_ids=input_ids,
            canvas_ids=canvas_ids,
            self_conditioning_logits=stored_self_conditioning_logits,
        ).logits
        embeddings_output = model(
            input_ids=input_ids,
            canvas_ids=canvas_ids,
            self_conditioning_embeddings=self_conditioning_embeddings,
        ).logits
        max_diff = mx.max(mx.abs(logits_output - embeddings_output))
        mx.eval(max_diff)

        self.assertLess(float(max_diff.item()), 1e-5)

    def test_transformers_58_logits_and_denoising_step_parity_if_available(self):
        try:
            import numpy as np
            import torch
            from transformers.cache_utils import DynamicCache
            from transformers.generation.logits_process import LogitsProcessorList
            from transformers.models.diffusion_gemma4.generation_diffusion_gemma4 import (
                LinearTemperatureScheduleConfig,
                LinearTemperatureScheduleLogitsProcessor,
            )
            from transformers.models.diffusion_gemma4.modeling_diffusion_gemma4 import (
                DiffusionGemma4Config,
                DiffusionGemma4ModelForBlockDiffusion,
            )
        except Exception as exc:
            self.skipTest(
                f"Transformers 5.8 DiffusionGemma4 reference unavailable: {exc}"
            )

        from mlx_vlm.generate.diffusion import _diffusion_linear_temperature
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        class ArgmaxNoRenoiseSampler:
            def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
                return torch.argmax(logits, dim=-1)

            def renoise_canvas(self, accepted_canvas, cur_step):
                return accepted_canvas

        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 4
        torch.manual_seed(123)
        hf_model = DiffusionGemma4ModelForBlockDiffusion(
            DiffusionGemma4Config(**config_dict)
        ).eval()
        mlx_model = Model(ModelConfig.from_dict(config_dict))
        weights = {
            key: mx.array(value.detach().cpu().numpy())
            for key, value in hf_model.state_dict().items()
        }
        mlx_model.load_weights(
            list(mlx_model.sanitize(weights).items()),
            strict=False,
        )

        input_ids_t = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        canvas_t = torch.tensor([[6, 7, 8]], dtype=torch.long)
        input_ids_m = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
        canvas_m = mx.array([[6, 7, 8]], dtype=mx.int32)

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids_t, canvas_ids=canvas_t).logits
        mlx_logits = mlx_model(input_ids=input_ids_m, canvas_ids=canvas_m).logits
        mx.eval(mlx_logits)
        self.assertLess(
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            ),
            1e-5,
        )

        sc_logits = np.linspace(
            -0.5,
            0.5,
            canvas_t.numel() * config_dict["text_config"]["vocab_size"],
            dtype=np.float32,
        ).reshape(1, canvas_t.shape[-1], -1)
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=input_ids_t,
                canvas_ids=canvas_t,
                self_conditioning_logits=torch.tensor(sc_logits),
            ).logits
        mlx_logits = mlx_model(
            input_ids=input_ids_m,
            canvas_ids=canvas_m,
            self_conditioning_logits=mx.array(sc_logits),
        ).logits
        mx.eval(mlx_logits)
        self.assertLess(
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            ),
            1e-5,
        )

        attention_t = torch.ones_like(input_ids_t, dtype=torch.bool)
        decoder_attention_t = torch.nn.functional.pad(
            attention_t,
            (0, canvas_t.shape[-1]),
            value=True,
        )
        with torch.no_grad():
            past_key_values = DynamicCache(
                config=hf_model.config.get_text_config(decoder=True)
            )
            encoder_outputs = hf_model.model.encoder(
                input_ids=input_ids_t,
                attention_mask=attention_t,
                past_key_values=past_key_values,
            )
            past_key_values = encoder_outputs.past_key_values
            mask_mapping = (
                hf_model.model.decoder.create_diffusion_decoder_attention_mask(
                    config=hf_model.config.text_config,
                    inputs_embeds=canvas_t.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=decoder_attention_t,
                )
            )
            logits_processor = LogitsProcessorList(
                [
                    LinearTemperatureScheduleLogitsProcessor(
                        LinearTemperatureScheduleConfig(t_min=0.4, t_max=0.8),
                        4,
                    )
                ]
            )
            hf_current, hf_argmax, hf_processed, _ = hf_model._denoising_step(
                decoder_forward=hf_model.forward,
                current_canvas=canvas_t,
                argmax_canvas=canvas_t,
                input_ids=input_ids_t,
                self_conditioning_logits=None,
                mask_mapping=mask_mapping,
                past_key_values=past_key_values,
                finished_denoising=torch.zeros(1, dtype=torch.bool),
                cur_step=3,
                sampler=ArgmaxNoRenoiseSampler(),
                logits_processor=logits_processor,
                diffusion_stopping_criteria=None,
            )

        attention_m = mx.ones(input_ids_m.shape, dtype=mx.bool_)
        kv_cache = mlx_model.make_cache()
        _, kv_cache = mlx_model.model.encoder(
            input_ids_m,
            attention_mask=attention_m,
            cache=kv_cache,
        )
        decoder_attention_m = mx.concatenate(
            [attention_m, mx.ones(canvas_m.shape, dtype=mx.bool_)],
            axis=-1,
        )
        mask_mapping = mlx_model.model.decoder._make_decoder_masks(
            canvas_m[..., None],
            kv_cache,
            decoder_attention_m,
        )
        mlx_processed = mlx_model(
            cache=kv_cache,
            canvas_ids=canvas_m,
            decoder_attention_mask=mask_mapping,
        ).logits / _diffusion_linear_temperature(3, 4, {"t_min": 0.4, "t_max": 0.8})
        mlx_argmax = mx.argmax(mlx_processed, axis=-1).astype(mx.int32)
        mx.eval(mlx_processed, mlx_argmax)

        self.assertLess(
            float(
                np.max(
                    np.abs(
                        hf_processed.detach().cpu().numpy() - np.array(mlx_processed)
                    )
                )
            ),
            1e-5,
        )
        self.assertEqual(hf_argmax.detach().cpu().numpy().tolist(), mlx_argmax.tolist())
        self.assertEqual(
            hf_current.detach().cpu().numpy().tolist(), mlx_argmax.tolist()
        )

    def test_sanitize_maps_fused_experts_and_keeps_encoder_scalars(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        gate_up = mx.zeros((4, 16, 16))
        weights = {
            "model.decoder.layers.0.experts.gate_up_proj": gate_up,
            "model.decoder.layers.0.experts.down_proj": mx.zeros((4, 16, 8)),
            "model.encoder.language_model.layers.0.layer_scalar": mx.ones((1,)),
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (16, 16)
            ),
            "model.encoder.embed_vision.embedding_projection.weight": mx.zeros(
                (16, 16)
            ),
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight": mx.ones(
                (16,)
            ),
            "lm_head.weight": mx.zeros((64, 16)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn(
            "model.decoder.layers.0.experts.gate_up_proj.weight",
            sanitized,
        )
        self.assertIn(
            "model.decoder.layers.0.experts.down_proj.weight",
            sanitized,
        )
        self.assertEqual(
            sanitized["model.decoder.layers.0.experts.gate_up_proj.weight"].shape,
            (4, 16, 16),
        )
        self.assertIn("model.encoder.language_model.layers.0.layer_scalar", sanitized)
        self.assertNotIn(
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight",
            sanitized,
        )
        self.assertNotIn(
            "model.encoder.embed_vision.embedding_projection.weight",
            sanitized,
        )
        self.assertNotIn(
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight",
            sanitized,
        )
        self.assertNotIn("lm_head.weight", sanitized)

    def test_stream_generate_uses_diffusion_loop(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=1,
            )
        )

        self.assertGreaterEqual(len(responses), 1)
        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].prompt_tokens, 2)
        self.assertEqual(responses[-1].diffusion_canvas_tokens, 3)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 1)
        self.assertEqual(responses[-1].diffusion_work_tokens, 3)

    def test_stream_generate_uses_model_owned_generator(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        original_generate = model.language_model.generate
        calls = {"count": 0}

        def counted_generate(*args, **kwargs):
            calls["count"] += 1
            return original_generate(*args, **kwargs)

        model.language_model.generate = counted_generate
        responses = list(
            stream_generate(
                model,
                FakeProcessor(),
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=1,
            )
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(responses[-1].generation_tokens, 2)

    def test_generate_verbose_omits_diffusion_work_stats(self):
        from mlx_vlm.generate import generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            generate(
                model,
                FakeProcessor(),
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=1,
                verbose=True,
            )

        output = stdout.getvalue()
        self.assertIn("Prompt:", output)
        self.assertIn("Generation:", output)
        self.assertIn("Peak memory:", output)
        self.assertNotIn("Diffusion:", output)
        self.assertNotIn("work tokens", output)
        self.assertNotIn("work-tokens-per-sec", output)

    def test_stream_generate_chunks_diffusion_prefill(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        recorder = RecordingEncoder(model.model.encoder)
        model.model.encoder = recorder
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32),
                max_tokens=1,
                max_denoising_steps=1,
                prefill_step_size=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 1)
        self.assertEqual(recorder.input_lengths, [2, 2, 1])
        self.assertEqual(recorder.attention_masks, [None, None, None])

    def test_stream_generate_honors_diffusion_chunked_prefill_policy(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        recorder = RecordingEncoder(model.model.encoder)
        model.model.encoder = recorder
        processor = FakeProcessor()

        with patch.object(
            model, "chunked_prefill_policy", return_value=False
        ) as policy:
            responses = list(
                stream_generate(
                    model,
                    processor,
                    "",
                    input_ids=mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32),
                    max_tokens=1,
                    max_denoising_steps=1,
                    prefill_step_size=2,
                )
            )

        self.assertEqual(responses[-1].generation_tokens, 1)
        self.assertEqual(recorder.input_lengths, [5])
        policy.assert_called_once()
        self.assertFalse(policy.call_args.kwargs["prefill_kwargs"]["has_padding"])

    def test_chunked_diffusion_prefill_matches_unchunked_tokens(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        def generated_tokens(prefill_step_size):
            mx.random.seed(42)
            model = Model(ModelConfig.from_dict(tiny_config_dict()))
            responses = list(
                stream_generate(
                    model,
                    FakeProcessor(),
                    "",
                    input_ids=mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32),
                    max_tokens=3,
                    max_denoising_steps=1,
                    prefill_step_size=prefill_step_size,
                )
            )
            return [r.token for r in responses if r.token is not None]

        self.assertEqual(generated_tokens(None), generated_tokens(2))

    def test_full_precision_generation_uses_model_self_conditioning(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        self.assertFalse(model.prefers_logits_self_conditioning)

        with (
            patch.object(
                model,
                "diffusion_prepare_self_conditioning",
                wraps=model.diffusion_prepare_self_conditioning,
            ) as prepare,
            patch.object(
                model,
                "diffusion_self_conditioning",
                wraps=model.diffusion_self_conditioning,
            ) as self_conditioning,
        ):
            responses = list(
                stream_generate(
                    model,
                    processor,
                    "",
                    input_ids=mx.array([[2, 3]], dtype=mx.int32),
                    max_tokens=2,
                    max_denoising_steps=2,
                )
            )

        self.assertEqual(responses[-1].generation_tokens, 2)
        prepare.assert_called_once()
        self_conditioning.assert_called()

    def test_stream_generate_keeps_padded_diffusion_prefill_unchunked(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        recorder = RecordingEncoder(model.model.encoder)
        model.model.encoder = recorder
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3, 4, 5, 0]], dtype=mx.int32),
                mask=mx.array([[1, 1, 1, 1, 0]], dtype=mx.bool_),
                max_tokens=1,
                max_denoising_steps=1,
                prefill_step_size=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 1)
        self.assertEqual(recorder.input_lengths, [5])
        self.assertIsNotNone(recorder.attention_masks[0])

    def test_stream_generate_keeps_visual_diffusion_prefill_unchunked(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        recorder = RecordingEncoder(model.model.encoder)
        model.model.encoder = recorder
        processor = FakeProcessor()
        mm_token_type_ids = mx.array([[0, 1, 1, 0, 0]], dtype=mx.int32)

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32),
                mm_token_type_ids=mm_token_type_ids,
                max_tokens=1,
                max_denoising_steps=1,
                prefill_step_size=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 1)
        self.assertEqual(recorder.input_lengths, [5])
        self.assertIs(recorder.mm_token_type_ids[0], mm_token_type_ids)

    def test_decoder_masks_skip_no_padding_short_context(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        cache = model.make_cache()
        _, cache = model.model.encoder(
            mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache
        )

        masks = model.model.decoder._make_decoder_masks(
            mx.zeros((1, 3, 1)),
            cache,
            decoder_attention_mask=None,
        )

        self.assertIsNone(masks["full_attention"])
        self.assertIsNone(masks["sliding_attention"])

    def test_static_prefix_cache_exposes_full_decoder_state(self):
        from mlx_vlm.models.cache import StaticPrefixKVCache

        cache = StaticPrefixKVCache(max_size=5)
        keys = mx.ones((1, 2, 3, 4))
        values = mx.ones((1, 2, 3, 4)) * 2

        prefix_keys, prefix_values = cache.update_and_fetch(keys, values)

        self.assertEqual(prefix_keys.shape, (1, 2, 3, 4))
        self.assertEqual(prefix_values.shape, (1, 2, 3, 4))
        self.assertEqual(cache.decoder_state[0].shape, (1, 2, 5, 4))
        self.assertEqual(cache.offset, 3)

    def test_stream_generate_supports_static_diffusion_cache(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=4,
                max_denoising_steps=1,
                diffusion_static_cache=True,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 4)
        self.assertGreaterEqual(responses[-1].diffusion_canvas_tokens, 3)

    def test_default_confidence_threshold_sampler_can_exit_after_one_step(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=4,
                diffusion_threshold=0.0,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 1)
        self.assertEqual(responses[-1].diffusion_work_tokens, 3)

    def test_entropy_bound_sampler_runs_configured_steps(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 4
        config_dict["generation_config"]["sampler_config"] = {
            "_cls_name": "EntropyBoundSamplerConfig",
            "entropy_bound": 1_000.0,
        }
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                diffusion_sampler="entropy-bound",
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 4)
        self.assertEqual(responses[-1].diffusion_work_tokens, 12)

    def test_stream_generate_uses_checkpoint_denoising_steps(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 48
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                diffusion_sampler="entropy-bound",
            )
        )

        self.assertEqual(responses[-1].diffusion_denoising_steps, 48)
        self.assertEqual(responses[-1].diffusion_work_tokens, 144)

    def test_stream_generate_respects_explicit_denoising_steps_override(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 48
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=48,
                diffusion_sampler="entropy-bound",
            )
        )

        self.assertEqual(responses[-1].diffusion_denoising_steps, 48)
        self.assertEqual(responses[-1].diffusion_work_tokens, 144)

    def test_stream_generate_respects_diffusion_max_canvas_length(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=4,
                max_denoising_steps=1,
                diffusion_max_canvas_length=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 4)
        self.assertEqual(responses[-1].diffusion_canvas_tokens, 4)
        self.assertEqual(responses[-1].diffusion_work_tokens, 4)

    def test_stream_generate_can_emit_unmasking_drafts(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=2,
                diffusion_show_unmasking=True,
            )
        )
        drafts = [response for response in responses if response.is_draft]
        finals = [response for response in responses if not response.is_draft]

        self.assertEqual(len(drafts), 3)
        self.assertTrue(all(response.draft_text for response in drafts))
        self.assertEqual(drafts[0].diffusion_step, 0)
        self.assertIn("[Mask]", drafts[0].draft_text)
        self.assertEqual(drafts[1].diffusion_step, 1)
        self.assertEqual(drafts[-1].diffusion_total_steps, 2)
        self.assertEqual(finals[-1].generation_tokens, 2)

    def test_diffusion_initial_canvas_pads_short_decoder_input_ids(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")

        decoder_input_ids = diffusion_module._normalize_decoder_input_ids(
            [[10, 11]],
            batch_size=1,
            dtype=mx.int32,
        )
        with patch.object(
            diffusion_module,
            "_diffusion_initialize_canvas",
            return_value=mx.array([[7, 8, 9]], dtype=mx.int32),
        ):
            canvas = diffusion_module._diffusion_initial_canvas(
                decoder_input_ids,
                start_index=0,
                batch_size=1,
                canvas_length=3,
                vocab_size=64,
                dtype=mx.int32,
            )
            mx.eval(canvas)

        self.assertEqual(canvas.tolist(), [[10, 11, 9]])

    def test_diffusion_decoder_input_ids_validation(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")

        with self.assertRaisesRegex(ValueError, "2D array"):
            diffusion_module._normalize_decoder_input_ids(
                [10, 11],
                batch_size=1,
                dtype=mx.int32,
            )

        with self.assertRaisesRegex(ValueError, "batch size"):
            diffusion_module._normalize_decoder_input_ids(
                [[10, 11], [12, 13]],
                batch_size=1,
                dtype=mx.int32,
            )

    def test_stream_generate_uses_decoder_input_ids_as_initial_canvas(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        seen_canvases = []

        def decoder_logits(current_canvas, *args, **kwargs):
            del args, kwargs
            mx.eval(current_canvas)
            seen_canvases.append(current_canvas.tolist())
            return mx.zeros(
                (
                    current_canvas.shape[0],
                    current_canvas.shape[1],
                    config.text_config.vocab_size,
                )
            )

        with (
            patch.object(model, "diffusion_decoder_logits", side_effect=decoder_logits),
            patch.object(
                diffusion_module,
                "_diffusion_initialize_canvas",
                return_value=mx.array([[7, 8, 9]], dtype=mx.int32),
            ),
        ):
            responses = list(
                stream_generate(
                    model,
                    FakeProcessor(),
                    "",
                    input_ids=mx.array([[2, 3]], dtype=mx.int32),
                    max_tokens=1,
                    max_denoising_steps=1,
                    decoder_input_ids=mx.array([[10, 11]], dtype=mx.int32),
                )
            )

        self.assertEqual(responses[-1].generation_tokens, 1)
        self.assertEqual(seen_canvases[0], [[10, 11, 9]])

    def test_stream_generate_slices_decoder_input_ids_by_canvas(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        seen_canvases = []

        def decoder_logits(current_canvas, *args, **kwargs):
            del args, kwargs
            mx.eval(current_canvas)
            seen_canvases.append(current_canvas.tolist())
            return mx.zeros(
                (
                    current_canvas.shape[0],
                    current_canvas.shape[1],
                    config.text_config.vocab_size,
                )
            )

        with patch.object(model, "diffusion_decoder_logits", side_effect=decoder_logits):
            responses = list(
                stream_generate(
                    model,
                    FakeProcessor(),
                    "",
                    input_ids=mx.array([[2, 3]], dtype=mx.int32),
                    max_tokens=4,
                    max_denoising_steps=1,
                    diffusion_max_canvas_length=2,
                    decoder_input_ids=mx.array([[10, 11, 12, 13]], dtype=mx.int32),
                )
            )

        self.assertEqual(responses[-1].generation_tokens, 4)
        self.assertEqual(seen_canvases[:2], [[[10, 11]], [[12, 13]]])

    def test_diffusion_zero_temperature_uses_argmax_canvas(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")
        from mlx_vlm.generate.diffusion import _diffusion_sample_canvas

        logits = mx.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]])

        with patch.object(diffusion_module.mx.random, "categorical") as categorical:
            sampled = _diffusion_sample_canvas(logits, mx.int32, temperature=0.0)
            mx.eval(sampled)

        categorical.assert_not_called()
        self.assertEqual(sampled.tolist(), [[1, 0]])

    def test_diffusion_positive_temperature_samples_canvas(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")
        from mlx_vlm.generate.diffusion import _diffusion_sample_canvas

        logits = mx.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]])

        with patch.object(diffusion_module.mx.random, "categorical") as categorical:
            categorical.return_value = mx.array([[2, 1]])
            sampled = _diffusion_sample_canvas(logits, mx.int32, temperature=0.7)
            mx.eval(sampled)

        categorical.assert_called_once()
        self.assertEqual(sampled.tolist(), [[2, 1]])

    def test_diffusion_confidence_transfer_forces_best_unrevealed_token(self):
        from mlx_vlm.generate.diffusion import _diffusion_confidence_transfer_mask

        confidence = mx.array([[0.1, 0.4, 0.2]])
        unrevealed = mx.array([[True, True, False]])

        transfer = _diffusion_confidence_transfer_mask(
            confidence,
            unrevealed,
            threshold=0.9,
        )
        mx.eval(transfer)

        self.assertEqual(transfer.tolist(), [[False, True, False]])

    def test_diffusion_entropy_transfer_forces_best_unrevealed_token(self):
        from mlx_vlm.generate.diffusion import _diffusion_entropy_transfer_mask

        entropy = mx.array([[0.4, 0.1, 0.2, 0.05]])

        transfer = _diffusion_entropy_transfer_mask(
            entropy,
            entropy_bound=0.16,
        )
        mx.eval(transfer)

        self.assertEqual(transfer.tolist(), [[False, True, True, True]])

    def test_unmasking_display_has_no_prefix_and_preserves_newlines(self):
        from mlx_vlm.generate import GenerationResult
        from mlx_vlm.generate.diffusion import (
            _format_diffusion_draft_line,
            _format_diffusion_live_text,
        )

        draft = GenerationResult(
            is_draft=True,
            draft_text="[Mask]\nHello",
            diffusion_canvas_index=1,
            diffusion_step=1,
            diffusion_total_steps=4,
        )

        self.assertEqual(_format_diffusion_draft_line(draft, 80), "[Mask]\nHello")
        self.assertEqual(
            _format_diffusion_live_text("hello\nworld", 80),
            "hello\nworld",
        )
        self.assertEqual(
            _format_diffusion_live_text(
                "hello\nworld",
                80,
                preserve_newlines=False,
            ),
            "hello\\nworld",
        )

    def test_diffusion_masked_draft_decode_preserves_newlines(self):
        from mlx_vlm.generate.diffusion import _decode_diffusion_masked_draft

        class NewlineTokenizer:
            def decode(self, tokens, skip_special_tokens=False):
                return "hello\nworld"

        self.assertEqual(
            _decode_diffusion_masked_draft(
                NewlineTokenizer(),
                [1],
                [True],
                skip_special_token_ids=[],
            ),
            "hello\nworld",
        )

    def test_unmasking_display_is_untrimmed_by_default(self):
        from mlx_vlm.generate import GenerationResult
        from mlx_vlm.generate.diffusion import (
            _format_diffusion_draft_line,
            _format_diffusion_live_text,
        )

        long_text = "A" * 200
        draft = GenerationResult(is_draft=True, draft_text=long_text)

        self.assertEqual(_format_diffusion_draft_line(draft), long_text)
        self.assertEqual(_format_diffusion_live_text(long_text), long_text)
        self.assertTrue(_format_diffusion_live_text(long_text, 20).endswith("..."))

    def test_generate_redraw_mode_prints_full_final_text(self):
        diffusion_module = importlib.import_module("mlx_vlm.generate.diffusion")
        dispatch_module = importlib.import_module("mlx_vlm.generate.dispatch")
        from mlx_vlm.generate import GenerationResult, generate

        class Config:
            model_type = "diffusion_gemma"
            eos_token_id = 999999

        class Model:
            config = Config()

        long_text = "The sky is blue because Rayleigh scattering favors blue light."
        chunks = [
            GenerationResult(is_draft=True, draft_text="[Mask] [Mask] [Mask]"),
            GenerationResult(
                text=long_text,
                token=1,
                prompt_tokens=3,
                generation_tokens=12,
                total_tokens=15,
                prompt_tps=10.0,
                generation_tps=5.0,
            ),
        ]

        buffer = io.StringIO()
        with (
            patch.object(dispatch_module, "stream_generate", return_value=iter(chunks)),
            patch.object(
                diffusion_module, "_supports_in_place_output", return_value=True
            ),
            contextlib.redirect_stdout(buffer),
        ):
            result = generate(
                Model(),
                FakeProcessor(),
                "",
                verbose=True,
                diffusion_show_unmasking=True,
                diffusion_unmasking_width=20,
            )

        self.assertEqual(result.text, long_text)
        self.assertIn(long_text, buffer.getvalue())

    def test_auto_processor_uses_local_text_only_processor(self):
        from transformers import AutoProcessor

        from mlx_vlm.models.diffusion_gemma import DiffusionGemma4Processor

        sentinel = object()
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(
                json.dumps({"model_type": "diffusion_gemma"})
            )
            with patch.object(
                DiffusionGemma4Processor,
                "from_pretrained",
                return_value=sentinel,
            ) as from_pretrained:
                processor = AutoProcessor.from_pretrained(tmpdir)

        self.assertIs(processor, sentinel)
        from_pretrained.assert_called_once()

    def test_processor_demotes_tool_parser_tokens_from_specials(self):
        from mlx_vlm.models.diffusion_gemma import DiffusionGemma4Processor
        from mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma import (
            _TOOL_PARSER_TOKENS,
        )
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

        tokenizer = TinyDiffusionGemma4Tokenizer()
        tokenizer.special_tokens.update(
            {token: 70 + i for i, token in enumerate(_TOOL_PARSER_TOKENS)}
        )
        tokenizer.special_tokens["<extra_special>"] = 90
        tokenizer.stc_token = "<|tool_call>"
        tokenizer.etc_token = "<tool_call|>"
        tokenizer.escape_token = '<|"|>'
        tokenizer.soc_token = "<|channel>"
        tokenizer.eoc_token = "<channel|>"
        tokenizer.additional_special_tokens = ["<extra_special>"]

        with patch.object(
            Gemma4Processor,
            "from_pretrained",
            return_value=SimpleNamespace(tokenizer=tokenizer),
        ):
            processor = DiffusionGemma4Processor.from_pretrained("demo")

        self.assertIs(processor.tokenizer, tokenizer)
        self.assertEqual(tokenizer.additional_special_tokens, ["<extra_special>"])
        self.assertEqual(tokenizer.all_special_ids, [90])
        self.assertIsNone(tokenizer.stc_token)
        self.assertIsNone(tokenizer.etc_token)
        self.assertIsNone(tokenizer.escape_token)
        self.assertIsNone(tokenizer.soc_token)
        self.assertIsNone(tokenizer.eoc_token)
        for token in _TOOL_PARSER_TOKENS:
            self.assertNotEqual(
                tokenizer.convert_tokens_to_ids(token), tokenizer.unk_token_id
            )

    def test_processor_returns_video_frames_as_pixel_values(self):
        processor = tiny_diffusion_gemma_processor()

        result = processor(
            text="<video> describe",
            videos=[np.zeros((2, 3, 4, 4), dtype=np.uint8)],
        )

        self.assertIn("pixel_values", result)
        self.assertNotIn("pixel_values_videos", result)
        self.assertIsInstance(result["pixel_values"], mx.array)
        self.assertEqual(result["pixel_values"].shape, (2, 3, 4, 4))
        self.assertEqual(result["num_frames_per_video"], [2])
        self.assertEqual(
            int(mx.sum(result["mm_token_type_ids"] == 2).item()),
            2,
        )

    def test_processor_video_outputs_can_cross_thread_boundary(self):
        raw_queue = Queue()
        result_queue = Queue()

        def run_thread(fn):
            errors = Queue()

            def wrapped():
                try:
                    fn()
                except BaseException as exc:
                    errors.put(exc)

            thread = Thread(target=wrapped)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            if not errors.empty():
                raise errors.get()

        def producer():
            processor = tiny_diffusion_gemma_processor()
            raw_queue.put(
                processor(
                    text="<video> describe",
                    videos=[np.zeros((2, 3, 4, 4), dtype=np.uint8)],
                )
            )

        def consumer():
            result = raw_queue.get(timeout=5)
            mx.eval(
                result["input_ids"],
                result["attention_mask"],
                result["mm_token_type_ids"],
                result["pixel_values"],
            )
            result_queue.put(
                (
                    result["pixel_values"].shape,
                    int(mx.sum(result["mm_token_type_ids"] == 2).item()),
                )
            )

        run_thread(producer)
        run_thread(consumer)

        self.assertEqual(result_queue.get(timeout=5), ((2, 3, 4, 4), 2))

    def test_apply_chat_template_includes_video_token_for_video_inputs(self):
        from mlx_vlm.prompt_utils import apply_chat_template

        processor = tiny_diffusion_gemma_processor()
        rendered = apply_chat_template(
            processor,
            SimpleNamespace(model_type="diffusion_gemma"),
            "Describe this video.",
            video=["clip.mp4"],
        )

        self.assertIn(processor.video_token, rendered)

        result = processor(
            text=rendered,
            videos=[np.zeros((2, 3, 4, 4), dtype=np.uint8)],
        )
        self.assertEqual(
            int(mx.sum(result["mm_token_type_ids"] == 2).item()),
            2,
        )

    def test_processor_orders_mixed_images_and_videos_in_pixel_values(self):
        processor = tiny_diffusion_gemma_processor(image_processor=TinyImageProcessor())

        image = np.ones((3, 4, 4), dtype=np.float32)
        result = processor(
            text="<video> then <image>",
            images=[image],
            videos=[np.zeros((2, 3, 4, 4), dtype=np.uint8)],
        )

        self.assertNotIn("pixel_values_videos", result)
        self.assertEqual(result["pixel_values"].shape, (3, 3, 4, 4))
        self.assertTrue(bool(mx.all(result["pixel_values"][:2] == 0).item()))
        self.assertTrue(bool(mx.all(result["pixel_values"][2] == 1).item()))


class TestDiffusionVisualization(unittest.TestCase):
    def test_wrap_text_wraps_on_spaces(self):
        from mlx_vlm.models.diffusion_gemma.visualizer import _wrap_text

        wrapped = _wrap_text("alpha beta gamma delta", 11)
        self.assertEqual(wrapped, "alpha beta\ngamma delta")
        # Newlines in the input are preserved.
        wrapped = _wrap_text("alpha\nbeta gamma", 20)
        self.assertEqual(wrapped, "alpha\nbeta gamma")
        # A single overlong word is hard-split.
        self.assertEqual(_wrap_text("abcdef", 3), "abc\ndef")

    def test_wrap_text_preserves_code_block_indentation(self):
        from mlx_vlm.models.diffusion_gemma.visualizer import _wrap_text

        code = (
            "import random\n\n"
            "def calculate_pi_monte_carlo(iterations):\n"
            "    inside_circle = 0\n"
            "    \n"
            "    for _ in range(iterations):"
        )

        self.assertEqual(_wrap_text(code, 80), code)

    def test_redrawer_overwrites_frames_in_place(self):
        import contextlib
        import io
        from unittest.mock import patch

        from mlx_vlm.models.diffusion_gemma.visualizer import _CanvasRedrawer

        class Terminal:
            columns = 40
            lines = 24

        redrawer = _CanvasRedrawer(min_interval=0.0)
        buffer = io.StringIO()
        with patch("shutil.get_terminal_size", return_value=Terminal()):
            with contextlib.redirect_stdout(buffer):
                redrawer.draw("one two three\nfour")
                first = buffer.getvalue()
                redrawer.draw("one two three\nfour five")
                second = buffer.getvalue()[len(first) :]
                redrawer.finish()

        # Frames overwrite line by line; no full-screen clears in normal mode.
        self.assertNotIn("\033[2J", first + second)
        self.assertIn("\033[2K", first)
        # The second frame moves the cursor up over the previous 2-row frame.
        self.assertIn("\033[1A", second)
        self.assertEqual(redrawer.rows, 0)

    def test_redrawer_escalates_to_alternate_screen(self):
        import contextlib
        import io
        from unittest.mock import patch

        from mlx_vlm.models.diffusion_gemma.visualizer import _CanvasRedrawer

        class Terminal:
            columns = 40
            lines = 6

        redrawer = _CanvasRedrawer(min_interval=0.0)
        buffer = io.StringIO()
        with patch("shutil.get_terminal_size", return_value=Terminal()):
            with contextlib.redirect_stdout(buffer):
                redrawer.draw("word " * 60)
                redrawer.finish()
        out = buffer.getvalue()

        self.assertIn("\033[?1049h", out)
        self.assertIn("\033[?25l", out)
        self.assertIn("\033[?1049l", out)
        self.assertIn("\033[?25h", out)
        self.assertFalse(redrawer.alternate_screen)

    def test_make_visualizer_defaults_for_verbose_terminals(self):
        from unittest.mock import patch

        from mlx_vlm.models.diffusion_gemma.visualizer import make_unmasking_visualizer

        with patch("sys.stdout.isatty", return_value=True):
            kwargs = {}
            visualizer = make_unmasking_visualizer(kwargs, verbose=True)
            self.assertTrue(kwargs.get("diffusion_show_unmasking"))
            self.assertIsNotNone(visualizer)

            # Explicit opt-out is respected.
            kwargs = {"diffusion_show_unmasking": False}
            self.assertIsNone(make_unmasking_visualizer(kwargs, verbose=True))
            self.assertFalse(kwargs["diffusion_show_unmasking"])

            # Quiet runs stay off.
            kwargs = {}
            self.assertIsNone(make_unmasking_visualizer(kwargs, verbose=False))
            self.assertNotIn("diffusion_show_unmasking", kwargs)

        # Piped output stays off.
        with patch("sys.stdout.isatty", return_value=False):
            kwargs = {}
            self.assertIsNone(make_unmasking_visualizer(kwargs, verbose=True))
            self.assertNotIn("diffusion_show_unmasking", kwargs)

    def test_visualizer_composes_full_canvas(self):
        from mlx_vlm.models.diffusion_gemma.visualizer import DiffusionGemma4Visualizer

        visualizer = DiffusionGemma4Visualizer()
        drawn = []

        class FakeRedrawer:
            def draw(self, text, wrap_width=None):
                drawn.append(text)

            def finish(self):
                drawn.append("<finish>")

        visualizer.redrawer = FakeRedrawer()

        class FakeDraft:
            draft_text = "[Mask]\nworld"

        visualizer.handle_text("Hello.\n")
        visualizer.handle_draft(FakeDraft())
        self.assertEqual(drawn[-1], "Hello.\n[Mask]\nworld")

        visualizer.handle_text(" Bye.")
        self.assertEqual(drawn[-1], "Hello.\n Bye.")

        visualizer.finish("Hello.\n Bye.")
        self.assertIn("<finish>", drawn)

    def test_output_handler_delegates_to_model_visualizer(self):
        from unittest.mock import patch

        from mlx_vlm.generate.diffusion import DiffusionOutputHandler
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        # Importing the model package installs the handler patch.
        self.assertTrue(getattr(DiffusionOutputHandler, "_model_visualizer_patched"))

        model = Model(ModelConfig.from_dict(tiny_config_dict()))

        with patch("sys.stdout.isatty", return_value=True):
            kwargs = {}
            handler = DiffusionOutputHandler(model, kwargs, verbose=True)

        self.assertIsNotNone(handler._model_visualizer)
        self.assertTrue(kwargs.get("diffusion_show_unmasking"))
        self.assertIsNone(handler.redrawer)

        events = []

        class FakeVisualizer:
            def handle_draft(self, response):
                events.append(("draft", response.draft_text))

            def handle_text(self, text):
                events.append(("text", text))
                return True

            def finish(self, text):
                events.append(("finish", text))

        handler._model_visualizer = FakeVisualizer()

        class FakeDraft:
            draft_text = "[Mask]"

        handler.handle_draft(FakeDraft())
        self.assertTrue(handler.handle_text("hi"))
        handler.finish("hi")
        self.assertEqual(
            events, [("draft", "[Mask]"), ("text", "hi"), ("finish", "hi")]
        )


class TestDiffusionBlockStreaming(unittest.TestCase):
    def test_stream_generate_emits_block_boundaries(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        # canvas_length=3 with max_tokens=5 -> two canvases, two boundaries.
        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=5,
            )
        )

        boundaries = [r for r in responses if r.diffusion_block_complete]
        self.assertEqual(len(boundaries), 2)
        self.assertTrue(all(r.text == "" for r in boundaries))
        self.assertEqual([r.diffusion_canvas_index for r in boundaries], [1, 2])
        # Token results carry their canvas index; the final result carries
        # the finish reason.
        token_results = [r for r in responses if r.text and not r.finish_reason]
        self.assertTrue(all(r.diffusion_canvas_index > 0 for r in token_results))
        self.assertEqual(responses[-1].finish_reason, "length")

    def test_diffusion_block_chunks_groups_by_block(self):
        from mlx_vlm.generate.common import GenerationResult
        from mlx_vlm.server.generation import _diffusion_block_chunks

        results = [
            GenerationResult(text="Hello", token=5, diffusion_canvas_index=1),
            GenerationResult(text=" world", token=6, diffusion_canvas_index=1),
            GenerationResult(diffusion_block_complete=True, diffusion_canvas_index=1),
            GenerationResult(text="!", token=7, diffusion_canvas_index=2),
            GenerationResult(diffusion_block_complete=True, diffusion_canvas_index=2),
            GenerationResult(text="", finish_reason="stop", token=7),
        ]

        chunks = list(_diffusion_block_chunks(iter(results)))
        self.assertEqual(
            [(c.text, c.finish_reason) for c in chunks],
            [("Hello world", None), ("!", None), ("", "stop")],
        )

    def test_diffusion_block_chunks_skips_drafts_and_empty_blocks(self):
        from mlx_vlm.generate.common import GenerationResult
        from mlx_vlm.server.generation import _diffusion_block_chunks

        results = [
            GenerationResult(is_draft=True, draft_text="[Mask]"),
            GenerationResult(diffusion_block_complete=True, diffusion_canvas_index=1),
            GenerationResult(text="Hi", token=4, diffusion_canvas_index=2),
            GenerationResult(text=".", finish_reason="stop", token=5),
        ]

        chunks = list(_diffusion_block_chunks(iter(results)))
        self.assertEqual(
            [(c.text, c.finish_reason) for c in chunks],
            [("Hi.", "stop")],
        )


class TestDiffusionGemma4Quantized(unittest.TestCase):
    def test_quant_predicate_uses_8bit_for_embeddings_and_attention(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config_dict()))
        predicate = model.quant_predicate
        decoder = model.model.decoder

        self.assertEqual(
            predicate("model.decoder.embed_tokens", decoder.embed_tokens),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.self_attn.q_proj",
                decoder.layers[0].self_attn.q_proj,
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.router.proj",
                decoder.layers[0].router.proj,
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.mlp.gate_proj",
                decoder.layers[0].mlp.gate_proj,
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertIs(
            predicate(
                "model.decoder.layers.0.experts.gate_up_proj",
                decoder.layers[0].experts.gate_up_proj,
            ),
            True,
        )

    def test_diffusion_kwargs_omits_default_confidence_threshold_sampler(self):
        from types import SimpleNamespace

        from mlx_vlm.generate.diffusion import diffusion_kwargs_from_args
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config_dict()))
        args = SimpleNamespace(
            max_denoising_steps=None,
            diffusion_full_canvas=False,
            diffusion_min_canvas_length=None,
            diffusion_max_canvas_length=None,
            diffusion_sampler="confidence-threshold",
            threshold=None,
        )

        self.assertEqual(diffusion_kwargs_from_args(args, model.config), {})

        args.diffusion_sampler = "entropy-bound"
        self.assertEqual(
            diffusion_kwargs_from_args(args, model.config),
            {"diffusion_sampler": "entropy-bound"},
        )

        args.threshold = 0.7
        self.assertEqual(
            diffusion_kwargs_from_args(args, model.config),
            {
                "diffusion_sampler": "entropy-bound",
                "diffusion_threshold": 0.7,
                "threshold": 0.7,
            },
        )

    def test_stream_generate_with_quantized_embeddings(self):
        import mlx.nn as nn

        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["text_config"]["hidden_size"] = 32
        # Several denoising steps so quantized self-conditioning reuses logits
        # and avoids materializing a dequantized embedding table.
        config_dict["generation_config"]["max_denoising_steps"] = 3
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        nn.quantize(
            model,
            group_size=32,
            bits=5,
            class_predicate=lambda path, module: isinstance(module, nn.Embedding),
        )
        self.assertIsInstance(model.model.decoder.embed_tokens, nn.QuantizedEmbedding)
        self.assertTrue(model.prefers_logits_self_conditioning)

        processor = FakeProcessor()
        with (
            patch.object(
                model,
                "diffusion_prepare_self_conditioning",
                wraps=model.diffusion_prepare_self_conditioning,
            ) as prepare,
            patch.object(
                model,
                "diffusion_self_conditioning",
                wraps=model.diffusion_self_conditioning,
            ) as self_conditioning,
        ):
            responses = list(
                stream_generate(
                    model,
                    processor,
                    "",
                    input_ids=mx.array([[2, 3]], dtype=mx.int32),
                    max_tokens=2,
                )
            )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertGreater(responses[-1].diffusion_work_tokens, 0)
        prepare.assert_called_once()
        self_conditioning.assert_called()

    def test_stream_generate_with_mxfp4_quantized_embeddings(self):
        import mlx.nn as nn

        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["text_config"]["hidden_size"] = 32
        config_dict["generation_config"]["max_denoising_steps"] = 3
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            mode="mxfp4",
            class_predicate=lambda path, module: isinstance(module, nn.Embedding),
        )
        self.assertEqual(model.model.decoder.embed_tokens.mode, "mxfp4")

        processor = FakeProcessor()
        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertGreater(responses[-1].diffusion_work_tokens, 0)

    def test_embed_canvas_quantized_self_conditioning_logits(self):
        import mlx.nn as nn

        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["text_config"]["hidden_size"] = 32
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        nn.quantize(
            model,
            group_size=32,
            bits=5,
            class_predicate=lambda path, module: isinstance(module, nn.Embedding),
        )

        decoder = model.model.decoder
        canvas_ids = mx.array([[5, 6, 7]])
        logits = mx.random.normal((1, 3, config.text_config.vocab_size))
        embeds = decoder._embed_canvas(canvas_ids, self_conditioning_logits=logits)
        self.assertEqual(embeds.shape, (1, 3, config.text_config.hidden_size))

    def test_embed_canvas_mxfp4_quantized_self_conditioning_logits(self):
        import mlx.nn as nn

        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["text_config"]["hidden_size"] = 32
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            mode="mxfp4",
            class_predicate=lambda path, module: isinstance(module, nn.Embedding),
        )

        decoder = model.model.decoder
        self.assertEqual(decoder.embed_tokens.mode, "mxfp4")
        canvas_ids = mx.array([[5, 6, 7]])
        logits = mx.random.normal((1, 3, config.text_config.vocab_size))
        embeds = decoder._embed_canvas(canvas_ids, self_conditioning_logits=logits)
        self.assertEqual(embeds.shape, (1, 3, config.text_config.hidden_size))


def tiny_vision_config_dict():
    config = tiny_config_dict()
    config["image_token_id"] = 60
    config["video_token_id"] = 61
    config["text_config"]["use_bidirectional_attention"] = "vision"
    config["vision_config"] = {
        "model_type": "gemma4_vision",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "patch_size": 2,
        "pooling_kernel_size": 2,
        "default_output_length": 1,
        "position_embedding_size": 8,
    }
    return config


class TestDiffusionGemma4Vision(unittest.TestCase):
    def test_image_features_scattered_into_embeddings(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_vision_config_dict())
        model = Model(config)

        image_token_id = config.image_token_id
        input_ids = mx.array([[2, image_token_id, 3]])
        pixel_values = mx.random.uniform(shape=(1, 3, 4, 4))

        text_only = model.get_input_embeddings(input_ids=input_ids).inputs_embeds
        with_image = model.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values
        ).inputs_embeds

        self.assertEqual(with_image.shape, text_only.shape)
        # Text positions are untouched; image positions carry vision features.
        self.assertTrue(bool(mx.allclose(with_image[0, 0], text_only[0, 0]).item()))
        self.assertTrue(bool(mx.allclose(with_image[0, 2], text_only[0, 2]).item()))
        self.assertFalse(bool(mx.allclose(with_image[0, 1], text_only[0, 1]).item()))

        expected = model.model.encoder.get_image_features(pixel_values).astype(
            with_image.dtype
        )
        self.assertTrue(
            bool(mx.allclose(with_image[0, 1], expected[0, 0], atol=1e-5).item())
        )

    def test_vision_block_bidirectional_encoder_mask(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_vision_config_dict())
        model = Model(config)
        encoder = model.model.encoder

        # text, image, image, text
        mm_token_type_ids = mx.array([[0, 1, 1, 0]])
        h = mx.zeros((1, 4, config.text_config.hidden_size))
        cache = encoder.make_cache()
        masks = encoder._make_encoder_masks(
            h, cache, None, mm_token_type_ids=mm_token_type_ids
        )

        for mask in masks:
            self.assertEqual(mask.shape, (1, 1, 4, 4))
            # Image tokens attend bidirectionally within the block.
            self.assertTrue(bool(mask[0, 0, 1, 2].item()))
            # Text tokens stay causal.
            self.assertFalse(bool(mask[0, 0, 0, 1].item()))
            self.assertFalse(bool(mask[0, 0, 0, 3].item()))
            # Later text token sees the whole prefix causally.
            self.assertTrue(bool(mask[0, 0, 3, 0].item()))

        # Without vision tokens the fast path is preserved.
        text_masks = encoder._make_encoder_masks(
            h, cache, None, mm_token_type_ids=mx.zeros((1, 4), dtype=mx.int32)
        )
        for mask in text_masks:
            self.assertFalse(isinstance(mask, mx.array) and mask.shape == (1, 1, 4, 4))

    def test_stream_generate_with_image_inputs(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_vision_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        image_token_id = config.image_token_id
        input_ids = mx.array([[2, image_token_id, 3]])
        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=input_ids,
                pixel_values=mx.random.uniform(shape=(1, 3, 4, 4)),
                mm_token_type_ids=mx.array([[0, 1, 0]]),
                max_tokens=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertGreater(responses[-1].diffusion_work_tokens, 0)

    def test_video_features_scattered_into_embeddings(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_vision_config_dict())
        model = Model(config)

        input_ids = mx.array([[2, config.video_token_id, 3]])
        pixel_values = mx.random.uniform(shape=(1, 3, 4, 4))

        text_only = model.get_input_embeddings(input_ids=input_ids).inputs_embeds
        with_video = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
        ).inputs_embeds

        self.assertEqual(with_video.shape, text_only.shape)
        self.assertTrue(bool(mx.allclose(with_video[0, 0], text_only[0, 0]).item()))
        self.assertTrue(bool(mx.allclose(with_video[0, 2], text_only[0, 2]).item()))
        self.assertFalse(bool(mx.allclose(with_video[0, 1], text_only[0, 1]).item()))

        expected = model.model.encoder.get_image_features(pixel_values).astype(
            with_video.dtype
        )
        self.assertTrue(
            bool(mx.allclose(with_video[0, 1], expected[0, 0], atol=1e-5).item())
        )

    def test_video_features_scattered_from_token_types_without_video_token_id(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_vision_config_dict()
        config_dict["video_token_id"] = None
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        input_ids = mx.array([[2, 61, 3]])
        pixel_values = mx.random.uniform(shape=(1, 3, 4, 4))
        mm_token_type_ids = mx.array([[0, 2, 0]])

        text_only = model.get_input_embeddings(input_ids=input_ids).inputs_embeds
        with_video = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
        ).inputs_embeds

        self.assertEqual(with_video.shape, text_only.shape)
        self.assertTrue(bool(mx.allclose(with_video[0, 0], text_only[0, 0]).item()))
        self.assertTrue(bool(mx.allclose(with_video[0, 2], text_only[0, 2]).item()))
        self.assertFalse(bool(mx.allclose(with_video[0, 1], text_only[0, 1]).item()))

        expected = model.model.encoder.get_image_features(pixel_values).astype(
            with_video.dtype
        )
        self.assertTrue(
            bool(mx.allclose(with_video[0, 1], expected[0, 0], atol=1e-5).item())
        )

    def test_stream_generate_with_video_inputs(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_vision_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        input_ids = mx.array([[2, config.video_token_id, 3]])
        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=input_ids,
                pixel_values=mx.random.uniform(shape=(1, 3, 4, 4)),
                mm_token_type_ids=mx.array([[0, 2, 0]]),
                max_tokens=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertGreater(responses[-1].diffusion_work_tokens, 0)

    def test_sanitize_handles_vision_weights(self):
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        vision_model = Model(ModelConfig.from_dict(tiny_vision_config_dict()))
        weights = {
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight": mx.zeros(
                (1,)
            ),
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max": mx.zeros(
                (1,)
            ),
            "model.encoder.embed_vision.embedding_projection.weight": mx.zeros((1,)),
        }
        sanitized = vision_model.sanitize(dict(weights))
        self.assertIn(
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
            sanitized,
        )
        self.assertIn(
            "model.encoder.embed_vision.embedding_projection.weight", sanitized
        )
        # Clipping calibration tensors are dropped when clipped linears are off.
        self.assertNotIn(
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max",
            sanitized,
        )

        text_model = Model(ModelConfig.from_dict(tiny_config_dict()))
        sanitized = text_model.sanitize(dict(weights))
        self.assertEqual(sanitized, {})


if __name__ == "__main__":
    unittest.main()
