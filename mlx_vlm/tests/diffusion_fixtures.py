"""Tiny DiffusionGemma fixtures shared by model, generation, CLI, and APC tests."""

import mlx.core as mx

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


def make_diffusion_model(config=None, *, vision=False, seed=0):
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

    mx.random.seed(seed)
    if config is None:
        config = tiny_vision_config_dict() if vision else tiny_config_dict()
    return Model(ModelConfig.from_dict(config))


def diffusion_responses(model, input_ids=(2, 3), **kwargs):
    from mlx_vlm.generate import stream_generate

    options = {"max_tokens": 2, **kwargs}
    return list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([input_ids], dtype=mx.int32),
            **options,
        )
    )
