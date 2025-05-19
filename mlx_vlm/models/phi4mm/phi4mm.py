import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor

from ...trainer.utils import LoRaLayer, apply_lora_layers, set_module_by_name
from ...utils import get_model_path
from ..base import BaseModelConfig, LanguageModelOutput, create_attention_mask
from .multimodal import Phi4MMImageAudioEmbedding
from .su_rope import SuScaledRotaryEmbedding


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    pad_token_id: int
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    embd_layer: Optional[Dict[str, str]] = None
    image_size: Optional[int] = 224
    patch_size: Optional[int] = 14
    audio_processor: Optional[Dict[str, Any]] = None
    vision_lora: Optional[Dict[str, Any]] = None
    speech_lora: Optional[Dict[str, Any]] = None
    resid_pdrop: float = 0.0
    sliding_window: int = 262144
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["longrope", "su", "linear"]:
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear', 'su', and 'longrope'; setting rope scaling to false."
                )
                self.rope_scaling = None


class Phi4MMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi4MMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, bias: bool = False):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        assert config.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers

        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        rope_dim = int(head_dim * config.partial_rotary_factor)
        if config.rope_scaling and config.rope_scaling["type"] in ["longrope", "su"]:
            self.rope = SuScaledRotaryEmbedding(
                rope_dim,
                base=config.rope_theta,
                traditional=config.rope_traditional,
                max_position_embeddings=config.max_position_embeddings,
                original_max_position_embeddings=config.original_max_position_embeddings,
                short_factor=config.rope_scaling["short_factor"],
                long_factor=config.rope_scaling["long_factor"],
            )
        else:
            rope_scale = 1.0
            if config.rope_scaling and config.rope_scaling["type"] == "linear":
                assert isinstance(config.rope_scaling["factor"], float)
                rope_scale = 1 / config.rope_scaling["factor"]
            self.rope = nn.RoPE(
                rope_dim,
                traditional=config.rope_traditional,
                base=config.rope_theta,
                scale=rope_scale,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if mask is not None and isinstance(mask, mx.array):
            key_len = keys.shape[-2]
            if mask.shape[-1] != key_len:
                mask = mask[..., -key_len:]

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Phi4MMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Phi4MMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


@partial(mx.compile)
def pad_embeddings(embeddings, padding_idx):
    # Create mask where padding_idx is 0 and everything else is 1
    mask = mx.where(
        embeddings == padding_idx,
        mx.zeros(embeddings.shape, dtype=mx.float32),
        mx.ones(embeddings.shape, dtype=mx.float32),
    )

    # Apply mask to zero out padding embeddings
    masked_embeddings = embeddings * mask

    return masked_embeddings


class Phi4Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens_extend = None
        if isinstance(config.embd_layer, dict):
            embedding_config = {
                "embedding_cls": config.embd_layer["embedding_cls"],
                **config.embd_layer,
            }
            self.embed_tokens_extend = Phi4MMImageAudioEmbedding(
                config, **embedding_config
            )
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # LoRA related settings
        assert getattr(config, "speech_lora", None) is not None
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
                if re.match(config.speech_lora["layer"], name):
                    # print(f"Applying Speech LoRA to {name}")
                    lora_layer = LoRaLayer(
                        module,
                        config.speech_lora["r"],
                        config.speech_lora["lora_alpha"],
                        config.speech_lora["dp"],
                        "speech",
                    )

                    set_module_by_name(self, name, lora_layer)

        self.config.speech_lora["r"] = config.speech_lora["r"]
        self.config.speech_lora["lora_alpha"] = config.speech_lora["lora_alpha"]
        self.config.speech_lora["layer"] = config.speech_lora["layer"]
        self.config.speech_lora["dp"] = config.speech_lora["dp"]

        assert getattr(config, "vision_lora", None) is not None
        for name, module in self.named_modules():

            if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
                if re.match(config.vision_lora["layer"], name):
                    # print(f"Applying Vision LoRA to {name}")
                    lora_layer = LoRaLayer(
                        module,
                        config.vision_lora["r"],
                        config.vision_lora["lora_alpha"],
                        config.vision_lora["dp"],
                        "vision",
                    )
                    name = name.replace(".base_layer", "")
                    set_module_by_name(self, name, lora_layer)

        self.config.vision_lora["r"] = config.vision_lora["r"]
        self.config.vision_lora["lora_alpha"] = config.vision_lora["lora_alpha"]
        self.config.vision_lora["layer"] = config.vision_lora["layer"]
        self.config.vision_lora["dp"] = config.vision_lora["dp"]

    def __call__(
        self,
        input_ids: mx.array,
        input_embeds: mx.array = None,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ):
        input_mode = kwargs.pop("input_mode", 0)
        if isinstance(input_mode, mx.array):
            assert len(input_mode) == 1
            input_mode = input_mode[0].item()
        input_mode = InputMode(input_mode)

        if input_mode in [InputMode.VISION_SPEECH, InputMode.VISION]:
            self.unset_lora_adapter()
            self.set_lora_adapter("vision")
            audio_projection_mode = "vision"
        elif input_mode == InputMode.SPEECH:
            self.set_lora_adapter("speech")
            audio_projection_mode = "speech"
        elif input_mode == InputMode.LANGUAGE:
            self.unset_lora_adapter()
            audio_projection_mode = "speech"
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        if input_embeds is None and pixel_values is not None:
            h = self.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=None,
                input_image_embeds=kwargs.pop(
                    "input_image_embeds", pixel_values.transpose(0, 1, 3, 4, 2)
                ),
                input_audio_embeds=kwargs.pop("input_audio_embeds", None),
                image_sizes=kwargs.pop("image_sizes", None),
                image_attention_mask=kwargs.pop("image_attention_mask", None),
                audio_embed_sizes=kwargs.pop("audio_embed_sizes", None),
                audio_attention_mask=kwargs.pop("audio_attention_mask", None),
                audio_projection_mode=audio_projection_mode,
                wte=self.embed_tokens,
            )
        elif input_embeds is None and pixel_values is None:
            h = self.embed_tokens(input_ids)
            h = pad_embeddings(h, self.config.pad_token_id)
        else:
            h = input_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        # if mask is None:
        mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        out = self.norm(h)

        if self.config.tie_word_embeddings:
            out = self.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return LanguageModelOutput(logits=out)

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def set_lora_adapter(self, adapter_name):
        if adapter_name == "vision":
            path = get_model_path("microsoft/Phi-4-multimodal-instruct")
            adapter_path = path / "vision-lora/adapter_model.safetensors"
            adapter_config = path / "vision-lora/adapter_config.json"

        elif adapter_name == "speech":
            path = get_model_path("microsoft/Phi-4-multimodal-instruct")
            adapter_path = path / "speech-lora/adapter_model.safetensors"
            adapter_config = path / "speech-lora/adapter_config.json"
        else:
            raise ValueError(f"Invalid adapter_name: {adapter_name}")

        with open(adapter_config, "r") as f:
            adapter_config = json.load(f)

        weights = mx.load(str(adapter_path))

        # Sanitize weights to match the expected format
        sanitized_weights = {}
        for k, v in weights.items():
            if "base_model.model.model.layers" in k and "lora_A.weight" in k:
                new_k = k.replace("base_model.model.model.layers", "layers")
                new_k = new_k.replace("lora_A.weight", f"lora_A.{adapter_name}.weight")
                sanitized_weights[new_k] = v.transpose(1, 0)
            elif "base_model.model.model.layers" in k and "lora_B.weight" in k:
                new_k = k.replace("base_model.model.model.layers", "layers")
                new_k = new_k.replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
                sanitized_weights[new_k] = v.transpose(1, 0)
            else:
                sanitized_weights[k] = v

        weights = sanitized_weights

        self.load_weights(list(weights.items()), strict=False)
        print(f"Loaded adapter weights for {adapter_name}")
        self.eval()
        for module in self.layers:
            for name, module in module.named_modules():
                if isinstance(module, LoRaLayer):
                    if "lora_A." + adapter_name + ".weight" in name:
                        module.disable_adapter = False
                    if "lora_B." + adapter_name + ".weight" in name:
                        module.disable_adapter = False

    def unset_lora_adapter(self):
        for module in self.layers:
            for name, module in module.named_modules():
                if isinstance(module, LoRaLayer):
                    if module.lora_name == "speech":
                        # print(f"Unsetting LoRA adapter for {name}, module: {module.lora_name}")
                        module.disable_adapter = True
                    if module.lora_name == "vision":
                        # print(f"Unsetting LoRA adapter for {name}, module: {module.lora_name}")
                        module.disable_adapter = True


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.language_model = Phi4Model(config)
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out = self.language_model(
            input_ids, pixel_values=pixel_values, mask=mask, cache=cache, **kwargs
        )
        return out

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            new_k = k
            if k.startswith("model"):
                new_k = k.replace("model.", "language_model.")
            if "lm_head" in k and "language_model.lm_head" not in k:
                new_k = k.replace("lm_head.", "language_model.lm_head.")
            sanitized_weights[new_k] = v
        weights = sanitized_weights

        for k, v in weights.items():
            if "lora_A.vision.weight" in k:
                weights[k] = v.transpose(1, 0)
            elif "lora_B.vision.weight" in k:
                weights[k] = v.transpose(1, 0)
            elif "lora_A.speech.weight" in k:
                weights[k] = v.transpose(1, 0)
            elif "lora_B.speech.weight" in k:
                weights[k] = v.transpose(1, 0)
            else:
                weights[k] = v

        weights = self.language_model.embed_tokens_extend.image_embed.sanitize(weights)
        weights = self.language_model.embed_tokens_extend.audio_embed.sanitize(weights)
        return weights
