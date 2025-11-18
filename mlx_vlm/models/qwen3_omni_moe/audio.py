f"""
thinker.audio_tower:
    - conv2d1: weight & b
    - conv2d2: weight & b
    - conv2d3: weight & b
    - conv_out: weight
    - layers.[i].fc1: weight & b
    - layers.[i].fc2: weight & b
    - layers.[i].final_layer_norm: weight & b
    - layers.[i].self_attn.q_proj: weight & b
    - layers.[i].self_attn.k_proj: weight & b
    - layers.[i].self_attn.v_proj: weight & b
    - layers.[i].self_attn.out_proj: weight & b
    - layers.[i].self_attn_layer_norm: weight & b
    - ln_post: weight & b
    - proj1: weight & b
    - proj2: weight & b

thinker.model:
    - embed_tokens: weight
    - norm: weight
    - layers.[i].input_layernorm: weight
    - layers.[i].mlp.experts.[j].down_proj: weight
    - layers.[i].mlp.experts.[j].gate_proj: weight
    - layers.[i].mlp.experts.[j].up_proj: weight
    - layers.[i].mlp.gate: weight
    - layers.[i].post_attention_layernorm: weight
    - layers.[i].self_attn.k_norm: weight
    - layers.[i].self_attn.k_proj: weight
    - layers.[i].self_attn.q_norm: weight
    - layers.[i].self_attn.q_proj: weight
    - layers.[i].self_attn.o_proj: weight
    - layers.[i].self_attn.v_proj: weight


thinker.visual:
    - blocks.[i].attn.proj: w & b
    - blocks.[i].attn.qkv: w & b
    - blocks.[i].mlp.linear_fc1: w & b
    - blocks.[i].mlp.linear_fc2: w & b
    - blocks.[i].norm1: w & b
    - blocks.[i].norm2: w & b
    - merger.ln_q: w & b
    - merger.mlp.0: w & b
    - merger.mlp.2: w & b
    - merger_list.[i].ln_q: w & b
    - merger_list.[i].mlp.0: w & b
    - merger_list.[i].mlp.2: w & b
    - patch_embed.proj: w & b
    - pos_embed: w

code2wav:
    - code_embedding: w
    - decoder.[i].conv: w & b
    - decoder.[i].block.[j]: alpha & beta
    - decoder.[i].block.[j].conv: w & b
    - decoder.[i].block.[j].act1: alpha & beta
    - decoder.[i].block.[j].act2: alpha & beta
    - decoder.[i].block.[j].conv1.conv: w & b
    - decoder.[i].block.[j].conv2.conv: w & b
    - pre_transformer.layers.[i].input_layernorm: w
    - pre_transformer.layers.[i].mlp.down_proj: w
    - pre_transformer.layers.[i].mlp.gate_proj: w
    - pre_transformer.layers.[i].mlp.up_proj: w
    - pre_transformer.layers.[i].mlp_layer_scale: scale
    - pre_transformer.layers.[i].post_attention_layernorm: w
    - pre_transformer.layers.[i].self_attn.k_proj: w
    - pre_transformer.layers.[i].self_attn.q_proj: w
    - pre_transformer.layers.[i].self_attn.v_proj: w
    - pre_transformer.layers.[i].self_attn.o_proj: w
    - pre_transformer.layers.[i].self_attn.k_proj: w
    - pre_transformer.layers.[i].self_attn_layer_scale: scale
    - pre_transformer.norm: w
    - upsample.[i].0.conv: w & b
    - upsample.[i].1.dwconv.conv: 2 & b
    - upsample.[i].1: gamma
    - upsample.[i].1.norm: 2 & b
    - upsample.[i].1.pwconv1: 2 & b
    - upsample.[i].1.pwconv2: 2 & b


talker: 
    - codec_head: w
    - hidden_projection.linear_fc1: w & b
    - hidden_projection.linear_fc2: w & b
    - text_projection.linear_fc1: w & b
    - text_projection.linear_fc2: w & b

talker.code_predictor: 
    - lm_head.[i]: w
    - lm_head.[i]: w
    - model.codec_embedding.[i]: w
    - model.layers[i].input_layernorm: w
    - model.layers[i].mlp.down_proj: w
    - model.layers[i].mlp.gate_proj: w
    - model.layers[i].mlp.up_proj: w
    - model.layers[i].post_attention_layernorm: w
    - model.layers[i].self_attn.k_norm: w
    - model.layers[i].self_attn.q_norm: w
    - model.layers[i].self_attn.k_proj: w
    - model.layers[i].self_attn.q_proj: w
    - model.layers[i].self_attn.v_proj: w
    - model.layers[i].self_attn.o_proj: w
    - model.norm: w

talker.model: 
    - codec_embedding: w
    - layers.[i].input_layernorm: w
    - layers.[i].mlp.experts.[j].down_proj: w
    - layers.[i].mlp.experts.[j].gate_proj: w
    - layers.[i].mlp.experts.[j].up_proj: w
    - layers.[i].mlp.experts.[j].down_proj: w
    - layers.[i].mlp.gate: w
    - layers.[i].mlp.shared_expert_down_proj: w
    - layers.[i].mlp.shared_expert_gate_proj: w
    - layers.[i].mlp.shared_expert_up_proj: w
    - layers.[i].mlp.shared_expert_gate_proj: w
    - layers.[i].post_attention_layernorm: w
    - layers.[i].self_attn.k_norm: w
    - layers.[i].self_attn.q_norm: w
    - layers.[i].self_attn.k_proj: w
    - layers.[i].self_attn.q_proj: w
    - layers.[i].self_attn.v_proj: w
    - layers.[i].self_attn.o_proj: w
    - norm: w




"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from regex import T

from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig
from .vision import check_array_shape


class Attention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()

        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        
class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig, idx: int):
        super().__init__()

        self.self_attn = Attention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    

class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        
        self.layers = [AudioEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model, 
            bias=False
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

        self.ln_post = nn.LayerNorm(config.d_model)


    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "audio_tower.conv2d" in k and "weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                # if check_array_shape(v):
                #     sanitized_weights[k] = v
                # else:
                #     sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                #     pass
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights


