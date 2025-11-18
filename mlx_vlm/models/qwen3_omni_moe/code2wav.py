from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers import RMSNorm
import numpy as np
from transformers.models.dac.modeling_dac import DacDecoderBlock

from mlx_vlm.models.qwen3_omni_moe.config import Code2WavConfig

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = mx.zeros((in_features,)) * alpha
        self.beta = mx.zeros((in_features,)) * alpha
        self.no_div_by_zero = 0.000000001
    
    def __call__(self, hidden_states):
        alpha = mx.expand_dims(mx.expand_dims(self.alpha, axis=0), axis=-1)
        beta = mx.expand_dims(mx.expand_dims(self.beta, axis=0), axis=-1)
        alpha = mx.exp(alpha)
        beta = mx.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * mx.power(
            mx.sin(hidden_states * alpha), 2
        )
        return hidden_states


class LayerScale(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = mx.full((channels,), initial_scale)
    
    def __call__(self, x: mx.array):
        return self.scale * x


class RoPE(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        dim = head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )
        self.inv_freq = inv_freq
        self.attention_scaling = 1.0
    
    def __call__(self, x: mx.array, position_ids: mx.array):
        batch_size = position_ids.shape[0]
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, :, None].astype(mx.float32),
            (batch_size, self.inv_freq.shape[0], 1)
        )
        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), axis=1)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 1, 2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class CausalConvNet(nn.Module):
    def __init__(
        self, 
        in_chn, 
        out_chn,
        kernel_sz, 
        dilation=1,
        stride=1,
        groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_chn, out_chn, kernel_sz, 
            stride=stride, dilation=dilation, groups=groups
        )

        pass

class CausalTransConvNet(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_sz, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_chn, out_chn, kernel_sz, stride=stride
        )
        
        pass


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dwconv = CausalConvNet(
            dim, dim, kernel_sz=7, groups=dim, dilation=1
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.full((dim,), 1e-6) 

        pass

class Code2WavDecoderResUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConvNet(dim, dim, kernel_sz=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConvNet(dim, dim, kernel_sz=1)

        pass

class Code2WavDecoderBlock(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()

        in_dim = config.decoder_dim // 2 ** idx
        out_dim = config.decoder_dim // 2 ** (idx + 1)
        upsample_rate = config.upsample_rates[idx]

        self.block = [
            SnakeBeta(in_dim),
            CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]
        self.block.extend([
            Code2WavDecoderResUnit(out_dim, dilation) for dilation in (1, 3, 9)
        ])

        pass

class Code2WavAttention(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.q_proj = nn.Linear(
            config.hidden_size, 
            config.num_attention_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, 
            config.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, 
            config.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, 
            config.hidden_size, 
            bias=config.attention_bias
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

class Code2WavMlp(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, 
            self.hidden_size, 
            bias=False
        )
        
        if config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approx="precise")
        else:
            self.act_fn = nn.silu

class Code2WavTransformerLayer(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()
        self.self_attn = Code2WavAttention(config, idx)
        self.mlp = Code2WavMlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = LayerScale(config)
        self.mlp_layer_scale = LayerScale(config)

        pass


class Code2WavTransformerModel(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()

        self.layers = [
            Code2WavTransformerLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        

        pass


class Code2WavModel(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()

        self.pre_transformer = Code2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        self.upsample = [
            [
                CausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                ConvNeXtBlock(config.hidden_size)
            ] for factor in config.upsampling_ratios
        ]
        self.decoder = [CausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        self.decoder.extend([
            Code2WavDecoderBlock(config, idx) for idx in range(len(config.upsample_rates))
        ])
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        self.decoder.extend([
            SnakeBeta(output_dim),
            CausalConvNet(output_dim, 1, 7)
        ])
        
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            # Convert ConvTranspose1d weights from PyTorch [in, out, k] to MLX [out, k, in]
            if ("upsample" in k and "conv.weight" in k and "dwconv" not in k) or \
               ("decoder" in k and "block" in k and "conv.weight" in k and "conv1" not in k and "conv2" not in k):
                sanitized_weights[k] = v.transpose(1, 2, 0)
            # Convert Conv1d weights from PyTorch [out, in, k] to MLX [out, k, in]
            elif ("dwconv.conv.weight" in k) or \
                 ("decoder" in k and "conv.weight" in k and "block" not in k) or \
                 ("decoder" in k and "block" in k and ("conv1.conv.weight" in k or "conv2.conv.weight" in k)):
                sanitized_weights[k] = v.transpose(0, 2, 1)
            else:
                sanitized_weights[k] = v
        
        return sanitized_weights