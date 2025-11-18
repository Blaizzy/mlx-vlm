from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_omni_moe.config import CodePredictorConfig, TalkerConfig, TextConfig
from .language import Attention, MLP, Qwen3OmniMoeThinkerTextSparseMoeBlock

class CodePredictorMLP(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Set activation function based on config
        if config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approx="precise")
        else:
            self.act_fn = nn.silu


class CodePredictorAttention(nn.Module):
    def __init__(self, config: CodePredictorConfig, idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # Note: q_norm and k_norm are applied on head_dim, not hidden_size
        self.q_norm = nn.RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if (hasattr(config, 'layer_types') and config.layer_types and idx < len(config.layer_types) and config.layer_types[idx] == "sliding_attention") else None


class CodePredictorDecoderLayer(nn.Module):
    def __init__(self, config: CodePredictorConfig, idx: int):
        self.self_attn = CodePredictorAttention(config, idx)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        super().__init__()

class CodePredictorModel(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.layers = [
            CodePredictorDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size) for _ in range(config.num_code_groups - 1)
        ]

        pass

class CodePredictor(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()

        self.model = CodePredictorModel(config)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)
        ]

        pass

class TalkerResizeMlp(nn.Module):
    def __init__(self, config: TalkerConfig):
        super().__init__()

        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, config.text_config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=True)

class TalkerTextMlp(nn.Module):
    def __init__(self, config: TextConfig, intermediate_sz: int):
        super().__init__()
        if not intermediate_sz:
            intermediate_sz = config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_sz, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_sz, bias=False)
        self.down_proj = nn.Linear(intermediate_sz, config.hidden_size, bias=False)

class TalkerSparseMoeBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = [
            TalkerTextMlp(config, config.moe_intermediate_size) for _ in range(config.num_experts)
        ]
        self.shared_expert = TalkerTextMlp(config, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)


class TalkerModelDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        
        # if (idx not in config.mlp_only_layers) and \
        #     (config.num_experts > 0 and (idx + 1) % config.decoder_sparse_step == 0):
        #     self.mlp = MLP(config.hidden_size, config.intermediate_size)
        # else:
        #     self.mlp = Qwen3OmniMoeThinkerTextSparseMoeBlock(config)
        
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TalkerSparseMoeBlock(config)

        pass

class TalkerModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.layers = [
            TalkerModelDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        pass

class Talker(nn.Module):
    def __init__(self, config: TalkerConfig):
        super().__init__()

        self.model = TalkerModel(config.text_config)
        self.text_projection = TalkerResizeMlp(config)
        self.hidden_projection = TalkerResizeMlp(config)
        self.code_predictor = CodePredictor(config.code_predictor_config)

        self.codec_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        pass

