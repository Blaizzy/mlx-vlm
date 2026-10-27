import mlx.core as mx
import mlx.nn as nn

from ..qwen2.config import ModelConfig as Qwen2Config
from ..qwen2.language import TransformerBlock
from .config import AudioConfig


class AudioTransformer(nn.Module):
    def __init__(self, config: AudioConfig):
        args = Qwen2Config(
            model_type="qwen2",
            hidden_size=config.input_local_dim,
            num_hidden_layers=config.input_local_layers,
            intermediate_size=config.input_local_intermediate_size,
            num_attention_heads=config.input_local_attn_heads,
            num_key_value_heads=config.input_local_attn_heads,
            rms_norm_eps=1e-6,
            vocab_size=1,
            rope_theta=config.rope_theta,
        )
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = (
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            if config.add_post_norm
            else nn.Identity()
        )
        self.full_attention = config.input_full_attention

    def __call__(self, x):
        mask = None
        if not self.full_attention:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AudioProjection(nn.Module):
    def __init__(self, input_size, output_size):
        self.mlp = [
            nn.Linear(input_size, input_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(input_size * 4, output_size, bias=False),
        ]

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        self.config = config
        self.input_local_transformer = AudioTransformer(config)
        projection_input = config.input_local_dim * config.group_size
        if config.projection_layers == 1:
            self.projection = nn.Linear(
                projection_input, config.out_hidden_size, bias=False
            )
        elif config.projection_layers == 2:
            self.projection = AudioProjection(projection_input, config.out_hidden_size)
        else:
            raise ValueError(
                f"Unsupported projection_layers={config.projection_layers}"
            )

    def __call__(self, audio_codes, speech_embeddings):
        if audio_codes.ndim != 2:
            raise ValueError(
                f"audio_codes must be 2D, received shape {audio_codes.shape}"
            )
        if audio_codes.shape[0] == 0:
            raise ValueError("audio_codes must contain at least one frame")
        if audio_codes.shape[1] < self.config.audio_channels:
            raise ValueError(
                f"audio_codes must contain at least {self.config.audio_channels} channels"
            )
        audio_codes = audio_codes[:, : self.config.audio_channels]
        length = audio_codes.shape[0]
        padded_length = (
            (length + self.config.group_size - 1) // self.config.group_size
        ) * self.config.group_size
        if padded_length > length:
            audio_codes = mx.concatenate(
                [
                    audio_codes,
                    mx.repeat(audio_codes[-1:], padded_length - length, axis=0),
                ]
            )
        audio_codes = audio_codes.reshape(
            -1, self.config.group_size, self.config.audio_channels
        )
        embeddings = sum(
            speech_embeddings[i](audio_codes[:, :, i])
            for i in range(self.config.audio_channels)
        )
        hidden = self.input_local_transformer(embeddings)
        return self.projection(hidden.reshape(hidden.shape[0], -1))


def build_speech_embeddings(config: AudioConfig):
    return [
        nn.Embedding(config.speech_vocab_size, config.input_local_dim)
        for _ in range(config.audio_channels)
    ]
