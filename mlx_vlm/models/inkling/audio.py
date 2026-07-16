import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig


class InklingAudioModelEmbeddings(nn.Module):
    """Sums one embedding lookup per dMel codebook into a single frame vector."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.embed_audio_tokens = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        # Non-persistent in the checkpoint (torch registers it with
        # `persistent=False`); recomputed here instead of stored as a param.
        offsets = mx.arange(self.n_mel_bins) * self.mel_vocab_size
        inputs_embeds = self.embed_audio_tokens(input_ids + offsets)
        return inputs_embeds.sum(axis=-2)


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.embed_audio_tokens = InklingAudioModelEmbeddings(config)
        self.norm = nn.RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def __call__(self, audio_input_ids: mx.array) -> mx.array:
        h = self.embed_audio_tokens(audio_input_ids)
        return self.norm(h)
