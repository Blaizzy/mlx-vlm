import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig


class AudioModel(nn.Module):
    """dMel audio front end (HiggsAudioV2-style): each of ``n_mel_bins`` mel channels is
    discretized into ``mel_vocab_size`` buckets; per-channel bins are offset into a single
    embedding table, looked up, and summed, then RMS-normed into LM space."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.model_type = config.model_type
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.hidden_size = config.text_hidden_size
        self.max_frames_per_chunk = config.max_frames_per_chunk
        if self.max_frames_per_chunk <= 0:
            raise ValueError("max_frames_per_chunk must be positive")
        self.embed_audio_tokens = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size
        )
        self.norm = nn.RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def _embed(self, audio_input_ids):
        offsets = mx.arange(self.n_mel_bins) * self.mel_vocab_size
        embeds = self.embed_audio_tokens(audio_input_ids + offsets)
        embeds = embeds.sum(axis=-2)
        return self.norm(embeds)

    def __call__(self, audio_input_ids):
        """audio_input_ids: [..., frames, n_mel_bins] of bucket indices -> [..., frames, hidden]."""
        output_shape = (*audio_input_ids.shape[:-1], self.hidden_size)
        frames = audio_input_ids.reshape(-1, self.n_mel_bins)
        if frames.shape[0] <= self.max_frames_per_chunk:
            return self._embed(frames).reshape(output_shape)

        chunks = []
        for start in range(0, frames.shape[0], self.max_frames_per_chunk):
            features = self._embed(frames[start : start + self.max_frames_per_chunk])
            mx.eval(features)
            chunks.append(features)
        return mx.concatenate(chunks, axis=0).reshape(output_shape)
