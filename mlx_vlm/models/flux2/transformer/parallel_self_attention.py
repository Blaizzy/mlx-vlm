import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.transformer.attention_utils import AttentionUtils
from mlx_vlm.models.flux2.transformer.feed_forward import Flux2SwiGLU


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(
            dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False
        )
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.mlp_act = Flux2SwiGLU()
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array, image_rotary_emb):
        proj = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
        query, key, value = mx.split(qkv, 3, axis=-1)

        batch, seq_len, _ = query.shape
        query = mx.transpose(
            mx.reshape(query, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )
        key = mx.transpose(
            mx.reshape(key, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )
        value = mx.transpose(
            mx.reshape(value, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )

        query = self.norm_q(query.astype(mx.float32)).astype(ModelConfig.precision)
        key = self.norm_k(key.astype(mx.float32)).astype(ModelConfig.precision)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        hidden_states = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        mlp_hidden = self.mlp_act(mlp_hidden)
        hidden_states = mx.concatenate([hidden_states, mlp_hidden], axis=-1)
        hidden_states = self.to_out(hidden_states)
        return hidden_states
