import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.transformer.attention_utils import AttentionUtils
from mlx_vlm.models.flux2.transformer.kv_cache import Flux2KVLayerCache


class Flux2Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int | None = None
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=1e-5)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=1e-5)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb,
        *,
        kv_cache: Flux2KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
    ):
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        enc_query = enc_key = enc_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            enc_query, enc_key, enc_value = AttentionUtils.process_qkv(
                hidden_states=encoder_hidden_states,
                to_q=self.add_q_proj,
                to_k=self.add_k_proj,
                to_v=self.add_v_proj,
                norm_q=self.norm_added_q,
                norm_k=self.norm_added_k,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
            query = mx.concatenate([enc_query, query], axis=2)
            key = mx.concatenate([enc_key, key], axis=2)
            value = mx.concatenate([enc_value, value], axis=2)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        num_txt_tokens = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        )
        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, :, ref_start:ref_end], value[:, :, ref_start:ref_end])

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = AttentionUtils.compute_kv_cache_attention(
                query=query,
                key=key,
                value=value,
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
                num_txt_tokens=num_txt_tokens,
                num_ref_tokens=num_ref_tokens,
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = AttentionUtils.compute_kv_cache_attention(
                query=query,
                key=key,
                value=value,
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
                num_txt_tokens=num_txt_tokens,
                num_ref_tokens=0,
                kv_cache=kv_cache,
            )
        else:
            hidden_states = AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
            )

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out(hidden_states)
        return hidden_states, encoder_hidden_states
