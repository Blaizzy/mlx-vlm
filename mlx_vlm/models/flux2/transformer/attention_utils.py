import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class AttentionUtils:
    @staticmethod
    def process_qkv(
        hidden_states: mx.array,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        norm_q: nn.RMSNorm,
        norm_k: nn.RMSNorm,
        num_heads: int,
        head_dim: int,
    ):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query = to_q(hidden_states)
        key = to_k(hidden_states)
        value = to_v(hidden_states)

        # Reshape [B, S, H*D] -> [B, H, S, D]
        query = mx.transpose(
            mx.reshape(query, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3)
        )
        key = mx.transpose(
            mx.reshape(key, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3)
        )
        value = mx.transpose(
            mx.reshape(value, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3)
        )

        # Apply normalization in float32 for stability, then cast back (like working version)
        q_dtype = query.dtype
        k_dtype = key.dtype
        query = norm_q(query.astype(mx.float32)).astype(q_dtype)
        key = norm_k(key.astype(mx.float32)).astype(k_dtype)

        return query, key, value

    @staticmethod
    def compute_attention(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        mask: mx.array | None = None,
    ) -> mx.array:
        scale = 1 / mx.sqrt(query.shape[-1])
        hidden_states = scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(
            hidden_states, (batch_size, -1, num_heads * head_dim)
        )
        return hidden_states

    @staticmethod
    def convert_key_padding_mask_to_additive_mask(
        mask: mx.array | None,
        joint_seq_len: int,
        txt_seq_len: int,
    ) -> mx.array | None:
        if mask is None:
            return None

        bsz = mask.shape[0]
        img_seq_len = joint_seq_len - txt_seq_len

        # Create joint mask: [text_mask, image_ones]
        ones_img = mx.ones((bsz, img_seq_len), dtype=mx.float32)
        joint_mask = mx.concatenate([mask.astype(mx.float32), ones_img], axis=1)

        # Convert to additive mask for scaled_dot_product_attention
        additive = (1.0 - joint_mask) * (-1e9)
        return additive.reshape((additive.shape[0], 1, 1, additive.shape[1]))

    @staticmethod
    def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
        xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).astype(mx.float32), xk_out.reshape(
            *xk.shape
        ).astype(mx.float32)

    @staticmethod
    def apply_rope_bshd(xq: mx.array, xk: mx.array, cos: mx.array, sin: mx.array):
        out_dtype = xq.dtype
        xq_f = xq.astype(mx.float32)
        xk_f = xk.astype(mx.float32)
        cos_b = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
        sin_b = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

        def mix(x: mx.array) -> mx.array:
            x2 = x.reshape(*x.shape[:-1], -1, 2)
            real = x2[..., 0]
            imag = x2[..., 1]
            out0 = real * cos_b + (-imag) * sin_b
            out1 = imag * cos_b + (real) * sin_b
            out2 = mx.stack([out0, out1], axis=-1)
            return out2.reshape(*x.shape)

        return mix(xq_f).astype(out_dtype), mix(xk_f).astype(out_dtype)
