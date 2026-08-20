import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.qwen.decoder_layer import Qwen3VLDecoderLayer
from mlx_vlm.models.flux2.qwen.rms_norm import Qwen3VLRMSNorm
from mlx_vlm.models.flux2.qwen.rotary import Qwen3TextRotaryEmbedding


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: list[int] | None = None,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=None,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=attention_scaling,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=mask_dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len), dtype=mask_dtype)
            neginf_2d = mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype)
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(
                mx.expand_dims(causal_tri_mask, axis=0), axis=0
            )
            causal_tri_mask = mx.broadcast_to(
                causal_tri_mask, (batch_size, 1, seq_len, seq_len)
            )
        attention_mask_4d = causal_tri_mask + padding_mask

        position_ids = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Match HF behavior: include embedding output as the first hidden state.
        hidden_states_list = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask_4d,
                position_embeddings,
                past_key_value=None,
            )
            if output_hidden_states:
                hidden_states_list.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_states_list

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        hidden_state_layers: tuple[int, ...] = (9, 18, 27),
    ) -> mx.array:
        _, hidden_states_list = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hidden_states_list is None:
            raise RuntimeError("Hidden states not available for prompt embedding.")
        stacked = mx.stack([hidden_states_list[i] for i in hidden_state_layers], axis=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = mx.transpose(stacked, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )
        return prompt_embeds
