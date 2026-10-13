"""Process-local benchmark adapter for the unaligned BF16 vocabulary head.

The extra padded weight copy is deliberately kept out of model parameters.
This prototype does not alter checkpoint loading or the default forward path.
"""


def install_padded_head(lm):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5SpeculativeVerifier
    from mlx_vlm.models.simd_mma import simd_mmas, supports_simd_linear

    if not Qwen3_5SpeculativeVerifier.supports_model(lm):
        raise ValueError("The SIMD verifier requires a supported GPU and model")
    head = lm.lm_head
    if (
        type(head) is not nn.Linear
        or head.weight.dtype != mx.bfloat16
        or "bias" in head
    ):
        raise ValueError("Experiment only covers an unquantized BF16 head without bias")
    vocab, width = head.weight.shape
    padded = head
    if vocab % 16:
        padded = nn.Linear(width, (vocab + 15) // 16 * 16, bias=False)
        padded.weight = mx.pad(head.weight, [(0, (-vocab) % 16), (0, 0)])
    mx.eval(padded.parameters())
    assert supports_simd_linear(padded)

    class PaddedHeadVerifier(Qwen3_5SpeculativeVerifier):
        def _linear(self, linear, x):
            if linear is head:
                output = simd_mmas((padded,), x, chunked=True)
                if output is not None:
                    return output[0][..., :vocab]
            return super()._linear(linear, x)

        def quantized_linear(self, linear, x):
            if linear is head:
                return self._linear(linear, x)
            return super().quantized_linear(linear, x)

        def quantized_argmax(self, linear, x, token_mask=None):
            if linear is head and token_mask is None:
                return mx.argmax(self._linear(linear, x), axis=-1)
            if linear is head:
                from mlx_vlm.models.quantized_verifier import _masked_argmax

                return _masked_argmax(self._linear(linear, x), token_mask)
            return super().quantized_argmax(linear, x, token_mask)

    verifier = PaddedHeadVerifier()
    # Only this MiniCPM class in this benchmark process uses the adapter.
    type(lm)._get_speculative_verifier = lambda self: verifier
    return verifier
