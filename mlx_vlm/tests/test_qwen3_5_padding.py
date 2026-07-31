import mlx.core as mx

from mlx_vlm.models.qwen3_5.language import _qwen3_5_fully_padded_row_output


def test_qwen3_5_fully_cached_batch_row_keeps_shape_without_forwarding():
    hidden_states = mx.ones((3, 4, 8))

    output = _qwen3_5_fully_padded_row_output(
        hidden_states,
        row=1,
        pad=hidden_states.shape[1],
    )
    mx.eval(output)

    assert output.shape == (1, 4, 8)
    assert output.tolist() == [[[0.0] * 8] * 4]
