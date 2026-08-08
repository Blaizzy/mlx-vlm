import inspect

import mlx.core as mx
import numpy as np

from mlx_vlm.models.sam3.config import TrackerMaskDecoderConfig
from mlx_vlm.models.sam3.sam_components import (
    PositionalEmbedding,
    SAMMaskDecoder,
    TwoWayAttentionBlock,
)
from mlx_vlm.models.sam3.tracker import TrackerModel


def _small_decoder() -> SAMMaskDecoder:
    return SAMMaskDecoder(
        TrackerMaskDecoderConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            attention_downsample_rate=1,
            num_multimask_outputs=3,
            mlp_dim=16,
        )
    )


def test_dense_position_encoding_uses_pixel_centers():
    embedding = PositionalEmbedding(num_pos_feats=2)
    embedding.positional_embedding = mx.array(
        [[1.0, 0.25], [0.5, 2.0]], dtype=mx.float32
    )
    centered = mx.array(
        [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]],
        dtype=mx.float32,
    )

    actual = embedding((2, 2))
    expected = embedding.forward_with_coords(centered)[0]

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


def test_unstable_single_mask_uses_best_multimask_candidate():
    decoder = _small_decoder()
    masks = mx.array(
        [[
            [[0.1, 0.0], [0.0, -0.1]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ]],
        dtype=mx.float32,
    )
    scores = mx.array([[0.1, 0.2, 0.9, 0.3]], dtype=mx.float32)

    selected_masks, selected_scores = decoder._select_masks(
        masks, scores, multimask_output=False
    )

    np.testing.assert_allclose(np.asarray(selected_masks), 3.0)
    np.testing.assert_allclose(np.asarray(selected_scores), [[0.9]])


def test_stable_single_mask_is_preserved():
    decoder = _small_decoder()
    masks = mx.ones((1, 4, 2, 2), dtype=mx.float32)
    masks = masks.at[:, 1:].multiply(5.0)
    scores = mx.array([[0.7, 0.9, 0.8, 0.6]], dtype=mx.float32)

    selected_masks, selected_scores = decoder._select_masks(
        masks, scores, multimask_output=False
    )

    np.testing.assert_allclose(np.asarray(selected_masks), 1.0)
    np.testing.assert_allclose(np.asarray(selected_scores), [[0.7]])


def test_decoder_source_preserves_trained_token_and_skip_invariants():
    source = inspect.getsource(SAMMaskDecoder.__call__)
    assert source.index("self.obj_score_token.weight") < source.index(
        "self.iou_token.weight"
    )
    assert source.index("self.iou_token.weight") < source.index(
        "self.mask_tokens.weight"
    )
    assert "mx.sigmoid(self.iou_prediction_head" in source
    assert "self.conv_s1(high_res_features[1])" in source
    assert "self.conv_s0(high_res_features[0])" in source
    assert source.index("self.conv_s1(high_res_features[1])") < source.index(
        "self.upscale_layer_norm(upscaled)"
    )


def test_first_two_way_layer_and_no_memory_path_are_explicit():
    attention_source = inspect.getsource(TwoWayAttentionBlock.__call__)
    tracker_source = inspect.getsource(TrackerModel.track_step)
    assert "if skip_first_layer_pe" in attention_source
    assert "self.self_attn(queries, queries, queries)" in attention_source
    assert "src = src + self.no_memory_embedding" in tracker_source
