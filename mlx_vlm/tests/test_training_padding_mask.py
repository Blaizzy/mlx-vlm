import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.vlm.cross_entropy.runtime import (
    PADDING_MASK_MODEL_TYPES,
    forward_vlm_logits,
    model_uses_padding_mask,
)


class _MaskForwardingVLM(nn.Module):
    """Mimics ~40 models: forwards the 2D mask straight into attention."""

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(11, 16)
        self.q = nn.Linear(16, 16)
        self.out = nn.Linear(16, 11)

    def __call__(self, input_ids, pixel_values=None, mask=None, **kwargs):
        h = self.embed(input_ids)
        q = self.q(h)[:, None]
        if mask is None:
            mask = "causal"
        attn = mx.fast.scaled_dot_product_attention(q, q, q, scale=0.25, mask=mask)
        return self.out(attn[:, 0])


def _batch():
    ids = mx.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 0, 0]])
    return {"input_ids": ids, "attention_mask": (ids != 0).astype(mx.int32)}


def test_default_models_do_not_receive_padding_mask():
    model = _MaskForwardingVLM("some_vlm")
    batch = _batch()
    assert not model_uses_padding_mask(model)
    logits = forward_vlm_logits(
        model, batch["input_ids"], batch["attention_mask"], batch
    )
    expected = model(batch["input_ids"], None, None)
    assert mx.allclose(logits, expected.astype(mx.float32)).item()


def test_single_row_matches_causal():
    model = _MaskForwardingVLM("some_vlm")
    ids = mx.array([[1, 2, 3, 4]])
    batch = {"input_ids": ids, "attention_mask": mx.ones_like(ids)}
    logits = forward_vlm_logits(model, ids, batch["attention_mask"], batch)
    assert mx.allclose(logits, model(ids).astype(mx.float32)).item()


def test_right_padding_does_not_change_real_token_logits():
    model = _MaskForwardingVLM("some_vlm")
    batch = _batch()
    padded = forward_vlm_logits(
        model, batch["input_ids"], batch["attention_mask"], batch
    )
    short = model(mx.array([[1, 2, 3, 4]]))
    assert mx.allclose(padded[1, :4], short[0].astype(mx.float32), atol=1e-5).item()


def test_padding_mask_allowlist_models_keep_mask():
    for model_type in PADDING_MASK_MODEL_TYPES:
        assert model_uses_padding_mask(_MaskForwardingVLM(model_type))
    assert not model_uses_padding_mask(_MaskForwardingVLM("gemma3"))
    assert not model_uses_padding_mask(_MaskForwardingVLM("glm_ocr"))
