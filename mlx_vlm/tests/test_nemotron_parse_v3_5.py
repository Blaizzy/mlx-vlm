"""Small contracts; full checkpoint fidelity is measured locally on Apple Silicon."""

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.nemotron_parse_v3_5 import Model, ModelConfig, TextConfig
from mlx_vlm.models.nemotron_parse_v3_5.language import LanguageModel
from mlx_vlm.models.nemotron_parse_v3_5.processing_nemotron_parse_v3_5 import (
    ImageProcessor,
)
from mlx_vlm.utils import get_model_type


def test_causal_visual_prefix_and_cached_gqa():
    mx.random.seed(7)
    config = TextConfig(
        d_model=32,
        decoder_attention_heads=4,
        self_attention_kv_heads=2,
        decoder_ffn_dim=64,
        decoder_layers=2,
        vocab_size=48,
    )
    model = LanguageModel(config)
    vision = mx.random.normal((1, 3, 32))
    prompt = mx.array([[2, 0, 7]])
    embeddings = mx.concatenate([vision, model.embed(prompt)], axis=1)
    full = model(inputs_embeds=embeddings).logits
    # A later visual token cannot change an earlier visual token's hidden state.
    changed = mx.concatenate(
        [vision[:, :2], vision[:, 2:] + 10, model.embed(prompt)], axis=1
    )
    np.testing.assert_allclose(
        np.asarray(full[:, :2]),
        np.asarray(model(inputs_embeds=changed).logits[:, :2]),
        atol=1e-5,
    )
    cache = model.make_cache()
    first = model(inputs_embeds=embeddings[:, :4], cache=cache).logits
    rest = model(inputs_embeds=embeddings[:, 4:], cache=cache).logits
    np.testing.assert_allclose(
        np.asarray(full), np.asarray(mx.concatenate([first, rest], axis=1)), atol=1e-5
    )
    assert cache[0].offset == 6
    assert cache[0].keys.shape[1] == 2


def test_packed_page_geometry_and_patch_order():
    image = Image.fromarray(np.arange(17 * 65 * 3, dtype=np.uint8).reshape(17, 65, 3))
    inputs = ImageProcessor()(image)
    assert inputs["image_sizes"].tolist() == [[32, 128]]
    assert inputs["content_sizes"].tolist() == [[17, 65]]
    assert inputs["pixel_values"].shape == (16, 768)
    expected = (
        np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255
        - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    ) / np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    np.testing.assert_array_equal(
        inputs["pixel_values"][0], expected[:, :16, :16].reshape(-1)
    )


def test_source_architecture_and_tied_head_mapping():
    config = {
        "model_type": "nemotron_parse",
        "architectures": ["RadioMBartPrefixForCausalLM"],
        "decoder": {"vocab_size": 72512, "self_attention_kv_heads": 8},
    }
    assert get_model_type(config) == "nemotron_parse_v3_5"
    assert ModelConfig.from_dict(config).text_config.vocab_size == 72512
    weights = Model.sanitize(
        {
            "decoder.embed_tokens.weight": mx.zeros((4, 2)),
            "decoder.extra_heads.0.weight": mx.ones((2, 2)),
            "lm_head.weight": mx.zeros((4, 2)),
        }
    )
    assert list(weights) == ["language_model.embed_tokens.weight"]
