import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.models import oxr
from mlx_vlm.models.oxr.vision import bicubic_matrix

CASES = json.loads((Path(__file__).parent / "model_cases.json").read_text())
CONFIG = next(c for c in CASES["cases"] if c["id"] == "TestModels.oxr")["config"]


@pytest.fixture
def model():
    mx.random.seed(0)
    return oxr.Model(oxr.ModelConfig.from_dict(json.loads(json.dumps(CONFIG))))


def checkpoint_weights(model):
    """Build an original-layout OXR checkpoint from a module's parameters."""
    from mlx.utils import tree_flatten

    out = {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("vision_tower."):
            rest = key[len("vision_tower.") :].replace("layers.", "encoder.layers.", 1)
            if rest == "embeddings.patch_embedding.weight":
                c = model.config.vision_config
                value = value.reshape(value.shape[0], c.num_channels, c.patch_size, -1)
            key = "visual_tokenizer.vit.vision_model." + rest
        elif key.startswith("vision_head.layers."):
            key = "visual_tokenizer.head." + key[len("vision_head.layers.") :]
        elif key.startswith("vision_embed."):
            key = "vte." + key[len("vision_embed.") :]
        elif key.startswith("language_model.model."):
            key = "llm.model." + key[len("language_model.model.") :]
        out[key] = value
    width = model.config.vision_config.hidden_size
    out["visual_tokenizer.vit.vision_model.post_layernorm.weight"] = mx.ones((width,))
    out["llm.lm_head.weight"] = out["llm.model.embed_tokens.weight"]
    return out


def test_config_accepts_checkpoint_keys():
    params = json.loads(json.dumps(CONFIG))
    params["llm_config"] = params.pop("text_config")
    params["vit_config"] = params.pop("vision_config")
    config = oxr.ModelConfig.from_dict(params)
    assert config.text_config.model_type == "qwen3"
    assert config.vision_config.hidden_stride == 2


def test_sanitize_maps_checkpoint_and_is_idempotent(model):
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    once = model.sanitize(checkpoint_weights(model))
    assert set(once) == set(expected)  # post_layernorm / tied lm_head dropped
    for key, value in expected.items():
        assert once[key].shape == value.shape, key
        assert mx.array_equal(once[key], value).item(), key
    twice = model.sanitize(once)
    assert set(twice) == set(once)
    assert all(mx.array_equal(twice[k], once[k]).item() for k in once)


def test_bicubic_matrix_matches_identity_and_partition_of_unity():
    assert np.allclose(bicubic_matrix(8, 8), np.eye(8), atol=1e-6)
    m = bicubic_matrix(32, 45)
    assert m.shape == (45, 32)
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-5)


class _Tokenizer:
    """Minimal stand-in: maps special tokens and characters to ids."""

    model_input_names = ["input_ids", "attention_mask"]
    specials = {"<|image_pad|>": 60, "<|vision_start|>": 61, "<|vision_end|>": 62}

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        rows = []
        for text in texts:
            ids, i = [], 0
            while i < len(text):
                for tok, tid in self.specials.items():
                    if text.startswith(tok, i):
                        ids.append(tid)
                        i += len(tok)
                        break
                else:
                    ids.append(ord(text[i]) % 50)
                    i += 1
            rows.append(ids)
        return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]}


def test_processor_expands_image_placeholder():
    image_processor = oxr.OXRImageProcessor(
        patch_size=4, hidden_stride=2, min_pixels=16 * 16, max_pixels=32 * 32
    )
    processor = oxr.OXRProcessor.__new__(oxr.OXRProcessor)
    processor.image_processor = image_processor
    processor.tokenizer = _Tokenizer()
    processor.image_token = "<image>"
    processor.image_pad_token = "<|image_pad|>"
    processor.vision_start_token = "<|vision_start|>"
    processor.vision_end_token = "<|vision_end|>"

    image = Image.fromarray(np.random.randint(0, 255, (24, 40, 3), dtype=np.uint8))
    out = processor(images=[image], text="ab<image>cd")
    grid = out["image_grid_thw"][0]
    n_atoms = int(np.prod(grid)) // 4
    ids = out["input_ids"][0]
    assert ids.count(60) == n_atoms
    assert ids[2] == 61 and ids[3 + n_atoms] == 62
    assert out["pixel_values"].shape == (int(np.prod(grid)), 3 * 4 * 4)
    assert grid[1] % 2 == 0 and grid[2] % 2 == 0


def test_input_embeddings_splice_visual_tokens_and_indicators(model):
    cfg = model.config
    grid = mx.array([[1, 4, 4]])
    pixels = mx.random.normal((16, 48))
    n_atoms = 16 // 4
    ids = mx.array(
        [
            [1, cfg.vision_start_token_id]
            + [cfg.image_token_id] * n_atoms
            + [cfg.vision_end_token_id, 2]
        ]
    )
    emb = model.get_input_embeddings(ids, pixels, image_grid_thw=grid).inputs_embeds
    text = model.language_model.model.embed_tokens(ids)
    visual = model.encode_images(pixels, grid)
    vv = cfg.visual_vocab_size
    assert mx.allclose(emb[0, 0], text[0, 0]).item()
    assert mx.allclose(emb[0, -1], text[0, -1]).item()
    assert mx.allclose(emb[0, 1], model.vision_embed.weight[vv - 4]).item()
    assert mx.allclose(emb[0, 2 + n_atoms], model.vision_embed.weight[vv - 3]).item()
    assert mx.allclose(emb[0, 2 : 2 + n_atoms], visual, atol=1e-5).item()

    with pytest.raises(ValueError):
        model.get_input_embeddings(ids[:, :-3], pixels, image_grid_thw=grid)
