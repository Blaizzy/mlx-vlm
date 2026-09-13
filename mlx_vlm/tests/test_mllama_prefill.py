"""Mllama prefill regression tests using small, randomly initialized weights."""

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar
from mlx_vlm.models import mllama


@pytest.fixture
def model():
    mx.random.seed(42)
    model = mllama.Model(
        mllama.ModelConfig(
            model_type="mllama",
            text_config=mllama.TextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                cross_attention_layers=[1],
            ),
            vision_config=mllama.VisionConfig(
                image_size=4,
                patch_size=2,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_global_layers=1,
                num_attention_heads=2,
                max_num_tiles=1,
                vision_output_dim=16,
                intermediate_layers_indices=[0],
            ),
        )
    )
    # Zero-initialized cross-attention gates would hide image/mask errors.
    cross_layer = model.language_model.model.layers[1]
    cross_layer.cross_attn_attn_gate = mx.ones((1,))
    cross_layer.cross_attn_mlp_gate = mx.ones((1,))
    return model


@pytest.mark.parametrize("prefill_step_size", [1, 2, 2048])
@pytest.mark.parametrize("num_images", [1, 2])
def test_mllama_chunked_prefill_matches_full_prompt(
    model, prefill_step_size, num_images
):
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    pixels = mx.random.normal((1, num_images, 1, 3, 4, 4))
    # Text before an image cannot attend to it. A second image becomes visible
    # later in the prompt, so chunks must retain their own mask rows.
    cross_mask = mx.array(
        [[[[int(t >= 1 + 3 * image)] for image in range(num_images)] for t in range(7)]]
    )
    kwargs = dict(
        input_ids=input_ids,
        model=model,
        pixel_values=pixels,
        mask=None,
        max_tokens=3,
        temperature=0,
        aspect_ratio_ids=mx.ones((1, num_images), dtype=mx.int32),
        aspect_ratio_mask=mx.ones((1, num_images, 1)),
        cross_attention_mask=cross_mask,
    )
    expected = list(ar.generate_step(**kwargs, prefill_step_size=None))
    actual = list(ar.generate_step(**kwargs, prefill_step_size=prefill_step_size))
    assert len(actual) == len(expected) == 3
    for (token, logprobs), (expected_token, expected_logprobs) in zip(actual, expected):
        assert token == expected_token
        assert mx.allclose(logprobs, expected_logprobs, atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("prefill_step_size", [1, 2, 2048])
def test_mllama_ragged_prefill_matches_individual_prompts(model, prefill_step_size):
    prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]]
    rows = []
    expected = []
    for prompt in prompts:
        ids = mx.array([prompt])
        features = model.get_input_embeddings(
            input_ids=ids,
            pixel_values=mx.random.normal((1, 1, 1, 3, 4, 4)),
            aspect_ratio_ids=mx.ones((1, 1), dtype=mx.int32),
            aspect_ratio_mask=mx.ones((1, 1, 1)),
            cross_attention_mask=mx.array(
                [[[[int(t > 0)]] for t in range(len(prompt))]]
            ),
        ).to_dict()
        logits = model.language_model(inputs=ids, **features).logits[:, -1, :]
        expected.append(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        rows.append(features)

    embeds, kwargs = ar._merge_prefill_prompt_kwargs(rows, prompts)
    batch = ar.PromptProcessingBatch(
        model=model.language_model,
        uids=[0, 1],
        input_ids=prompts,
        max_tokens=[1, 1],
        inputs_embeds=embeds,
        prompt_kwargs=kwargs,
        prefill_step_size=prefill_step_size,
    )
    while batch.needs_processing():
        batch.prompt_step()
    sampled = []

    def sampler(logprobs):
        sampled.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    batch.generate(sampler=sampler, stop_criteria=lambda _: False)
    assert mx.allclose(
        sampled[0], mx.concatenate(expected), atol=1e-5, rtol=1e-5
    ).item()
