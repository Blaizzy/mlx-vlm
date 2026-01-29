from typing import Optional

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.utils import load

from transformers.image_utils import load_image

model_id = "moonshotai/Kimi-K2.5"
prompt = "Describe this image"
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

def _broadcast_embeds(
    embeds: Optional[mx.array],
    input_ids: mx.array,
    model,
    group: mx.distributed.Group,
) -> mx.array:
    hidden_size = model.config.text_config.hidden_size
    dtype = model.language_model.model.embed_tokens.weight.dtype
    shape = (input_ids.shape[0], input_ids.shape[1], hidden_size)

    if embeds is None:
        embeds = mx.zeros(shape, dtype=dtype)

    gathered = mx.distributed.all_gather(embeds)
    batch = input_ids.shape[0]
    return gathered[:batch]

def _compute_embeddings(model, inputs):
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])
    pixel_values = mx.array(inputs["pixel_values"])
    grid_thws = mx.array(inputs["grid_thws"])
    grid_thws = grid_thws[:, -2:]

    embeddings, attention_mask, position_ids, expanded_ids = model.get_input_embeddings(input_ids, pixel_values, attention_mask, grid_thws)
    return embeddings, attention_mask, position_ids, expanded_ids


def main():
    group = mx.distributed.init()
    rank = group.rank()

    def rprint(*r_args, **r_kwargs):
        if rank == 0:
            print(*r_args, **r_kwargs)

    model, processor = load(model_id, lazy=True, trust_remote_code=True)
    model.language_model.shard(group)

    # Notes: the model uses a custom processor that does not yet work with transformers v5,
    # and is not compatible with mlx_vlm's prepare_inputs
    image = load_image(image_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "image_url": image},
            ]
        },
    ]
    inputs = processor(messages)

    input_ids = inputs.get("input_ids")
    mask = inputs.get("attention_mask", None)
    inputs_embeds = None
    if rank == 0:
        inputs_embeds, _, _, expanded_ids = _compute_embeddings(model, inputs)

    inputs_embeds = _broadcast_embeds(inputs_embeds, input_ids, model, group)

    for chunk in stream_generate(
        model,
        processor,
        "",
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        mask=mask,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        prefill_step_size=8000,
    ):
        rprint(chunk, end="", flush=True)

    rprint()


if __name__ == "__main__":
    main()
