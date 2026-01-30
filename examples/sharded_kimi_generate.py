from typing import Optional

import mlx.core as mx
from mlx_vlm.generate import stream_generate
from mlx_vlm.utils import load

from transformers.image_utils import load_image

model_id = "moonshotai/Kimi-K2.5"
prompt = "Describe this image"
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

def _broadcast_embeds(
    embeds: Optional[mx.array],
    batch: int,
    seq_len: int,
    model,
) -> mx.array:
    hidden_size = model.config.text_config.hidden_size
    dtype = model.language_model.model.embed_tokens.weight.dtype
    shape = (batch, seq_len, hidden_size)

    if embeds is None:
        embeds = mx.zeros(shape, dtype=dtype)

    gathered = mx.distributed.all_gather(embeds)
    return gathered[:batch]

def _broadcast_ids(
    expanded_ids: Optional[mx.array],
    input_ids: mx.array,
) -> mx.array:
    if expanded_ids is None:
        length = mx.array([0], mx.int32)
    else:
        length = mx.array([expanded_ids.shape[1]], mx.int32)

    lengths = mx.distributed.all_gather(length)
    seq_len = int(lengths[0].item())

    if expanded_ids is None:
        expanded_ids = mx.zeros((input_ids.shape[0], seq_len), dtype=input_ids.dtype)

    gathered = mx.distributed.all_gather(expanded_ids)
    batch = input_ids.shape[0]
    return gathered[:batch]

def _compute_embeddings(model, inputs):
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])
    pixel_values = mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
    grid_thws = mx.array(inputs["grid_thws"])[:, -2:] if "grid_thws" in inputs else None

    embeddings, attention_mask, position_ids, expanded_ids = model.get_input_embeddings(input_ids, pixel_values, attention_mask, grid_thws)
    return embeddings, attention_mask, position_ids, expanded_ids


def main():
    group = mx.distributed.init()
    rank = group.rank()

    def rprint(*r_args, **r_kwargs):
        if rank == 0:
            print(*r_args, **r_kwargs)

    model, processor = load(model_id, lazy=True, trust_remote_code=True)

    # Shard and ensure weights are materialized across ranks
    model.language_model.shard(group)
    mx.eval(model.language_model.parameters())
    print("rank", rank, "materialized", flush=True)

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
    has_image = any(
        item.get("type") == "image"
        for message in messages
        for item in message.get("content", [])
    )

    input_ids = mx.array(inputs.get("input_ids"))
    mask = inputs.get("attention_mask", None)
    if mask is not None:
        mask = mx.array(mask)
    pixel_values = inputs.get("pixel_values", None) if has_image else None
    inputs_embeds = None
    expanded_ids = None
    expanded_mask = None
    if pixel_values is None:
        # Text-only: broadcast input_ids from rank 0 and build embeddings there.
        expanded_ids = input_ids if rank == 0 else None
        input_ids = _broadcast_ids(expanded_ids, input_ids)
        if rank == 0:
            inputs_embeds = model.language_model.embed_tokens(input_ids)
        inputs_embeds = _broadcast_embeds(
            inputs_embeds, input_ids.shape[0], input_ids.shape[1], model
        )
        expanded_mask = None
    elif rank == 0:
        inputs_embeds, expanded_mask, _, expanded_ids = _compute_embeddings(model, inputs)

    if pixel_values is not None:
        input_ids = _broadcast_ids(expanded_ids, input_ids)
        inputs_embeds = _broadcast_embeds(
            inputs_embeds, input_ids.shape[0], input_ids.shape[1], model
        )
        if expanded_mask is None:
            expanded_mask = mx.zeros_like(input_ids)
        expanded_mask = _broadcast_ids(expanded_mask, input_ids)

    for response in stream_generate(
        model,
        processor,
        "",
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        mask=expanded_mask,
        max_tokens=10, #1024,
        temperature=0,
        top_p=1,
        # prefill_step_size=64,
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")


if __name__ == "__main__":
    main()
