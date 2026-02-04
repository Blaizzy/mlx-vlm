import mlx.core as mx
from transformers.image_utils import load_image

from mlx_vlm.generate import stream_generate
from mlx_vlm.utils import load

model_id = "moonshotai/Kimi-K2.5"
prompt = "Describe this image"
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"


def _compute_embeddings(model, inputs):
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])
    pixel_values = (
        mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
    )
    grid_thws = mx.array(inputs["grid_thws"])[:, -2:] if "grid_thws" in inputs else None

    embeddings, attention_mask, position_ids, expanded_ids = model.get_input_embeddings(
        input_ids, pixel_values, attention_mask, grid_thws
    )
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
            ],
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
    expanded_mask = None
    if pixel_values is None:
        inputs_embeds = model.language_model.embed_tokens(input_ids)
    else:
        # Multimodal embeddings and expanded sequences
        inputs_embeds, expanded_mask, _, input_ids = _compute_embeddings(model, inputs)

    for response in stream_generate(
        model,
        processor,
        "",
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        mask=expanded_mask,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        prefill_step_size=None,
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
