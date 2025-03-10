from mlx_vlm import load, apply_chat_template, generate
import mlx.core as mx

# gg-hf-g/gemma-3-4b-it-pr
# to be updated!
model, processor = load("/Users/pedro/code/hf/apple/mlx/models/gemma-3-4b-it-new")

url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
messages = [
    {
        "role": "user", "content": [
            {"type": "text", "text": "What can you say about this image ?"},
            {"type": "image", "url": url},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="np",
)

input_ids = mx.array(inputs["input_ids"])
token_type_ids = mx.array(inputs["token_type_ids"])
pixel_values = mx.array(inputs["pixel_values"][0])
pixel_values = mx.expand_dims(pixel_values, 0)
mask = mx.array(inputs["attention_mask"])

kwargs = {}
kwargs["input_ids"] = input_ids
kwargs["token_type_ids"] = token_type_ids
kwargs["pixel_values"] = pixel_values
kwargs["mask"] = mask
kwargs["temp"] = 0
kwargs["max_tokens"] = 100

response = generate(
    model,
    processor,
    prompt="",
    verbose=True,
    **kwargs,
)
