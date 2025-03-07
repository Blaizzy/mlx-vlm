from mlx_vlm import load, apply_chat_template, generate

model, processor = load("/Users/pedro/code/hf/apple/mlx/models/gemma-3-4b-it")

# This may still be a tokenizer
print(processor)
