from mlx_vlm import load, apply_chat_template, generate

# gg-hf-g/gemma-3-4b-it-pr
model, processor = load("/Users/pedro/code/hf/apple/mlx/models/gemma-3-4b-it-pr")

# This may still be a tokenizer
print(processor)
