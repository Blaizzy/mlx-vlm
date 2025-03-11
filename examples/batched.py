import mlx_vlm

# Load model
model, processor = mlx_vlm.load("mlx-community/llava-interleave-qwen-0.5b-bf16")

# Batch of prompts and images
prompts = [
    "Describe this image in detail.",
]
images = [
    "examples/images/cats.jpg",
]

# Generate responses for the batch
responses = mlx_vlm.batch_generate(
    model,
    processor,
    prompts=prompts,
    images=images,
    max_tokens=256,
    temperature=0.5,
    verbose=True
)

# Print responses
for i, response in enumerate(responses):
    print(f"Response {i+1}:\n{response}\n")
