import mlx_vlm

# Load model
model, processor = mlx_vlm.load("mlx-community/Qwen2-VL-2B-Instruct-4bit")

# Batch of prompts and images
prompts = [
    "Describe this image in detail.",
    # "What's happening in this picture?",
    # "What can you tell me about this scene?"
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
