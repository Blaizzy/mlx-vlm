import mlx_vlm

# Load model
model, processor = mlx_vlm.load("mlx-community/llava-interleave-qwen-0.5b-bf16")

# Set required processor attributes for LlavaProcessor
if hasattr(processor, 'tokenizer') and not hasattr(processor, 'patch_size') or processor.patch_size is None:
    # Default values for llava models
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

# Batch of prompts and images
prompts = [
    "Describe this image in detail.",
]
images = [
    "examples/images/cats.jpg",
]

# Generate responses for the batch
# responses = mlx_vlm.batch_generate(
#     model,
#     processor,
#     prompts=prompts,
#     images=images,
#     max_tokens=256,
#     temperature=0.5,
#     verbose=True
# )


response = mlx_vlm.generate(
    model,
    processor,
    prompt=prompts[0],
    image=images[0],
    max_tokens=256,
    temperature=0.5,
    verbose=True
)

print(response)
# # Print responses
# for i, response in enumerate(responses):
#     print(f"Response {i+1}:\n{response}\n")
