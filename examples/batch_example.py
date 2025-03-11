import mlx_vlm
from mlx_vlm.utils import load_image
import sys

images = [
    "examples/images/cats.jpg",
    "examples/images/desktop_setup.png",
    "examples/images/latex.png"
]

prompts = [
    "What animals do you see in this image?",
    "Describe this workspace setup.",
    "What kind of document is shown in this image?"
]

try:
    model, processor = mlx_vlm.load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    
    responses = mlx_vlm.batch_generate(
        model,
        processor,
        prompts=prompts,
        images=images,
        max_tokens=500,
        temperature=0.7,
        batch_size=2,
        verbose=True
    )
    
    for i, response in enumerate(responses):
        print(f"\nImage: {images[i]}")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {response}")

except KeyboardInterrupt:
    sys.exit(1)
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 
