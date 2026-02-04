import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "n-Arno/joycaption-mlx-mxfp4"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
# image = [Image.open("...")] can also be used with PIL.Image.Image objects
prompt = "Write a long descriptive caption for this image in a formal tone."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
