import mlx.core as mx
from mlx_vlm import load, stream_generate, generate
from mlx_vlm.video_generate import process_vision_info
from outlines.processors import JSONLogitsProcessor
from outlines.models.transformers import TransformerTokenizer

# need install outlines with version 1.1.1, uv pip install outlines==1.1.1

image_path = "examples/images/password.jpg"

# Load model and processor
model, processor = load("mlx-community/Qwen3-VL-2B-Thinking-8bit")

# Define JSON schema for structured output
json_schema = {
    "properties": {
        "username": {
            "type": "string",
            "description": "The username of the account"
        },
        "password": {
            "type": "string",
            "description": "The password of the account"
        }
    }
}

# Setup outlines processor for JSON schema enforcement
outlines_tokenizer = TransformerTokenizer(processor.tokenizer)
json_logits_processor = JSONLogitsProcessor(
    schema=json_schema,
    tokenizer=outlines_tokenizer,
    tensor_library_name="mlx"
)

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Extract the wifi internet service provider information (including username and password) from the image"
            },
            {
                "type": "image",
                "image": image_path
            }
        ]
    }
]

# Process vision inputs and generate prompt
image_inputs, video_inputs = process_vision_info(messages)
input_prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Process inputs - try MLX first, fallback to PyTorch if needed
try:
    inputs = processor(
        text=[input_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="mlx"
    )
    # MLX arrays are already in the correct format
except (TypeError, AttributeError, ValueError):
    # Fallback to PyTorch and convert to MLX
    inputs = processor(
        text=[input_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    # Convert PyTorch tensors to MLX arrays efficiently
    inputs = {k: mx.array(v) if not isinstance(v, (str, list, mx.array)) else v 
              for k, v in inputs.items()}

# Add JSON logits processor to inputs
inputs["json_logits_processor"] = json_logits_processor

# Test with generate
print("=" * 50)
print("Testing generate()")
print("=" * 50)
response = generate(
    model,
    processor,
    prompt=input_prompt,
    **inputs
)
print("RESPONSE:", response)

# Test with streaming generate
print("\n" + "=" * 50)
print("Testing stream_generate()")
print("=" * 50)
response_generator = stream_generate(
    model,
    processor,
    prompt=input_prompt,
    **inputs
)

# Use list for efficient string building
chunks = []
for chunk in response_generator:
    if chunk and chunk.text:
        chunks.append(chunk.text)

final_text = "".join(chunks)
print("FINAL TEXT:", final_text)