"""

Module `prompt_utils` Overview

The module `prompt_utils` provides utility functions for generating message dictionaries in JSON format which are tailored for different artificial intelligence (AI) models. Each AI model may require the data to be structured in a unique way that is compatible with its processing requirements.

### Function: `get_message_to_json`

This function takes in two parameters: `model name` and `prompt`. Based on the `model_name` provided, it crafts a JSON message structure suitable for the specific AI model. The `prompt` parameter contains the actual content or message to be sent to the model.

The function supports multiple models, each with a distinct way of handling the input prompt:

- `idefics2`: For this model, the message format includes a role, content type as 'image', and the text of the prompt.
- `llava-qwen2` and `llava`: These models expect the content to be a string with an inline image indicator followed by the prompt text.
- `paligemma`: This model requires the prompt as a plain string without any additional formatting or surrounding metadata.

An exception is thrown if the provided `model_name` is not supported, indicating that the user has specified an unsupported or unknown AI model.

The function returns the formatted message dictionary, which can then be used by the corresponding AI model's processing system.
"""


def get_message_json(model_name, prompt):
    """
    Constructs a JSON message based on the specified model name and prompt.

    Args:
        model_name (str):
             The name of the model to construct the message for.
        prompt (str):
             The text or prompt to include in the message.

    Returns:
        (dict):
             A JSON object representing the message for the idefics2 or llava-qwen2/llava models.
        (str):
             A raw prompt string for the paligemma model.

    Raises:
        ValueError:
             If an unsupported model name is provided.

    """
    if model_name == "idefics2":
        message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    elif model_name in ["llava-qwen2", "llava"]:
        message = {"role": "user", "content": f"<image>\n{prompt}"}
    elif model_name == "paligemma":
        message = prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return message
