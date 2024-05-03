def get_message_json(model_name, prompt):
    """
    Get the appropriate JSON message based on the specified model.

    Args:
    model_name (str): The model for which to generate the message. Options: 'Idefics 2', 'nanollava', 'llava'.
    prompt (str): The text prompt to be included in the message.

    Returns:
    dict: A dictionary representing the JSON message for the specified model.
    """
    if model_name == "idefics2":
        message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    elif model_name in ["llava-qwen2", "llava"]:
        message = {"role": "user", "content": f"<image>\n{prompt}"}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return message
