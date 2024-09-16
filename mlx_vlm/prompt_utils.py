def get_message_json(model_name, prompt):
    """
    Get the appropriate JSON message based on the specified model.

    Args:
        model_name (str): The model for which to generate the message.
        prompt (str): The text prompt to be included in the message.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: A dictionary representing the JSON message for the specified model.
    """
    if model_name.lower() in ["idefics2", "qwen2_vl"]:
        message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }

    elif model_name.lower() in ["llava-qwen2", "llava", "llava_next", "bunny-llama"]:
        message = {"role": "user", "content": f"<image>\n{prompt}"}
    elif model_name.lower() == "phi3_v":
        message = {"role": "user", "content": f"<|image_1|>\n{prompt}"}
    elif model_name.lower() == "multi_modality":
        message = {"role": "user", "content": f"<image>{prompt}"}
    elif model_name.lower() == "paligemma":
        message = prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return message


def apply_chat_template(processor, config, prompt):
    message = get_message_json(config["model_type"], prompt)

    if "chat_template" in processor.__dict__.keys():
        return processor.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
        )

    elif "tokenizer" in processor.__dict__.keys():
        return processor.tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
        )

    else:
        raise ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )
