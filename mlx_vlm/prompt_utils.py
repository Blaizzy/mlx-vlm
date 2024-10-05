def get_message_json(
    model_name, prompt, role="user", skip_image_token=False, num_images=1
):
    """
    Get the appropriate JSON message based on the specified model.

    Args:
        model_name (str): The model for which to generate the message.
        prompt (str): The text prompt to be included in the message.
        role (str): The role of the message (default: "user").
        skip_image_token (bool): Whether to skip adding image tokens (default: False).
        num_images (int): Number of image tokens to add (default: 1).

    Returns:
        dict: A dictionary representing the JSON message for the specified model.
    """
    model_name = model_name.lower()

    def create_message(role, prompt):
        return {"role": role, "content": prompt}

    def add_image_tokens(message, token_format):
        if role == "user" and not skip_image_token:
            if isinstance(message["content"], list):
                message["content"].extend([{"type": "image"}] * num_images)
            else:
                if model_name == "phi3_v":
                    message["content"] = f"{token_format}{message['content']}"
                else:
                    message["content"] = (
                        f"{token_format * num_images}{message['content']}"
                    )
        return message

    message_formats = {
        "message_list_with_image": lambda: add_image_tokens(
            {"role": role, "content": [{"type": "text", "text": prompt}]}, ""
        ),
        "message_list_with_image_type": lambda: add_image_tokens(
            {"role": role, "content": [{"type": "text", "content": prompt}]}, ""
        ),
        "message_with_image_token": lambda: add_image_tokens(
            create_message(role, prompt), "<image>"
        ),
        "message_with_image_token_new_line": lambda: add_image_tokens(
            create_message(role, prompt), "<image>\n"
        ),
        "message_with_numbered_image_tokens": lambda: add_image_tokens(
            create_message(role, prompt),
            " ".join([f"<|image_{i+1}|>" for i in range(num_images)]),
        ),
        "prompt_only": lambda: prompt,
    }

    model_to_format = {
        "idefics2": "message_list_with_image",
        "qwen2_vl": "message_list_with_image",
        "llava": "message_list_with_image",
        "llava_next": "message_list_with_image",
        "llava-qwen2": "message_with_image_token_new_line",
        "bunny-llama": "message_with_image_token_new_line",
        "phi3_v": "message_with_numbered_image_tokens",
        "multi_modality": "message_with_image_token",
        "pixtral": "message_list_with_image_type",
        "paligemma": "prompt_only",
    }

    if num_images > 1 and model_name in [
        "llava_next",
        "llava-qwen2",
        "bunny-llama",
        "paligemma",
        "multi_modality",
    ]:
        raise ValueError(
            f"Model {model_name} does not support multi-image chat. Please only use 1 image."
        )

    format_key = model_to_format.get(model_name)

    if format_key:
        return message_formats[format_key]()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def apply_chat_template(
    processor,
    config,
    prompt,
    add_generation_prompt=True,
    return_messages=False,
    num_images=1,
):
    if not isinstance(config, dict):
        config = config.__dict__

    messages = []
    if isinstance(prompt, list):
        if isinstance(prompt[0], dict) and len(prompt) >= 1:
            for i, p in enumerate(prompt):
                if isinstance(p, str):
                    message = get_message_json(
                        config["model_type"],
                        p,
                        skip_image_token=i >= 1,
                        num_images=num_images,
                    )
                elif isinstance(p, dict) and "role" in p.keys():
                    message = get_message_json(
                        config["model_type"],
                        p["content"],
                        p["role"],
                        skip_image_token=i >= 1,
                        num_images=num_images,
                    )
                else:
                    raise ValueError("Invalid prompt type")
                messages.append(message)
        else:
            for prompts in prompt:
                for i, p in enumerate(prompts):
                    if isinstance(p, str):
                        message = get_message_json(
                            config["model_type"], p, skip_image_token=i >= 1
                        )
                    elif isinstance(p, dict) and "role" in p.keys():
                        message = get_message_json(
                            config["model_type"],
                            p["content"],
                            p["role"],
                            skip_image_token=i >= 1,
                            num_images=num_images,
                        )
                    else:
                        raise ValueError("Invalid prompt type")
                    messages.append(message)
    else:
        if isinstance(prompt, str):
            message = get_message_json(
                config["model_type"], prompt, num_images=num_images
            )
        elif isinstance(prompt, dict) and "role" in prompt.keys():
            message = get_message_json(
                config["model_type"],
                prompt["content"],
                prompt["role"],
                num_images=num_images,
            )
        else:
            raise ValueError("Invalid prompt type")
        messages.append(message)

    if return_messages:
        return messages

    if "chat_template" in processor.__dict__.keys():
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    elif "tokenizer" in processor.__dict__.keys():
        return processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    else:
        raise ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )
