def get_message_json(model_name, prompt, role="user", skip_image_token=False):
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
    if model_name.lower() in ["idefics2", "qwen2_vl", "llava", "llava_next"]:
        message = {
            "role": role,
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
        if role == "user" and not skip_image_token:
            message["content"].append({"type": "image"})
    elif model_name.lower() in ["llava-qwen2", "bunny-llama"]:

        message = {"role": role}
        if role == "user" and not skip_image_token:
            message["content"] = f"<image>\n{prompt}"
        else:
            message["content"] = prompt

    elif model_name.lower() == "phi3_v":
        message = {"role": role}
        if role == "user" and not skip_image_token:
            message["content"] = f"<|image_1|>\n{prompt}"
        else:
            message["content"] = prompt

    elif model_name.lower() == "multi_modality":
        message = {"role": role}
        if role == "user" and not skip_image_token:
            message["content"] = f"<image>{prompt}"
        else:
            message["content"] = prompt
    elif model_name.lower() == "pixtral":
        message = {"role": role, "content": prompt}

        if role == "user" and not skip_image_token:
            message["content"] = [
                {"type": "text", "content": prompt},
            ]
            message["content"].append({"type": "image"})
    elif model_name.lower() == "paligemma":
        message = prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return message


def apply_chat_template(
    processor, config, prompt, add_generation_prompt=True, return_messages=False
):
    messages = []
    if isinstance(prompt, list):
        if isinstance(prompt[0], dict) and len(prompt) >= 1:
            for i, p in enumerate(prompt):
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
                        )
                    else:
                        raise ValueError("Invalid prompt type")
                    messages.append(message)
    else:
        if isinstance(prompt, str):
            message = get_message_json(config["model_type"], prompt)
        elif isinstance(prompt, dict) and "role" in prompt.keys():
            message = get_message_json(
                config["model_type"], prompt["content"], prompt["role"]
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
