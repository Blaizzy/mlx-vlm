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
        if role == "system":
            return message
        if role == "user" and not skip_image_token:
            if isinstance(message["content"], list):
                if model_name in ["pixtral", "idefics3"]:
                    message["content"] = [{"type": "image"}] * num_images + message[
                        "content"
                    ]
                else:
                    message["content"].extend([{"type": "image"}] * num_images)
            else:
                if model_name == "phi3_v":
                    message["content"] = f"{token_format}{message['content']}"
                else:
                    message["content"] = (
                        f"{token_format * num_images}{message['content']}"
                    )
        if role == "assistant" and model_name == "pixtral":
            message["content"] = message["content"][0]["content"]
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
        "prompt_with_image_token": lambda: "<image>" * num_images + prompt,
    }

    model_to_format = {
        "idefics2": "message_list_with_image",
        "idefics3": "message_list_with_image",
        "qwen2_vl": "message_list_with_image",
        "llava": "message_list_with_image",
        "llava_next": "message_list_with_image",
        "llava-qwen2": "message_with_image_token_new_line",
        "bunny-llama": "message_with_image_token_new_line",
        "phi3_v": "message_with_numbered_image_tokens",
        "multi_modality": "message_with_image_token",
        "pixtral": "message_list_with_image_type",
        "paligemma": "prompt_with_image_token",
        "florence2": "prompt_only",
        "mllama": "message_list_with_image",
        "molmo": "prompt_only",
        "deepseek_vl_v2": "message_with_image_token_new_line",
    }

    if num_images > 1 and model_name in [
        "llava_next",
        "llava-qwen2",
        "bunny-llama",
        "paligemma",
        "multi_modality",
        "mllama",
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
    config = config if isinstance(config, dict) else config.__dict__

    def process_single_prompt(p, is_first=True):
        if isinstance(p, str):
            return get_message_json(
                config["model_type"],
                p,
                skip_image_token=not is_first,
                num_images=num_images,
            )
        elif isinstance(p, dict) and "role" in p:
            return get_message_json(
                config["model_type"],
                p["content"],
                p["role"],
                skip_image_token=not is_first,
                num_images=num_images,
            )
        else:
            raise ValueError("Invalid prompt type")

    messages = []
    if isinstance(prompt, list):
        if isinstance(prompt[0], dict):
            for i, p in enumerate(prompt):
                if p.get("role") in ["system", "assistant"]:
                    messages.append(process_single_prompt(p, False))
                else:
                    is_first = i == 0 or i == 1
                    messages.append(process_single_prompt(p, is_first))
        else:
            for prompts in prompt:
                for i, p in enumerate(prompts):
                    if p.get("role") in ["system", "assistant"]:
                        messages.append(process_single_prompt(p, False))
                    else:
                        is_first = i == 0 or i == 1
                        messages.append(process_single_prompt(p, is_first))
    else:
        messages = [process_single_prompt(prompt)]

    if return_messages:
        return messages

    if config["model_type"] in ["paligemma", "molmo", "florence2"]:
        return messages[-1]

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
