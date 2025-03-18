from functools import partial


def get_message_json(
    model_name, prompt, role="user", skip_image_token=False, num_images=1, **kwargs
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

    # Base message creation
    def create_text_message(text):
        return {"type": "text", "text": text}

    def create_text_content_message(text):
        return {"type": "text", "content": text}

    def create_video_message(video_path, max_pixels=224 * 224, fps=1):
        return {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
            "fps": fps,
        }

    # Message format handlers
    def handle_list_with_image(image_first=False):
        content = [create_text_message(prompt)]
        if role == "user" and not skip_image_token:
            image_tokens = [{"type": "image"}] * num_images
            content = image_tokens + content if image_first else content + image_tokens
        return {"role": role, "content": content}

    def handle_list_with_image_type():
        message = {"role": role, "content": [create_text_content_message(prompt)]}
        if role == "user" and not skip_image_token:
            message["content"] = [{"type": "image"}] * num_images + message["content"]
        if role == "assistant":
            message["content"] = message["content"][0]["content"]
        return message

    def handle_image_token(token_format, image_first=True):
        content = prompt
        if role != "system" and role == "user" and not skip_image_token:

            prefix = (
                token_format * num_images if model_name != "phi3_v" else token_format
            )
            if image_first:
                content = f"{prefix}{content}"
            else:
                content = f"{content}{prefix}"
        return {"role": role, "content": content}

    def handle_video_with_text():
        return {
            "role": "user",
            "content": [
                create_video_message(
                    kwargs["video"],
                    kwargs.get("max_pixels", 224 * 224),
                    kwargs.get("fps", 1),
                ),
                create_text_message(prompt),
            ],
        }

    # Message format mapping
    message_formats = {
        "message_list_with_image": handle_list_with_image,
        "message_list_with_image_first": partial(
            handle_list_with_image, image_first=True
        ),
        "message_list_with_image_type": handle_list_with_image_type,
        "message_with_image_token": lambda: handle_image_token("<image>"),
        "message_with_start_image_token": lambda: handle_image_token(
            "<start_of_image>", image_first=False
        ),
        "message_with_image_token_new_line": lambda: handle_image_token("<image>\n"),
        "message_with_numbered_image_tokens": lambda: handle_image_token(
            " ".join([f"<|image_{i+1}|>" for i in range(num_images)])
        ),
        "prompt_only": lambda: prompt,
        "prompt_with_image_token": lambda: "<image>" * num_images + prompt,
        "prompt_with_start_image_token": lambda: prompt
        + "<start_of_image>" * num_images,
        "message_video_with_text": handle_video_with_text,
    }

    # Model to format mapping
    model_to_format = {
        # Models using message_list_with_image format
        "idefics2": "message_list_with_image",
        "idefics3": "message_list_with_image_first",
        "aya_vision": "message_list_with_image_first",
        "gemma3": "message_with_start_image_token",
        "smolvlm": "message_list_with_image_first",
        "llava": "message_list_with_image",
        "llava_next": "message_list_with_image",
        "mllama": "message_list_with_image",
        # Models that can handle both image and video formats
        "qwen2_vl": (
            "message_video_with_text"
            if kwargs.get("video")
            else "message_list_with_image"
        ),
        "qwen2_5_vl": (
            "message_video_with_text"
            if kwargs.get("video")
            else "message_list_with_image"
        ),
        # Models using message_with_image_token_new_line format
        "llava-qwen2": "message_with_image_token_new_line",
        "bunny-llama": "message_with_image_token_new_line",
        # Models using message_with_numbered_image_tokens format
        "phi3_v": "message_with_numbered_image_tokens",
        # Models using message_with_image_token format
        "multi_modality": "message_with_image_token",
        "deepseek_vl_v2": "message_with_image_token_new_line",
        # Models using message_list_with_image_type format
        "pixtral": "message_list_with_image_type",
        # Models using prompt_with_image_token format
        "paligemma": "prompt_with_image_token",
        # Models using prompt_only format
        "florence2": "prompt_only",
        "molmo": "prompt_only",
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


def get_chat_template(
    processor, messages, add_generation_prompt, tokenize=False, **kwargs
):
    if "chat_template" in processor.__dict__.keys():
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    elif "tokenizer" in processor.__dict__.keys():
        return processor.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    else:
        raise ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )


def apply_chat_template(
    processor,
    config,
    prompt,
    add_generation_prompt=True,
    return_messages=False,
    num_images=1,
    **kwargs,
):

    config = config if isinstance(config, dict) else config.__dict__

    def process_single_prompt(p, is_first=True):
        if isinstance(p, str):
            return get_message_json(
                config["model_type"],
                p,
                skip_image_token=not is_first,
                num_images=num_images,
                **kwargs,
            )
        elif isinstance(p, dict) and "role" in p:
            return get_message_json(
                config["model_type"],
                p["content"],
                p["role"],
                skip_image_token=not is_first,
                num_images=num_images,
                **kwargs,
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

    return get_chat_template(processor, messages, add_generation_prompt)
