from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template


def inference(
    model,
    processor,
    question,
    image,
    max_tokens=3000,
    temperature=0.0,
    resize_shape=None,
    verbose=False,
):
    """Run inference on a single question."""
    if image is None:
        num_images = 0
    elif isinstance(image, list):
        num_images = len(image)
    else:
        num_images = 1

    prompt = apply_chat_template(
        processor, model.config, question, num_images=num_images
    )

    response = generate(
        model,
        processor,
        prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=temperature,
        resize_shape=resize_shape,
        verbose=verbose,
    )
    return response.text
