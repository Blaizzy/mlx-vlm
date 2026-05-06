import mlx.core as mx


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output. Shape [vocab] or [B, vocab].
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion. Shape [] or [B].
    """
    unbatched = logits.ndim == 1
    if unbatched:
        logits = logits[None]

    if (
        logits.dtype == mx.bfloat16
    ):  # workaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
        logits = logits.astype(mx.float32)

    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sampled_pos = mx.random.categorical(mx.log(top_probs))
    token = mx.take_along_axis(sorted_indices, sampled_pos[:, None], axis=-1).squeeze(
        -1
    )
    return token.squeeze(0) if unbatched else token
