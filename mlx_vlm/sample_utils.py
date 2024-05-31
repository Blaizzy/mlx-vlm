"""

This module `sample_utils` provides utility functions related to the processing and sampling of probability distributions typically used in machine learning models for generating content, such as text or images. It particularly contains a function that implements the top-p sampling strategy, which is an advanced technique for sampling from a categorical distribution, often used in natural language processing (NLP) to generate human-like text by probabilistic models.

The `top_p_sampling` function is designed to operate on an array of logits that represent the unnormalized log probabilities of different events (such as the next word in a sequence). The logits are first converted to probabilities through a softmax function, optionally using a temperature parameter to soften or sharpen the distribution. The resulting probabilities are then processed to retain only those outcomes that cumulatively compose the top p portion of the distribution, effectively truncating the tail of low-probability events. Random sampling is then performed on this adjusted probability distribution to select an outcome, which helps ensure diversity in the generated content while still maintaining coherence by favoring more likely possibilities.

The function handles specific details such as sorting probabilities, accumulating them to form the cumulative distribution, and selecting the top probabilities based on the provided `top_p` threshold. Edge cases, such as support for different data types (`mx.bfloat16`), are also taken into consideration, with conversions applied as needed to ensure compatibility and stability of operations.

Overall, the utility function provided in this module encapsulates the complexity of the top-p sampling method, making it accessible to other parts of the code that require stochastic content generation based on model logits.
"""

import mlx.core as mx


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Performs top-p sampling on logits using a specified threshold and temperature.
    Top-p sampling (nucleus sampling) is a technique used to generate a probability distribution
    for selecting tokens, where the sampling is restricted to the tokens with a cumulative
    probability less than or equal to the specified top_p threshold, after scaling the logits
    by the temperature.

    Args:
        logits (mx.array):
             Pre-softmax logits from a language model. The logits are assumed
            to be of shape (1, vocabulary_size).
        top_p (float):
             The cumulative probability threshold used for the top-p sampling.
            Only tokens with a cumulative probability that is less than or equal to the top_p are
            considered in sampling. This value must be between 0 and 1.
        temperature (float):
             A temperature hyperparameter for scaling the logits. A higher
            temperature leads to more random sampling and a lower temperature results in more
            deterministic sampling behavior.

    Returns:
        (mx.array):
             The sampled token as an array with a single element.

    Raises:
        TypeError:
             If logits are not of the expected data type.

    Note:
        This implementation was referenced from the Huggingface transformers library
        (see provided GitHub link in the original code for the specific implementation).

    """
    if (
        logits.dtype == mx.bfloat16
    ):  # workaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
        logits = logits.astype(mx.float32)

    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token
