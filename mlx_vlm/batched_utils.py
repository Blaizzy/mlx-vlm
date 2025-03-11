
import time
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .models.base import BatchedKVCache
from .prompt_utils import apply_chat_template, get_message_json
from .sample_utils import top_p_sampling
from .utils import (
    GenerationResult,
    load_image,
    prepare_inputs,
    process_image,
    apply_repetition_penalty
)

def create_detokenizer(processor):
    """Create a new detokenizer instance compatible with the given processor."""
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    if hasattr(processor, "detokenizer"):
        detokenizer_class = type(processor.detokenizer)
        return detokenizer_class(tokenizer)
    else:
        raise ValueError("Processor does not have a detokenizer attribute")

class BatchedGenerationResult:
    """Container for batched generation results."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.texts = [""] * batch_size
        self.tokens = [None] * batch_size
        self.logprobs = [None] * batch_size
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.prompt_tps = 0.0
        self.generation_tps = 0.0
        self.peak_memory = 0.0

    def get_result(self, index: int) -> GenerationResult:
        """Get individual result for a batch index."""
        return GenerationResult(
            text=self.texts[index],
            token=self.tokens[index],
            logprobs=self.logprobs[index],
            prompt_tokens=self.prompt_tokens,
            generation_tokens=self.generation_tokens,
            prompt_tps=self.prompt_tps,
            generation_tps=self.generation_tps,
            peak_memory=self.peak_memory
        )

    def update_segments(self, indices: List[int], segments: List[str]):
        """Update text segments for specific batch indices."""
        for idx, segment in zip(indices, segments):
            if idx < self.batch_size:
                self.texts[idx] += segment


def batch_generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values: mx.array,
    mask: mx.array,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    **kwargs,
):
    """
    Generate tokens for a batch of inputs.

    Args:
        input_ids: Batch of input token IDs [batch_size, seq_len]
        model: The model to use for generation
        pixel_values: Batch of image features [batch_size, ...]
        mask: Attention mask [batch_size, seq_len]
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeating tokens
        repetition_context_size: Size of context window for repetition penalty
        top_p: Top-p (nucleus) sampling parameter
        logit_bias: Optional dictionary mapping token IDs to bias values

    Yields:
        Tuple of token IDs and log probabilities for each step
    """
    batch_size = input_ids.shape[0]

    def sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        if temperature == 0:
            tokens = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                tokens = mx.concatenate([
                    top_p_sampling(logits[i:i+1], top_p, temperature)
                    for i in range(batch_size)
                ])
            else:
                tokens = mx.random.categorical(logits * (1 / temperature))

        return tokens, mx.take_along_axis(logprobs, mx.expand_dims(tokens, axis=-1), axis=-1).squeeze(-1)

    # Set up KV cache for the batch
    if hasattr(model.language_model, "make_cache"):
        cache = model.language_model.make_batch_cache(batch_size)
    else:
        kv_heads = (
            [model.language_model.n_kv_heads] * len(model.language_model.layers)
            if isinstance(model.language_model.n_kv_heads, int)
            else model.language_model.n_kv_heads
        )

        if model.config.model_type == "florence2":
            raise NotImplementedError("Batched generation not yet supported for Florence2 model")
        else:
            cache = [BatchedKVCache(model.language_model.head_dim, n, batch_size) for n in kv_heads]

    # Track repetition context for each item in the batch if needed
    repetition_contexts = None
    if repetition_penalty:
        repetition_contexts = [input_ids[i].tolist() for i in range(batch_size)]
        if repetition_context_size:
            repetition_contexts = [
                ctx[-repetition_context_size:] for ctx in repetition_contexts
            ]

    def _step(y, **kwargs):
        nonlocal repetition_contexts

        if "decoder_input_ids" in kwargs:
            outputs = model.language_model(
                cache=cache,
                **kwargs,
            )
        else:
            outputs = model.language_model(
                y,
                cache=cache,
                mask=mask,
                **kwargs,
            )

        logits = outputs.logits[:, -1, :]

        if repetition_penalty:
            # Apply repetition penalty for each sequence in the batch
            for i in range(batch_size):
                logits[i:i+1] = apply_repetition_penalty(
                    logits[i:i+1], repetition_contexts[i], repetition_penalty
                )

        y, logprobs = sample(logits)

        # Update repetition contexts
        if repetition_penalty:
            for i in range(batch_size):
                repetition_contexts[i].append(y[i].item())
                if repetition_context_size and len(repetition_contexts[i]) > repetition_context_size:
                    repetition_contexts[i] = repetition_contexts[i][-repetition_context_size:]

        return y, logprobs

    # Initial forward pass
    outputs = model(input_ids, pixel_values, cache=cache, mask=mask, **kwargs)
    logits = outputs.logits[:, -1, :]
    y, logprobs = sample(logits)
    mx.async_eval(y)

    # Set up kwargs for next token prediction
    if outputs.cross_attention_states is not None:
        kwargs = {
            "cross_attention_states": outputs.cross_attention_states
        }
    elif outputs.encoder_outputs is not None:
        kwargs = {
            "decoder_input_ids": y[:, None],
            "encoder_outputs": outputs.encoder_outputs,
        }
    else:
        kwargs = {}

    # Generate tokens
    n = 0
    while n < max_tokens:
        next_y, next_logprobs = _step(y[:, None], **kwargs)
        mx.async_eval(next_y)

        if "decoder_input_ids" in kwargs:
            kwargs["decoder_input_ids"] = next_y[:, None]

        yield y, logprobs
        y, logprobs = next_y, next_logprobs
        n += 1


def batch_stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompts: List[str],
    images: Optional[List[Union[str, List[str]]]] = None,
    **kwargs
):
    """
    Stream generation for a batch of inputs.

    Args:
        model: The vision-language model
        processor: The tokenizer/processor
        prompts: List of text prompts
        images: List of image paths or image objects
        **kwargs: Additional generation arguments

    Yields:
        BatchedGenerationResult for each generation step
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)

    detokenizers = [create_detokenizer(processor) for _ in range(batch_size)]
    for detokenizer in detokenizers:
        detokenizer.reset()

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)
    max_tokens = kwargs.get("max_tokens", 256)

    # Check if pixel values are provided directly
    if kwargs.get("pixel_values") is None:
        if not images:
            # Text-only mode
            encodings = tokenizer(prompts, padding=True, return_tensors="np")
            input_ids = mx.array(encodings["input_ids"])
            pixel_values = None
            mask = mx.array(encodings["attention_mask"]) if "attention_mask" in encodings else None
        else:
            # Ensure images is a list of lists (for multi-image support)
            if len(images) != batch_size:
                raise ValueError(f"Expected {batch_size} images, got {len(images)}")

            # Format input images correctly
            if not all(isinstance(img, list) for img in images) and all(isinstance(img, str) or not isinstance(img, list) for img in images):
                # Convert single images to lists
                images = [[img] if not isinstance(img, list) else img for img in images]

            # Prepare inputs for each prompt/image pair
            all_inputs = []
            for i in range(batch_size):
                inputs = prepare_inputs(
                    processor,
                    images[i],
                    prompts[i],
                    image_token_index,
                    resize_shape
                )
                all_inputs.append(inputs)

            # Combine into batched inputs
            batch_inputs = {}
            for key in all_inputs[0].keys():
                batch_inputs[key] = mx.stack([inp[key] for inp in all_inputs])

            input_ids = batch_inputs["input_ids"]
            pixel_values = batch_inputs["pixel_values"]
            mask = batch_inputs["attention_mask"]

            # Add additional kwargs
            data_kwargs = {
                k: v
                for k, v in batch_inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }
            kwargs.update(data_kwargs)
    else:
        # Use provided inputs
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask", None)

    # Start generation
    tic = time.perf_counter()
    result = BatchedGenerationResult(batch_size)

    active_indices = list(range(batch_size))
    completed_tokens = [False] * batch_size

    for n, (tokens, logprobs) in enumerate(
        batch_generate_step(input_ids, model, pixel_values, mask, **kwargs)
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            prompt_tps = input_ids.size / prompt_time
            result.prompt_tokens = input_ids.shape[1]
            result.prompt_tps = prompt_tps
            tic = time.perf_counter()

        # Process each token in the batch
        still_active = []
        for i, token in enumerate(tokens.tolist()):
            if i in active_indices and not completed_tokens[i]:
                batch_idx = active_indices[i]

                # Check for EOS token
                if token == tokenizer.eos_token_id:
                    completed_tokens[batch_idx] = True
                else:
                    # Add token to detokenizer
                    detokenizers[batch_idx].add_token(token)
                    result.texts[batch_idx] = detokenizers[batch_idx].text
                    result.tokens[batch_idx] = token
                    result.logprobs[batch_idx] = logprobs[i].tolist()
                    still_active.append(batch_idx)

        active_indices = still_active
        result.generation_tokens = n + 1
        result.generation_tps = (n + 1) / (time.perf_counter() - tic)
        result.peak_memory = mx.metal.get_peak_memory() / 1e9 if hasattr(mx, "metal") else 0

        # Yield intermediate results
        yield result

        # Break if all sequences completed
        if not active_indices:
            break

    # Finalize all detokenizers
    for i, detokenizer in enumerate(detokenizers):
        detokenizer.finalize()
        result.texts[i] = detokenizer.text

    yield result


def batch_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompts: Union[str, List[str]],
    images: Optional[Union[str, List[str], List[List[str]]]] = None,
    verbose: bool = False,
    format_prompts: bool = True,
    **kwargs
) -> Union[str, List[str]]:
    """
    Generate text from a batched model.

    Args:
        model: The vision-language model
        processor: The tokenizer/processor
        prompts: A single prompt string or list of prompt strings
        images: A single image path or list of image paths or list of lists of image paths
        verbose: Whether to print generation details
        format_prompts: Whether to apply chat template formatting to prompts
        **kwargs: Additional generation arguments

    Returns:
        List of generated text responses
    """
    # Format inputs to lists
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = len(prompts)

    # Format images consistently
    if images is not None:
        if isinstance(images, str):
            # Single image -> make a list of the same image for each prompt
            images = [[images]] * batch_size
        elif isinstance(images, list):
            if len(images) == 1 or (isinstance(images[0], str) and not any(isinstance(img, list) for img in images)):
                # List of image paths -> make a list of lists with one image per prompt
                images = [[img] for img in images]
                if len(images) < batch_size:
                    # Duplicate last image for remaining prompts if needed
                    images.extend([images[-1]] * (batch_size - len(images)))
                elif len(images) > batch_size:
                    # Truncate extra images
                    images = images[:batch_size]

    # Format prompts using chat template if needed
    if format_prompts:
        config = model.config.__dict__ if hasattr(model.config, "__dict__") else model.config
        formatted_prompts = []

        for i, prompt in enumerate(prompts):
            img_count = len(images[i]) if images is not None and i < len(images) else 0
            prompt_msgs = [[{"role": "user", "content": prompt}]]

            formatted_prompt = apply_chat_template(
                processor,
                config,
                prompt_msgs[0],
                add_generation_prompt=True,
                tokenize=False,
                num_images=img_count
            )
            formatted_prompts.append(formatted_prompt)

        prompts = formatted_prompts

    if verbose:
        print("=" * 10)
        if images is not None:
            print(f"Processing batch of {batch_size} inputs with images")
        else:
            print(f"Processing batch of {batch_size} text inputs")

    # Collect all generated texts
    texts = [""] * batch_size
    last_result = None

    for result in batch_stream_generate(model, processor, prompts, images, **kwargs):
        texts = result.texts
        if verbose:
            # Print incremental output for each batch item
            for i, text in enumerate(texts):
                if text and text != texts[i]:
                    print(f"[Batch {i}] {text}", end="", flush=True)
        last_result = result

    if verbose:
        print("\n" + "=" * 10)
        if len(texts) == 0:
            print("No text generated for this batch")
            return []

        print(
            f"Prompt: {last_result.prompt_tokens} tokens, "
            f"{last_result.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_result.generation_tokens} tokens, "
            f"{last_result.generation_tps:.3f} tokens-per-sec"
        )

        if hasattr(mx, "metal"):
            print(f"Peak memory: {last_result.peak_memory:.3f} GB")

    return texts
