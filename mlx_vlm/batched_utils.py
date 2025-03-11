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
    apply_repetition_penalty,
    stream_generate as single_stream_generate
)
from .batch_processor import fix_processor_for_batch, get_model_type, batch_preprocess_images

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
    prompt: str,
    images: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Stream generate text for a single prompt with multiple images.
    
    Args:
        model: The vision-language model
        processor: The tokenizer/processor
        prompt: The prompt string
        images: Optional list of image paths
        **kwargs: Additional generation arguments
        
    Returns:
        Generated text response
    """
    text = ""
    last_response = None
    
    for response in single_stream_generate(model, processor, prompt, images, **kwargs):
        text += response.text
        last_response = response
    
    return text

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
    # Fix processor configuration for batch processing
    processor = fix_processor_for_batch(processor, model.config)
    model_type = get_model_type(model.config)
    
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
            
        # Preprocess images based on model type
        for i in range(len(images)):
            images[i] = batch_preprocess_images(processor, model_type, images[i])

    # Format prompts using chat template if needed
    if format_prompts:
        config = model.config.__dict__ if hasattr(model.config, "__dict__") else model.config
        formatted_prompts = []
        for i, prompt in enumerate(prompts):
            num_images = len(images[i]) if images is not None else 0
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            formatted_prompt = apply_chat_template(processor, config, messages, num_images=num_images)
            formatted_prompts.append(formatted_prompt)
        prompts = formatted_prompts

    # Generate responses for each prompt
    responses = []
    for i in range(batch_size):
        if verbose:
            print(f"\nGenerating response {i+1}/{batch_size}...")
            print(f"Prompt: {prompts[i]}")
            if images is not None:
                print(f"Images: {images[i]}")
        
        current_images = images[i] if images is not None else None
        current_prompt = prompts[i]
        
        # For single image case, unpack from list
        if current_images and len(current_images) == 1:
            current_images = current_images[0]
            
        response = batch_stream_generate(
            model,
            processor,
            current_prompt,
            current_images,
            **kwargs
        )
        responses.append(response)
        
        if verbose:
            print(f"Response: {response}")
    
    return responses
