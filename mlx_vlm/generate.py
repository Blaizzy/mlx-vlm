import argparse
import codecs
import contextlib
import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache
from transformers import PreTrainedTokenizer

from mlx_vlm.models.gemma3n import audio

from .models import cache
from .prompt_utils import apply_chat_template
from .sample_utils import top_p_sampling
from .utils import StoppingCriteria, apply_repetition_penalty, load, prepare_inputs

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_QUANTIZED_KV_START = 5000


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=DEFAULT_AUDIO,
        help="URL or path of the audio to process.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument("--verbose", action="store_false", help="Detailed output.")
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for the KV cache.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model from Hugging Face.",
    )

    return parser.parse_args()


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        logit_bias (dictionary, optional): Additive logit bias.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if temperature == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temperature)
            else:
                token = mx.random.categorical(logits * (1 / temperature))

        return token, logprobs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = input_ids

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    repetition_context = input_ids.reshape(-1).tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y, **kwargs):
        with mx.stream(generation_stream):
            nonlocal repetition_context
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y[None],
                    cache=prompt_cache,
                    **kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if repetition_penalty:
                logits = apply_repetition_penalty(
                    logits, repetition_context, repetition_penalty
                )
                y, logprobs = sample(logits)
                repetition_context.append(y.item())
            else:
                y, logprobs = sample(logits)

            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]

            quantize_cache_fn(prompt_cache)
            return y, logprobs.squeeze(0)

    outputs = model(input_ids, pixel_values, cache=prompt_cache, mask=mask, **kwargs)

    logits = outputs.logits[:, -1, :]
    quantize_cache_fn(prompt_cache)
    y, logprobs = sample(logits)
    mx.async_eval(y)

    if outputs.cross_attention_states is not None:
        kwargs = {
            k: v
            for k, v in zip(
                ["cross_attention_states"], [outputs.cross_attention_states]
            )
        }
    elif outputs.encoder_outputs is not None:
        kwargs = {
            "decoder_input_ids": y[None],
            "encoder_outputs": outputs.encoder_outputs,
        }
    else:
        kwargs = {}

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y, **kwargs)
            mx.async_eval(next_y)
            if "decoder_input_ids" in kwargs:
                kwargs["decoder_input_ids"] = next_y[None]
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs
        if n == max_tokens:
            break

        n += 1

        # Periodically clear cache to prevent memory accumulation
        if n % 256 == 0:  # Clear cache every 256 tokens
            mx.clear_cache()


def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    with wired_limit(model, [generation_stream]):
        detokenizer = processor.detokenizer
        detokenizer.reset()
        tic = time.perf_counter()
        for n, (token, logprobs) in enumerate(
            generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = input_ids.size / prompt_time
                tic = time.perf_counter()

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=input_ids.size,
                generation_tokens=n + 1,
                total_tokens=input_ids.size + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            total_tokens=input_ids.size + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> GenerationResult:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if kwargs.get("video") is not None:
            files.extend(kwargs.get("video"))

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return GenerationResult(
                text=text,
                token=None,
                logprobs=None,
                prompt_tokens=0,
                generation_tokens=0,
                total_tokens=0,
                prompt_tps=0.0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
    )


@dataclass
class BatchGenerationResult:
    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _make_cache(model, left_padding):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """

    def to_batch_cache(c):
        if isinstance(c, cache.KVCache):
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.BatchCacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        return [to_batch_cache(c) for c in model_cache]
    else:
        return [cache.BatchKVCache(left_padding) for _ in model.layers]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
    """

    texts: List[str]
    stats: BatchStats


@dataclass
class Batch:
    uids: List[int]
    y: mx.array
    logprobs: mx.array
    max_tokens: List[int]
    num_tokens: List[int]
    cache: List[Any]

    def __len__(self):
        return len(self.uids)

    def filter(self, keep_idx: List[int]):
        self.uids = [self.uids[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        keep_idx = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx]
        self.logprobs = self.logprobs[keep_idx]
        for c in self.cache:
            c.filter(keep_idx)

    def extend(self, other):
        self.uids.extend(other.uids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs = mx.concatenate([self.logprobs, other.logprobs])
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        for c, o in zip(self.cache, other.cache):
            c.extend(o)


class BatchGenerator:

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]

    def __init__(
        self,
        model,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        prompt_cache=None,
    ):
        self.model = model
        self.unprocessed_prompts = []
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size
        self.prompt_cache = prompt_cache
        self._stats = BatchStats()

        self.active_batch = None

    def insert(self, prompts, max_tokens: Union[List[int], int, None] = None):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        for p, m in zip(prompts, max_tokens):
            self.unprocessed_prompts.append((self.uid_count, p, m))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self.unprocessed_prompts = sorted(
            self.unprocessed_prompts, key=lambda x: len(x[1])
        )
        return uids

    def _process_prompts(self, prompts, **kwargs) -> Batch:
        uids, inputs, max_tokens = zip(*prompts)
        lengths = [len(p) for p in inputs]
        max_length = max(lengths)
        batch_size = self.prefill_batch_size
        self._stats.prompt_tokens += sum(lengths)
        left_padding = [max_length - l for l in lengths]
        inputs = _left_pad_prompts(inputs, max_length=max_length)

        prompt_cache = (
            _make_cache(self.model, left_padding)
            if self.prompt_cache is None
            else self.prompt_cache
        )

        while inputs.shape[1] > 1 and kwargs.get("pixel_values", None) is None:
            n_to_process = min(self.prefill_step_size, inputs.shape[1] - 1)
            self.model(inputs[:, :n_to_process], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            inputs = inputs[:, n_to_process:]
            mx.clear_cache()

        y, logprobs = self._step(inputs, prompt_cache, **kwargs)
        mx.async_eval(y, logprobs)
        return Batch(
            list(uids), y, logprobs, list(max_tokens), [0] * len(uids), prompt_cache
        )

    def _step(self, input_tokens: mx.array, prompt_cache: List[Any], **kwargs):
        output = self.model(input_tokens, cache=prompt_cache, **kwargs)
        logits = output.logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs

    def stats(self):
        self._stats.prompt_tps = self._stats.prompt_tokens / self._stats.prompt_time
        self._stats.generation_tps = (
            self._stats.generation_tokens / self._stats.generation_time
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _next(self, **kwargs):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts, **kwargs)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason))

        # Remove any finished completions
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def next(self, **kwargs):
        with mx.stream(generation_stream):
            return self._next(**kwargs)


def batch_generate(
    model,
    processor,
    images: Union[str, List[str]] = None,
    audios: Union[str, List[str]] = None,
    prompts: List[int] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    **kwargs,
):
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (List[List[int]]): The input prompts.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """
    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    prompts = [
        apply_chat_template(processor, model.config, p, num_images=1, num_audios=1)
        for p in prompts
    ]

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        audio=audios,
        prompts=prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    gen = BatchGenerator(
        model.language_model, stop_tokens=[tokenizer.eos_token_ids], **kwargs
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    with wired_limit(model, [generation_stream]):
        if pixel_values is not None:
            inputs_embeds = model.get_input_embeddings(
                input_ids, pixel_values, **data_kwargs
            )

            kwargs.update(
                {
                    "pixel_values": pixel_values,
                    **data_kwargs,
                    **(
                        inputs_embeds
                        if isinstance(inputs_embeds, dict)
                        else {"inputs_embeds": inputs_embeds}
                    ),
                }
            )

        uids = gen.insert(mx.squeeze(input_ids).tolist(), max_tokens)
        results = {uid: [] for uid in uids}
        while responses := gen.next(**kwargs):
            for r in responses:
                if verbose and r.finish_reason != None:
                    fin += 1
                    print(
                        f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    stats = gen.stats()
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats)


def main():
    args = parse_arguments()
    if isinstance(args.image, str):
        args.image = [args.image]

    model, processor = load(args.model, args.adapter_path, trust_remote_code=True)
    config = model.config

    prompt = args.prompt

    num_images = len(args.image) if args.image is not None else 0
    num_audios = (
        1 if args.audio is not None else 0
    )  # TODO: Support multiple audio files
    prompt = apply_chat_template(
        processor, config, prompt, num_images=num_images, num_audios=num_audios
    )

    kwargs = {}

    if args.resize_shape is not None:
        if len(args.resize_shape) not in [1, 2]:
            raise ValueError("Resize shape must be 1 or 2 integers")
        kwargs["resize_shape"] = (
            (args.resize_shape[0],) * 2
            if len(args.resize_shape) == 1
            else tuple(args.resize_shape)
        )

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    if args.chat:
        chat = []
        if args.system:
            chat.append({"role": "system", "content": args.system})
        while user := input("User:"):
            chat.append({"role": "user", "content": user})
            prompt = apply_chat_template(
                processor, config, chat, num_images=len(args.image)
            )
            response = ""
            print("Assistant:", end="")
            for chunk in stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                **kwargs,
            ):
                response += chunk.text
                print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            print()

    else:
        result = generate(
            model,
            processor,
            prompt,
            image=args.image,
            audio=args.audio,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            prompt_cache=None,  # TODO: Load prompt cache from file
            max_kv_size=args.max_kv_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            **kwargs,
        )
        if not args.verbose:
            print(result.text)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.generate ...` directly is deprecated."
        " Use `mlx_vlm generate` or `python -m mlx_vlm generate` instead."
    )
    main()
