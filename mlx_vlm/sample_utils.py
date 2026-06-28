import mlx.core as mx


UNLIMITED_OCR_MODEL_TYPES = {"unlimited-ocr", "unlimited_ocr"}
UNLIMITED_OCR_NO_REPEAT_NGRAM_SIZE = 35
UNLIMITED_OCR_SINGLE_IMAGE_NGRAM_WINDOW = 128
UNLIMITED_OCR_MULTI_IMAGE_NGRAM_WINDOW = 1024


class SlidingWindowNoRepeatNGramProcessor:
    """Block repeated n-grams within a recent token window.

    This mirrors Unlimited-OCR's upstream ``SlidingWindowNoRepeatNgramProcessor``
    / SGLang ``DeepseekOCRNoRepeatNGramLogitProcessor`` behavior: if the
    current ``ngram_size - 1`` token prefix has appeared in the recent window,
    the token that completed that prior n-gram is banned for the next step.
    """

    def __init__(
        self,
        ngram_size: int,
        window_size: int | None = None,
        whitelist_token_ids=None,
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"ngram_size must be a positive integer, got {ngram_size}")
        if window_size is not None and (
            not isinstance(window_size, int) or window_size <= 0
        ):
            raise ValueError(
                f"window_size must be a positive integer or None, got {window_size}"
            )
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist = set(whitelist_token_ids or [])

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size == 0:
            return logits

        logits_was_unbatched = logits.ndim == 1
        if logits_was_unbatched:
            logits = logits[None, :]

        sequence_tokens = tokens
        if self.window_size is not None and tokens.shape[-1] > self.window_size:
            sequence_tokens = tokens[..., -self.window_size :]

        if sequence_tokens.ndim == 1:
            sequences = [sequence_tokens.tolist()] * logits.shape[0]
        else:
            sequences = sequence_tokens.tolist()

        for batch_idx, sequence in enumerate(sequences[: logits.shape[0]]):
            banned = self._banned_tokens(sequence)
            if not banned:
                continue
            indices = mx.array(sorted(banned), dtype=mx.int32)
            logits = logits.at[batch_idx, indices].minimum(-float("inf"))

        return logits[0] if logits_was_unbatched else logits

    def _banned_tokens(self, sequence: list[int]) -> set[int]:
        n = self.ngram_size
        if len(sequence) < n:
            return set()

        search_start = 0
        if self.window_size is not None:
            search_start = max(0, len(sequence) - self.window_size)
        search_end = len(sequence) - n + 1
        if search_end <= search_start:
            return set()

        current_prefix = tuple(sequence[-(n - 1) :]) if n > 1 else tuple()
        banned = set()
        for idx in range(search_start, search_end):
            ngram = sequence[idx : idx + n]
            if n == 1 or tuple(ngram[:-1]) == current_prefix:
                banned.add(ngram[-1])
        banned.difference_update(self.whitelist)
        return banned


def _model_type(model_or_config) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    if isinstance(config, dict):
        return str(config.get("model_type", ""))
    return str(getattr(config, "model_type", ""))


def _count_media_items(media) -> int:
    if media is None:
        return 0
    if isinstance(media, (list, tuple)):
        return len(media)
    return 1


def make_sliding_window_no_repeat_ngram_processor(
    model_or_config=None,
    *,
    media=None,
    no_repeat_ngram_size: int | None = None,
    ngram_window: int | None = None,
):
    """Build the no-repeat processor used by Unlimited-OCR's reference code.

    Unlimited-OCR's official examples run deterministic decoding with a
    sliding-window 35-gram repetition blocker.  When ``no_repeat_ngram_size`` is
    omitted, this helper only returns a processor for Unlimited-OCR configs;
    callers may still opt in for other models by passing an explicit size.
    """

    is_unlimited_ocr = _model_type(model_or_config) in UNLIMITED_OCR_MODEL_TYPES
    if no_repeat_ngram_size is None:
        if not is_unlimited_ocr:
            return None
        no_repeat_ngram_size = UNLIMITED_OCR_NO_REPEAT_NGRAM_SIZE

    no_repeat_ngram_size = int(no_repeat_ngram_size)
    if no_repeat_ngram_size <= 0:
        return None

    if ngram_window is None and is_unlimited_ocr:
        ngram_window = (
            UNLIMITED_OCR_MULTI_IMAGE_NGRAM_WINDOW
            if _count_media_items(media) > 1
            else UNLIMITED_OCR_SINGLE_IMAGE_NGRAM_WINDOW
        )
    elif ngram_window is not None:
        ngram_window = int(ngram_window)
        if ngram_window <= 0:
            ngram_window = None

    return SlidingWindowNoRepeatNGramProcessor(
        no_repeat_ngram_size,
        window_size=ngram_window,
    )


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output. Shape [..., vocab];
            commonly [vocab], [B, vocab], or [B, T, vocab] (e.g. MTP
            speculative verify output).
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion. Shape matches logits
        with the trailing vocab axis removed (e.g. [], [B], or [B, T]).
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
    token = mx.take_along_axis(sorted_indices, sampled_pos[..., None], axis=-1).squeeze(
        -1
    )
    return token.squeeze(0) if unbatched else token
