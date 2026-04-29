import json
from typing import Any

import mlx.core as mx


class LLGuidanceLogitsProcessor:
    """MLX logits processor backed by llguidance.

    Accepts a single sequence or a batch. Expected shapes are input_ids as
    (seq_len,) or (batch, seq_len), and logits as (vocab,), (1, vocab), or
    (batch, vocab).
    """

    def __init__(self, grammar: str, llg_tokenizer) -> None:
        self.grammar = grammar
        self.llg_tokenizer = llg_tokenizer
        self.is_first_token = True

    def clone(self) -> "LLGuidanceLogitsProcessor":
        return LLGuidanceLogitsProcessor(self.grammar, self.llg_tokenizer)

    def reset(self):
        self.is_first_token = True
        self.ll_matchers = None
        self.bitmask = None

    def _setup(self, batch_size: int) -> None:
        import llguidance
        import llguidance.numpy
        from llguidance import LLMatcher

        self.ll_matchers = [
            LLMatcher(self.llg_tokenizer, self.grammar) for _ in range(batch_size)
        ]
        self.bitmask = llguidance.numpy.allocate_token_bitmask(
            batch_size, self.llg_tokenizer.vocab_size
        )

    def _consume_tokens(self, last_tokens: list[int]) -> None:
        for i, last_token in enumerate(last_tokens):
            self.ll_matchers[i].consume_token(last_token)
            error = self.ll_matchers[i].get_error()
            if error:
                raise ValueError(f"LLGuidance matcher error: {error}")

    def _apply_bitmask(self, logits: mx.array) -> mx.array:
        import llguidance.mlx
        import llguidance.numpy

        biased_logits = []
        for i in range(logits.shape[0]):
            llguidance.numpy.fill_next_token_bitmask(
                self.ll_matchers[i], self.bitmask, i
            )
            row = mx.array(
                llguidance.mlx.apply_token_bitmask(logits[i], self.bitmask[i])
            )
            if row.ndim == 2 and row.shape[0] == 1:
                row = row[0]
            biased_logits.append(row)
        return mx.concatenate(
            [mx.array(logit)[None, :] for logit in biased_logits], axis=0
        )

    def process_last_token(self, last_token: int, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            return self.process_last_token(last_token, mx.expand_dims(logits, 0))[0]

        if self.is_first_token:
            self._setup(logits.shape[0])
            self.is_first_token = False
        else:
            self._consume_tokens([last_token])

        return self._apply_bitmask(logits)

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if input_ids.ndim == 1 and logits.ndim == 2 and logits.shape[0] == 1:
            input_ids = mx.expand_dims(input_ids, 0)
        elif logits.ndim == 1:
            return self(mx.expand_dims(input_ids, 0), mx.expand_dims(logits, 0))[0]

        batch_size = input_ids.shape[0]
        if self.is_first_token:
            self._setup(batch_size)
            self.is_first_token = False
        else:
            self._consume_tokens(input_ids[:, -1].tolist())

        return self._apply_bitmask(logits)


def _serialize_schema(schema: str | dict[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema)


# Building an llguidance tokenizer walks the entire vocab (~1.5s for a 150k
# token model), so we keep the result around for the lifetime of the process.
_llg_tokenizer_cache = {}


def build_json_schema_logits_processor(tokenizer, schema: str | dict[str, Any]):
    try:
        import llguidance as llg
        import llguidance.hf
    except ImportError as exc:
        raise ImportError(
            "llguidance is required for structured response_format generation. "
            "Install mlx-vlm with llguidance available."
        ) from exc

    llg_tokenizer = _llg_tokenizer_cache.get(id(tokenizer))
    if llg_tokenizer is None:
        llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        _llg_tokenizer_cache[id(tokenizer)] = llg_tokenizer

    grammar = llg.grammar_from("json_schema", _serialize_schema(schema))
    return LLGuidanceLogitsProcessor(grammar, llg_tokenizer)
