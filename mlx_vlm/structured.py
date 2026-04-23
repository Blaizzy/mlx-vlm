import json
from typing import Any

import mlx.core as mx


class LLGuidanceLogitsProcessor:
    """MLX logits processor backed by llguidance."""

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

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        import llguidance.mlx
        import llguidance.numpy

        if input_ids.ndim == 1 and logits.ndim == 2 and logits.shape[0] == 1:
            input_ids = mx.expand_dims(input_ids, 0)
        elif logits.ndim == 1:
            return self(mx.expand_dims(input_ids, 0), mx.expand_dims(logits, 0))[0]

        batch_size = input_ids.shape[0]
        if self.is_first_token:
            self._setup(batch_size)
            self.is_first_token = False
        else:
            for i in range(batch_size):
                last_token = input_ids[i][-1].item()
                self.ll_matchers[i].consume_token(last_token)
                error = self.ll_matchers[i].get_error()
                if error:
                    raise ValueError(f"LLGuidance matcher error: {error}")

        biased_logits = []
        for i in range(batch_size):
            llguidance.numpy.fill_next_token_bitmask(
                self.ll_matchers[i], self.bitmask, i
            )
            row = mx.array(llguidance.mlx.apply_token_bitmask(logits[i], self.bitmask[i]))
            if row.ndim == 2 and row.shape[0] == 1:
                row = row[0]
            biased_logits.append(row)
        return mx.concatenate(
            [mx.array(logit)[None, :] for logit in biased_logits], axis=0
        )


def _serialize_schema(schema: str | dict[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema)


def build_json_schema_logits_processor(tokenizer, schema: str | dict[str, Any]):
    try:
        import llguidance as llg
        import llguidance.hf
    except ImportError as exc:
        raise ImportError(
            "llguidance is required for structured response_format generation. "
            "Install mlx-vlm with llguidance available."
        ) from exc

    llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
    grammar = llg.grammar_from("json_schema", _serialize_schema(schema))
    return LLGuidanceLogitsProcessor(grammar, llg_tokenizer)
