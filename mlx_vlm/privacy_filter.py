"""OpenAI Privacy Filter: constrained BIOES Viterbi decoding over the shared
token-classification pipeline in ``mlx_vlm.token_classification``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from transformers import AutoTokenizer

from .token_classification import (
    BOUNDARIES,
    TokenClassificationResult,
    TokenClassifier,
    TokenSpan,
    build_label_info,
    is_token_classifier_config,
    load_token_classifier,
    run_cli,
)
from .utils import get_model_path, load_config, load_model

PrivacySpan = TokenSpan
PrivacyFilterResult = TokenClassificationResult

VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


class ViterbiDecoder:
    """Linear-time constrained BIOES decoder used by Privacy Filter."""

    def __init__(
        self,
        labels: Sequence[str],
        transition_biases: Optional[Mapping[str, float]] = None,
    ):
        self.label_info = build_label_info(labels)
        for span_name, states in self.label_info.states_by_span.items():
            missing = BOUNDARIES - set(states)
            if missing:
                raise ValueError(
                    f"Privacy Filter labels for {span_name!r} are missing "
                    f"{sorted(missing)}"
                )
        supplied = dict(transition_biases or {})
        unknown = set(supplied) - set(VITERBI_BIAS_KEYS)
        if unknown:
            raise ValueError(f"Unknown Viterbi transition biases: {sorted(unknown)}")
        self.biases = {key: float(supplied.get(key, 0.0)) for key in VITERBI_BIAS_KEYS}

        states = self.label_info.states_by_span
        self._b = np.asarray([value["B"] for value in states.values()], dtype=np.int32)
        self._i = np.asarray([value["I"] for value in states.values()], dtype=np.int32)
        self._e = np.asarray([value["E"] for value in states.values()], dtype=np.int32)
        self._s = np.asarray([value["S"] for value in states.values()], dtype=np.int32)
        self._start_states = np.concatenate(
            (np.asarray([self.label_info.background], dtype=np.int32), self._b, self._s)
        )
        self._end_states = np.concatenate(
            (np.asarray([self.label_info.background], dtype=np.int32), self._e, self._s)
        )
        self._closed_states = np.concatenate((self._e, self._s))

    def decode(self, emissions: np.ndarray) -> list[int]:
        emissions = np.asarray(emissions, dtype=np.float32)
        if emissions.ndim != 2:
            raise ValueError("emissions must have shape [tokens, labels]")
        length, num_labels = emissions.shape
        if num_labels != len(self.label_info.labels):
            raise ValueError(
                f"Expected {len(self.label_info.labels)} labels, got {num_labels}"
            )
        if length == 0:
            return []

        negative_infinity = np.float32(-1e9)
        scores = np.full((num_labels,), negative_infinity, dtype=np.float32)
        scores[self._start_states] = emissions[0, self._start_states]
        backpointers = np.full((length - 1, num_labels), -1, dtype=np.int16)
        background = self.label_info.background
        biases = self.biases

        for step in range(1, length):
            next_scores = np.full_like(scores, negative_infinity)
            pointers = backpointers[step - 1]

            closed_values = scores[self._closed_states]
            closed_offset = int(np.argmax(closed_values))
            best_closed = float(closed_values[closed_offset])
            best_closed_state = int(self._closed_states[closed_offset])

            background_options = (
                float(scores[background]) + biases["transition_bias_background_stay"],
                best_closed + biases["transition_bias_end_to_background"],
            )
            if background_options[0] >= background_options[1]:
                background_score, background_prev = background_options[0], background
            else:
                background_score, background_prev = (
                    background_options[1],
                    best_closed_state,
                )
            next_scores[background] = background_score + emissions[step, background]
            pointers[background] = background_prev

            start_options = (
                float(scores[background])
                + biases["transition_bias_background_to_start"],
                best_closed + biases["transition_bias_end_to_start"],
            )
            if start_options[0] >= start_options[1]:
                start_score, start_prev = start_options[0], background
            else:
                start_score, start_prev = start_options[1], best_closed_state
            next_scores[self._b] = start_score + emissions[step, self._b]
            next_scores[self._s] = start_score + emissions[step, self._s]
            pointers[self._b] = start_prev
            pointers[self._s] = start_prev

            open_scores = np.stack((scores[self._b], scores[self._i]), axis=1)
            choose_inside = open_scores[:, 1] > open_scores[:, 0]
            best_open = np.where(choose_inside, open_scores[:, 1], open_scores[:, 0])
            best_open_state = np.where(choose_inside, self._i, self._b)
            next_scores[self._i] = (
                best_open
                + biases["transition_bias_inside_to_continue"]
                + emissions[step, self._i]
            )
            next_scores[self._e] = (
                best_open
                + biases["transition_bias_inside_to_end"]
                + emissions[step, self._e]
            )
            pointers[self._i] = best_open_state
            pointers[self._e] = best_open_state
            scores = next_scores

        end_scores = scores[self._end_states]
        last_label = int(self._end_states[int(np.argmax(end_scores))])
        path = np.empty((length,), dtype=np.int32)
        path[-1] = last_label
        for step in range(length - 2, -1, -1):
            last_label = int(backpointers[step, last_label])
            path[step] = last_label
        return path.tolist()


def _load_transition_biases(model_path: Path, operating_point: str) -> dict[str, float]:
    calibration_path = model_path / "viterbi_calibration.json"
    if not calibration_path.is_file():
        return {key: 0.0 for key in VITERBI_BIAS_KEYS}
    with calibration_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        raw_biases = payload["operating_points"][operating_point]["biases"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown Privacy Filter operating point {operating_point!r}"
        ) from exc
    missing = set(VITERBI_BIAS_KEYS) - set(raw_biases)
    if missing:
        raise ValueError(f"Viterbi calibration is missing biases: {sorted(missing)}")
    return {key: float(raw_biases[key]) for key in VITERBI_BIAS_KEYS}


class PrivacyFilter(TokenClassifier):
    """Privacy Filter pipeline: raw token windows, Viterbi decoding by default."""

    decode_modes = ("viterbi", "argmax")
    default_decode = "viterbi"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        transition_biases: Optional[Mapping[str, float]] = None,
        context_size: Optional[int] = None,
    ):
        super().__init__(
            model,
            tokenizer,
            context_size=context_size,
            special_tokens=((), ()),
            group_words=False,
        )
        self.decoder = ViterbiDecoder(self.labels, transition_biases)
        self.label_info = self.decoder.label_info

    def decode_path(self, scores: np.ndarray, decode: str) -> list[int]:
        if decode == "viterbi":
            return self.decoder.decode(scores)
        return super().decode_path(scores, decode)


def load_privacy_filter(
    path_or_hf_repo: str = "openai/privacy-filter",
    *,
    revision: Optional[str] = None,
    force_download: bool = False,
    lazy: bool = False,
    strict: bool = True,
    operating_point: str = "default",
    transition_biases: Optional[Mapping[str, float]] = None,
    context_size: Optional[int] = None,
) -> TokenClassifier:
    """Load a Privacy Filter checkpoint, or any BIO token-classification one.

    ``*ForTokenClassification`` encoder checkpoints (for example Rampart) load
    through ``load_token_classifier`` and decode with argmax; the Viterbi
    options apply only to OpenAI Privacy Filter.
    """

    model_path = get_model_path(
        path_or_hf_repo,
        revision=revision,
        force_download=force_download,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "viterbi_calibration.json",
        ],
    )
    config = load_config(model_path)
    if config.get("model_type") != "openai_privacy_filter" and (
        is_token_classifier_config(config)
    ):
        if transition_biases is not None or operating_point != "default":
            raise ValueError(
                "Viterbi transition biases only apply to OpenAI Privacy Filter"
            )
        return load_token_classifier(
            model_path, lazy=lazy, strict=strict, context_size=context_size
        )
    model = load_model(model_path, lazy=lazy, strict=strict)
    if model.model_type != "openai_privacy_filter":
        raise ValueError(
            f"Expected an OpenAI Privacy Filter checkpoint, got {model.model_type!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    biases = (
        dict(transition_biases)
        if transition_biases is not None
        else _load_transition_biases(model_path, operating_point)
    )
    return PrivacyFilter(
        model,
        tokenizer,
        transition_biases=biases,
        context_size=context_size,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(
        argv,
        load_privacy_filter,
        description="Detect and redact PII with MLX",
        default_model="openai/privacy-filter",
    )


if __name__ == "__main__":
    main()
