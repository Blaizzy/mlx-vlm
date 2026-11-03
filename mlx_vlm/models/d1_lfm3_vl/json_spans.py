# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""Spans of a rendered JSON state: where its strings sit and what an escaped slice of one says, without torch."""

from __future__ import annotations

import json


def string_paths(text: str) -> list[tuple[int, int, str]]:
    """String-value spans in rendered JSON, excluding quotes; paths use RFC6901 escaping."""
    decoder = json.JSONDecoder()
    ranges = []

    def skip(at):
        while at < len(text) and text[at].isspace():
            at += 1
        return at

    def value(at, path):
        at = skip(at)
        if text[at] == "{":
            at = skip(at + 1)
            while text[at] != "}":
                key, end = decoder.raw_decode(text, at)
                if not isinstance(key, str) or text[skip(end)] != ":":
                    raise ValueError("invalid JSON object")
                name = key.replace("~", "~0").replace("/", "~1")
                at = skip(value(skip(end) + 1, path + "/" + name))
                if text[at] == ",":
                    at = skip(at + 1)
                elif text[at] != "}":
                    raise ValueError("invalid JSON object separator")
            return at + 1
        if text[at] == "[":
            at, index = skip(at + 1), 0
            while text[at] != "]":
                at = skip(value(at, path + "/" + str(index)))
                index += 1
                if text[at] == ",":
                    at = skip(at + 1)
                elif text[at] != "]":
                    raise ValueError("invalid JSON array separator")
            return at + 1
        item, end = decoder.raw_decode(text, at)
        if isinstance(item, str):
            ranges.append((at + 1, end - 1, path))
        return end

    end = value(0, "")
    if skip(end) != len(text):
        raise ValueError("trailing JSON text")
    return ranges


def json_text(escaped: str) -> str:
    """A slice of a rendered JSON string as the characters it encodes (unchanged when it cuts an escape)."""
    try:
        return json.loads(f'"{escaped}"')
    except ValueError:
        return escaped
