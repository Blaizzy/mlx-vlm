from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimpleVocab:
    bos: int = 1
    eos: int = 2
    pad: int = 0
    unk: int = 3
    image: int = 151652


class SimpleTokenizer:
    """Whitespace-based tokenizer with minimal special handling."""

    def __init__(self, image_token_id: int | None = None):
        self.vocab = SimpleVocab()
        if image_token_id is not None:
            self.vocab.image = int(image_token_id)
        self._base = 200_000
        self._specials = {
            "<bos>": self.vocab.bos,
            "<eos>": self.vocab.eos,
            "<pad>": self.vocab.pad,
            "<unk>": self.vocab.unk,
            "<image>": self.vocab.image,
        }

    @property
    def image_token_id(self) -> int:
        return self.vocab.image

    def encode(self, text: str) -> list[int]:
        tokens = []
        for token in text.strip().split():
            tok = token.strip()
            if not tok:
                continue
            if tok in self._specials:
                tokens.append(self._specials[tok])
            else:
                token_id = self._base + (abs(hash(tok)) % 10_000)
                tokens.append(token_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self._specials.items()}
        pieces = []
        for idx in ids:
            value = int(idx)
            pieces.append(inverse.get(value, f"<tok{value}>") )
        return " ".join(pieces)


def render_chat(prompt: str) -> str:
    """Placeholder chat templating hook."""
    return prompt
