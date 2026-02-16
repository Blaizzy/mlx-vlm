from transformers import AddedToken
from transformers.models.florence2.processing_florence2 import Florence2Processor

from ..base import install_auto_processor_patch

_ORIGINAL_INIT = Florence2Processor.__init__

_BOOTSTRAP_PREFIX_TOKENS = ["</od>", "<ocr>", "</ocr>"]
_BOOTSTRAP_LOC_TOKENS = [f"<loc_{i}>" for i in range(1000)]
_BOOTSTRAP_SUFFIX_TOKENS = [
    "<cap>",
    "</cap>",
    "<ncap>",
    "</ncap>",
    "<dcap>",
    "</dcap>",
    "<grounding>",
    "</grounding>",
    "<seg>",
    "</seg>",
    "<sep>",
    "<region_cap>",
    "</region_cap>",
    "<region_to_desciption>",
    "</region_to_desciption>",
    "<proposal>",
    "</proposal>",
    "<poly>",
    "</poly>",
    "<and>",
]
_IMAGE_PLACEHOLDER_TOKEN = "<florence_image>"


def _added_token(text: str) -> AddedToken:
    return AddedToken(
        text,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
    )


def _ensure_florence_tokens(tokenizer) -> None:
    full_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if "<image>" not in full_vocab:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [_added_token("<image>")]}
        )

    added_vocab = (
        tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    )
    if "<loc_0>" in added_vocab:
        return

    bootstrap_tokens = (
        _BOOTSTRAP_PREFIX_TOKENS + _BOOTSTRAP_LOC_TOKENS + _BOOTSTRAP_SUFFIX_TOKENS
    )
    missing = [tok for tok in bootstrap_tokens if tok not in added_vocab]
    if missing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [_added_token(tok) for tok in missing]}
        )

    added_vocab = tokenizer.get_added_vocab()
    if _IMAGE_PLACEHOLDER_TOKEN not in added_vocab:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [_added_token(_IMAGE_PLACEHOLDER_TOKEN)]}
        )


def _patched_init(self, image_processor=None, tokenizer=None, **kwargs):
    if tokenizer is not None:
        _ensure_florence_tokens(tokenizer)

        if not hasattr(tokenizer, "image_token"):
            tokenizer.image_token = "<image>"

        added_vocab = (
            tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
        )
        if _IMAGE_PLACEHOLDER_TOKEN in added_vocab:
            tokenizer.image_token = _IMAGE_PLACEHOLDER_TOKEN
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(
                _IMAGE_PLACEHOLDER_TOKEN
            )
        elif not hasattr(tokenizer, "image_token_id"):
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.image_token
            )

    _ORIGINAL_INIT(self, image_processor=image_processor, tokenizer=tokenizer, **kwargs)


Florence2Processor.__init__ = _patched_init

install_auto_processor_patch("florence2", Florence2Processor)
