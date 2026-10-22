from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch


class FlintProcessor:
    """Load the checkpoint's tokenizer without executing its remote config."""

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        # Flint ships its own pre-tokenizer; the Mistral heuristic must not
        # rewrite it merely because the vocabulary sizes are similar.
        kwargs.setdefault("fix_mistral_regex", False)
        return PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs)


install_auto_processor_patch("flint", FlintProcessor)
