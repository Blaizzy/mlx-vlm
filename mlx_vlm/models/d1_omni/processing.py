from ..base import install_auto_processor_patch

_TOKENIZER_KWARGS = ("revision", "cache_dir", "token", "local_files_only", "subfolder")


class D1OmniProcessor:
    """d1-omni ships only tokenizer files; its config's ``auto_map`` would make
    ``AutoProcessor``/``AutoTokenizer`` ask for remote code, so load the fast
    tokenizer directly."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from transformers import PreTrainedTokenizerFast

        kwargs = {k: v for k, v in kwargs.items() if k in _TOKENIZER_KWARGS}
        return PreTrainedTokenizerFast.from_pretrained(path, **kwargs)


install_auto_processor_patch(["d1_omni"], D1OmniProcessor)
