from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence, Union

from ...tokenizer_utils import NaiveStreamingDetokenizer
from ...utils import StoppingCriteria


def _require_mistral_common():
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except Exception as exc:
        msg = (
            "Tekken tokenizer requires optional dependency 'mistral-common'. "
            "Install with `pip install mlx-vlm[tekken]`."
        )
        raise ImportError(msg) from exc

    return MistralTokenizer


def _resolve_model_dir(model_dir_or_repo: Union[str, Path]) -> Path:
    from ...utils import get_model_path

    model_path = Path(model_dir_or_repo)
    if not model_path.exists():
        model_path = get_model_path(str(model_dir_or_repo))
    return model_path


def _load_tekken_from_file(tekken_path: Path):
    MistralTokenizer = _require_mistral_common()
    return MistralTokenizer.from_file(str(tekken_path))


def load_tekken_tokenizer(model_dir_or_repo: Union[str, Path]):
    model_path = _resolve_model_dir(model_dir_or_repo)
    tekken_path = model_path / "tekken.json"
    if not tekken_path.exists():
        raise FileNotFoundError(f"tekken.json not found at {tekken_path}")
    return _load_tekken_from_file(tekken_path)


def _build_chat_request(messages):
    from mistral_common.protocol.instruct.messages import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AssistantMessage(content=content))
        else:
            converted.append(UserMessage(content=content))
    return ChatCompletionRequest(messages=converted)


def encode_instruct(
    prompt_or_messages: Union[str, Sequence[dict]], tokenizer
) -> List[int]:
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = list(prompt_or_messages)

    if hasattr(tokenizer, "encode_chat_completion"):
        tokenized = tokenizer.encode_chat_completion(_build_chat_request(messages))
        return list(tokenized.tokens)

    raise ValueError("Tokenizer does not support chat completion encoding.")


def decode(token_ids: Iterable[int], tokenizer) -> str:
    return tokenizer.decode(list(token_ids))


class TekkenTokenizerWrapper:
    def __init__(self, tokenizer, eos_token_ids: Union[int, List[int], None]):
        self._tokenizer = tokenizer
        self._raw_tokenizer = tokenizer.instruct_tokenizer.tokenizer
        self.eos_token_ids = (
            [eos_token_ids] if isinstance(eos_token_ids, int) else (eos_token_ids or [])
        )
        self.all_special_ids = []
        self.stopping_criteria = StoppingCriteria(self.eos_token_ids, self)

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(self._raw_tokenizer.encode(text, bos=False, eos=False))

    def decode(self, token_ids: Iterable[int]):
        try:
            from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

            return self._tokenizer.decode(
                list(token_ids), special_token_policy=SpecialTokenPolicy.KEEP
            )
        except Exception:
            return self._tokenizer.decode(list(token_ids))

    def encode_messages(self, messages):
        if hasattr(self._tokenizer, "encode_messages"):
            return self._tokenizer.encode_messages(messages)
        raise AttributeError("encode_messages is not available on the tokenizer")

    def encode_chat_completion(self, messages):
        if hasattr(self._tokenizer, "encode_chat_completion"):
            return self._tokenizer.encode_chat_completion(messages)
        raise AttributeError("encode_chat_completion is not available on the tokenizer")

    def get_special_token_id(self, token: str) -> int:
        if hasattr(self._raw_tokenizer, "get_special_token"):
            return self._raw_tokenizer.get_special_token(token)
        raise AttributeError("get_special_token is not available on the tokenizer")

    def __call__(
        self,
        prompts: Union[str, Sequence[str]],
        add_special_tokens: bool = False,
        padding: bool = True,
        padding_side: str = "left",
    ):
        if isinstance(prompts, str):
            input_ids = self.encode(prompts, add_special_tokens=add_special_tokens)
            attention_mask = [1] * len(input_ids)
            return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

        encoded = [
            self.encode(p, add_special_tokens=add_special_tokens) for p in prompts
        ]
        max_len = max(len(ids) for ids in encoded) if padding else None
        input_ids = []
        attention_mask = []
        for ids in encoded:
            if padding and max_len is not None:
                pad_len = max_len - len(ids)
                pad_value = self.eos_token_ids[0] if self.eos_token_ids else 0
                pad = [pad_value] * pad_len
                if padding_side == "left":
                    input_ids.append(pad + ids)
                    attention_mask.append([0] * pad_len + [1] * len(ids))
                else:
                    input_ids.append(ids + pad)
                    attention_mask.append([1] * len(ids) + [0] * pad_len)
            else:
                input_ids.append(ids)
                attention_mask.append([1] * len(ids))
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class TekkenProcessor:
    def __init__(
        self, tokenizer: TekkenTokenizerWrapper, tekken_path: Path | None = None
    ):
        self.tokenizer = tokenizer
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer)
        self._tekken_path = tekken_path

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        if self._tekken_path and self._tekken_path.exists():
            import shutil

            shutil.copy(self._tekken_path, save_path / "tekken.json")


def build_tekken_processor(
    model_dir_or_repo: Union[str, Path], eos_token_ids: Union[int, List[int], None]
):
    tokenizer = load_tekken_tokenizer(model_dir_or_repo)
    wrapper = TekkenTokenizerWrapper(tokenizer, eos_token_ids)
    model_path = _resolve_model_dir(model_dir_or_repo)
    tekken_path = model_path / "tekken.json"
    processor = TekkenProcessor(wrapper, tekken_path=tekken_path)
    try:
        from .audio import VoxtralFeatureExtractor

        processor.feature_extractor = VoxtralFeatureExtractor.from_pretrained(
            model_path
        )
    except FileNotFoundError:
        pass
    return processor
