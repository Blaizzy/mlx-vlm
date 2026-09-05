"""Unlimited-OCR prompt boundaries, without model weights or downloads."""

import json

import pytest

pytest.importorskip("jinja2")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from mlx_vlm.models.deepseekocr.processing_deepseekocr import DeepseekOCRProcessor
from mlx_vlm.models.unlimited_ocr.processing_unlimitedocr import UnlimitedOCRProcessor
from mlx_vlm.prompt_utils import apply_chat_template


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "<image>": 4}
    for char in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[char] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        additional_special_tokens=["<image>"],
    )


@pytest.fixture
def processor(tokenizer):
    return UnlimitedOCRProcessor(tokenizer=tokenizer)


def render(processor, messages, **kwargs):
    return processor.apply_chat_template(messages, tokenize=False, **kwargs)


@pytest.mark.parametrize("task", ["document parsing.", "Multi page parsing."])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_completed_task_has_no_template_suffix(processor, task, add_generation_prompt):
    prompt = "<image>" + task
    assert (
        render(
            processor,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=add_generation_prompt,
        )
        == prompt
    )


@pytest.mark.parametrize(
    "task,num_images", [("document parsing.", 1), ("Multi page parsing.", 3)]
)
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_public_prompt_helper_keeps_native_single_image_marker(
    processor, task, num_images, add_generation_prompt
):
    prompt = apply_chat_template(
        processor,
        {"model_type": "unlimited-ocr"},
        task,
        num_images=num_images,
        add_generation_prompt=add_generation_prompt,
    )
    assert prompt == "<image>" + task


@pytest.mark.parametrize("whitespace", [" ", " \t", "\n"])
def test_final_user_content_whitespace_is_not_stripped(processor, whitespace):
    content = "<image>document parsing." + whitespace
    assert render(processor, [{"role": "user", "content": content}]) == content


def test_intermediate_message_separators_are_preserved(processor):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "<image>document parsing."},
    ]
    assert render(processor, messages) == "system first reply <image>document parsing."


@pytest.mark.parametrize(
    "continue_final_message,expected",
    [
        (False, "<image>document parsing. <table>  "),
        (True, "<image>document parsing. <table> "),
    ],
)
def test_final_assistant_whitespace_is_preserved(
    processor, continue_final_message, expected
):
    messages = [
        {"role": "user", "content": "<image>document parsing."},
        {"role": "assistant", "content": "<table> "},
    ]
    assert (
        render(
            processor,
            messages,
            add_generation_prompt=False,
            continue_final_message=continue_final_message,
        )
        == expected
    )


def test_completed_prompt_tokenization_has_no_added_space_token(processor):
    prompt = "<image>document parsing."
    rendered = render(processor, [{"role": "user", "content": prompt}])
    tokenizer = processor.tokenizer
    assert tokenizer.encode(rendered) == tokenizer.encode(prompt)
    assert tokenizer.encode(prompt + " ") != tokenizer.encode(prompt)


def test_constructor_template_override_is_preserved(tokenizer):
    processor = UnlimitedOCRProcessor(
        tokenizer=tokenizer, chat_template="CUSTOM:{{messages[0]['content']}}  "
    )
    assert render(processor, [{"role": "user", "content": "task"}]) == "CUSTOM:task  "


def test_per_call_template_override_does_not_persist(processor):
    messages = [{"role": "user", "content": "task"}]
    assert (
        render(processor, messages, chat_template="CALL:{{messages[0]['content']}}  ")
        == "CALL:task  "
    )
    assert render(processor, messages) == "task"


def test_processor_config_and_loader_override_precedence(tokenizer, tmp_path):
    tokenizer.chat_template = "TOKENIZER:{{messages[0]['content']}}"
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"chat_template": "CONFIG:{{messages[0]['content']}}  "})
    )
    messages = [{"role": "user", "content": "task"}]
    configured = UnlimitedOCRProcessor.from_pretrained(tmp_path)
    assert render(configured, messages) == "CONFIG:task  "
    overridden = UnlimitedOCRProcessor.from_pretrained(
        tmp_path, chat_template="LOADER:{{messages[0]['content']}}  "
    )
    assert render(overridden, messages) == "LOADER:task  "


def test_checkpoint_tokenizer_template_does_not_replace_processor_default(
    tokenizer, tmp_path
):
    tokenizer.chat_template = "TOKENIZER:{{messages[0]['content']}}"
    tokenizer.save_pretrained(tmp_path)
    processor = UnlimitedOCRProcessor.from_pretrained(tmp_path)
    assert render(processor, [{"role": "user", "content": "task"}]) == "task"


def test_deepseek_default_is_unchanged(tokenizer):
    processor = DeepseekOCRProcessor(
        tokenizer=tokenizer,
        candidate_resolutions=((1024, 1024),),
        patch_size=16,
        downsample_ratio=4,
    )
    assert render(processor, [{"role": "user", "content": "task"}]) == "task "
