"""Tests for the batch_generate left-padding fix (text-only mixed-length batches).

Bug: batch_generate tokenizes text-only prompts with padding=True/padding_side="left", producing a
uniform-length batch, then hands it to BatchGenerator.insert. Because the rows are now equal length,
PromptProcessingBatch derived left_padding = [max_len - len] = all-zero and lost the real padding, so the
model ran pad-blind over the pad tokens and the more-padded rows decoded incorrectly / empty. Fix: derive
the true per-row left_padding from the tokenizer's attention_mask and thread it (authoritative-when-provided)
into the cache. Model-agnostic; backward compatible (unchanged when no left_padding is supplied).

Model-gated tests use a local Qwen2.5-VL checkpoint if present, else skip.
"""
import os
import tempfile
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate import BatchGenerator

MODEL_PATH = os.environ.get("MLX_VLM_TEST_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")


# --------------------------- model-free invariant tests ---------------------------

def test_left_padding_derivation_from_attention_mask():
    """The derivation `left_padding = L - attention_mask.sum(axis=1)` recovers the true per-row
    left-padding IFF the mask is binary with contiguous leading zeros (left padding). This locks the
    invariant the fix relies on."""
    # Rows with 2, 0, 3 leading pad columns (length 5).
    mask = mx.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=mx.int32)
    L = mask.shape[1]
    left_padding = (L - mask.sum(axis=1)).astype(mx.int32).tolist()
    assert left_padding == [2, 0, 3]
    # Binary invariant: every entry is 0 or 1.
    assert set(mask.reshape(-1).tolist()) <= {0, 1}
    # Contiguous-leading-zeros invariant: once a 1 appears, no 0 follows (left padding only).
    for row in mask.tolist():
        seen_one = False
        for v in row:
            if v == 1:
                seen_one = True
            elif seen_one:
                pytest.fail(f"non-contiguous / non-left padding in mask row {row}")


def _bare_batch_generator():
    """A BatchGenerator with only the attributes insert() touches (no model / weights)."""
    bg = object.__new__(BatchGenerator)
    bg.max_tokens = 16
    bg.logits_processors = None
    bg.uid_count = 0
    bg._unprocessed_sequences = []
    bg._pending_left_padding = {}
    bg._wire_stack = None  # so __del__ -> close() doesn't AttributeError on GC of this bare object
    return bg


def test_insert_left_padding_survives_length_sort():
    """insert() sorts the queue by prompt length. The per-row left_padding must stay bound to its own
    sequence afterward. It is tracked by uid (sort-immune), so this verifies the plumbing end to end."""
    bg = _bare_batch_generator()
    prompts = [[1, 2, 3, 4], [1], [1, 2, 3, 4, 5, 6], [1, 2]]  # distinct lengths, deliberately unsorted
    left_padding = [10, 20, 30, 40]  # distinct sentinel per row
    uids = bg.insert(prompts, max_tokens=8, left_padding=left_padding)

    # Each uid maps to the left_padding of the row it was submitted with, regardless of the sort.
    for uid, lpad in zip(uids, left_padding):
        assert bg._pending_left_padding[uid] == lpad
    # The queue itself is sorted ascending by length (sort actually happened).
    lengths = [len(t[1]) for t in bg._unprocessed_sequences]
    assert lengths == sorted(lengths)


def test_insert_without_left_padding_is_unchanged():
    """Backward-compat: callers that don't pass left_padding leave _pending_left_padding empty, so the
    downstream length-derived behavior is untouched."""
    bg = _bare_batch_generator()
    bg.insert([[1, 2, 3], [1, 2]], max_tokens=8)
    assert bg._pending_left_padding == {}


# --------------------------- model-gated end-to-end tests ---------------------------

@pytest.fixture(scope="module")
def vlm():
    from mlx_vlm import load
    try:
        return load(MODEL_PATH)
    except Exception as e:  # noqa: BLE001 — model unavailable (no local copy / no network) -> skip
        pytest.skip(f"model unavailable ({MODEL_PATH}): {e}")


def _tiny_png():
    from PIL import Image
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.new("RGB", (64, 64), (10, 120, 200)).save(path)
    return path


def test_tokenizer_emits_contiguous_left_pad_mask(vlm):
    """Guard the tokenizer invariant against a future change: the text-only batched tokenize must emit a
    binary attention_mask with contiguous LEADING zeros (left padding) — otherwise the L - sum derivation
    would silently be wrong. Fails loudly if a future tokenizer/config alters padding behavior."""
    _model, processor = vlm
    prompts = ["Hi.", "A somewhat longer prompt that tokenizes to more tokens than the first one."]
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    enc = tok(prompts, add_special_tokens=True, padding=True, padding_side="left",
              return_tensors="np")
    mask = enc["attention_mask"]
    assert mask.ndim == 2
    assert set(mask.reshape(-1).tolist()) <= {0, 1}, "attention_mask must be binary"
    for row in mask.tolist():
        seen_one = False
        for v in row:
            if v == 1:
                seen_one = True
            elif seen_one:
                pytest.fail("attention_mask is not contiguous-leading-zero (left) padding")


def test_batch_generate_mixed_length_text_only_matches_oracle(vlm):
    """The crux: batch_generate on DIFFERENT-length text-only prompts must produce 0 empty rows AND match
    the width-1 (single-prompt) greedy oracle for every row."""
    from mlx_vlm.generate import batch_generate
    model, proc = vlm
    prompts = [
        "Reply with only the word alpha.",
        "In one short word only, name a fruit that is red. Word only.",
        "Considering trees in temperate climates, reply with only the single word beta.",
        "Reply with only the word gamma.",
        "Think about the ocean and its creatures, then reply with only the single word delta.",
        "Reply with only the word epsilon.",
    ]
    batched = list(batch_generate(model, proc, images=None, prompts=prompts,
                                  max_tokens=24, verbose=False).texts)
    assert all(t.strip() for t in batched), f"empty row(s): {batched}"
    for i, p in enumerate(prompts):
        oracle = batch_generate(model, proc, images=None, prompts=[p],
                                max_tokens=24, verbose=False).texts[0]
        assert batched[i] == oracle, f"row {i} diverged: {batched[i]!r} != oracle {oracle!r}"


def test_batch_generate_image_path_unchanged(vlm):
    """Multimodal regression: the image path (pixel_values is not None) does NOT derive left_padding and
    still produces non-empty output — the fix must not perturb multimodal batching."""
    from mlx_vlm.generate import batch_generate
    model, proc = vlm
    png = _tiny_png()
    try:
        out = batch_generate(model, proc, images=[png],
                             prompts=["What color is this image? Answer in one word."],
                             max_tokens=16, verbose=False)
        assert out.texts and out.texts[0].strip(), "image path returned empty"
    finally:
        os.remove(png)
