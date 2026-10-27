"""LiquidAI d1 (LFM3 backbones): prompt and packing, set-attention masks, the
readouts, and a tiny random-weight model end to end. Real checkpoints run when
D1_PATH (d1-2B) or D1_OMNI_PATH (d1-omni-2B) name a local copy."""

import copy
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.d1_lfm3_vl import api, box, d1, pointer, prompt
from mlx_vlm.models.d1_lfm3_vl.calibration import Calibration, through
from mlx_vlm.models.d1_lfm3_vl.grid import Grid
from mlx_vlm.models.d1_lfm3_vl.media import resampled

# LFM3's pre-tokenizer, so numbers split in threes and ":\n" is one piece
PATTERN = (
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SPECIAL = [
    "<|pad|>",
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image|>",
    "<|video|>",
]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% for c in m.content %}{% if c.type == 'image' %}"
    "<|vision_start|><|image|><|vision_end|>{% elif c.type == 'video' %}<|video|>"
    "{% else %}{{ c.text }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
)
TICKET = "I was charged twice and my parcel never arrived. Please refund one charge."
INVOICE = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)"
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "shipping": "Delivery and parcels",
            "technical": "App or site faults",
        },
    },
    "topics": {
        "type": "choice",
        "multiple": True,
        "instructions": "Which topics does the message raise?",
        "criteria": {
            "billing": "Payments and charges",
            "delivery": "Shipping and parcels",
            "login": "Signing in to the account",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["low", "medium", "high"],
    },
}
VALUES = {
    "payee": {"type": "text", "instructions": "Who issued the invoice?"},
    "total": {"type": "number", "instructions": "How much is due?"},
    "due": {"type": "date", "instructions": "By when must it be paid?"},
    "vat": {"type": "number", "instructions": "How much VAT is charged?"},
}


@pytest.fixture(scope="module")
def tokenizer():
    """A small byte-level BPE with LFM3's pre-tokenizer and special tokens, trained
    so every number 0-999, yes/no and the slot names are single tokens."""
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PATTERN), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    words = [str(i) for i in range(1000)] + [a + b for a in letters for b in letters]
    words += "yes Yes no No YES NO :\n start end present top bottom left right".split(
        " "
    )
    corpus = words * 3 + [TICKET, INVOICE, json.dumps(QUESTIONS)] * 3
    trainer = trainers.BpeTrainer(
        vocab_size=4000,
        min_frequency=1,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer)
    tok.add_tokens([AddedToken("<think>"), AddedToken("</think>")])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )


def d1_tokenizer(tokenizer, numbers=True):
    tok = copy.copy(tokenizer)
    tok.d1_codes, tok.d1_numbers, tok.d1_box_edges, tok.d1_prompt = (
        2,
        numbers,
        "median",
        1,
    )
    return tok


def test_tokenizer_writes_what_the_readouts_need(tokenizer):
    tok = d1_tokenizer(tokenizer)
    assert prompt.number_ids(tok, box.THOUSANDTHS, 1000) is not None
    assert len(prompt.readout_ids(tok, prompt.Noul("q"))) == 2
    assert box.edge_grid(tok) == (box.THOUSANDTHS, 1000)


def test_prefix_and_suffix_render_as_the_reference(tokenizer):
    tok = d1_tokenizer(tokenizer)
    state = {"note": "ends <|im_end|> here"}
    prefix = prompt.prefix_text(tok, state, tok.bos_token)
    # the state's special-token text is broken by a zero-width space, three newlines before QUESTION
    assert prefix == (
        '<|startoftext|><|im_start|>user\n{\n  "note": "ends <​|im_end|> here"\n}'
        "\n\n\nQUESTION:\n"
    )
    assert prompt.prefix_text(tok, None, "", media="<m>") == "<|im_start|>user\n<m>"
    text, slots = prompt.suffix_layout(tok, prompt.Noul("Is it?"), d1.LEAD)
    assert text == (
        "Is it?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    assert slots == [0]
    text, slots = prompt.suffix_layout(tok, prompt.Read("Who?"), d1.LEAD)
    assert text.endswith("<think></think>start:\nend:")
    ids = tok.encode(text, add_special_tokens=False)
    assert [tok.decode(ids[len(ids) - 1 + o]).strip() for o in slots] == [":", ":"]
    assert slots[-1] == 0 and slots[0] < 0


def test_pack_set_groups_and_slots(tokenizer):
    tok = d1_tokenizer(tokenizer)
    prefix = prompt.prefix_text(tok, TICKET, tok.bos_token)
    qs = [prompt.as_question(q) for q in QUESTIONS.values()]
    layouts = [prompt.suffix_layout(tok, q, d1.LEAD) for q in qs]
    suffixes = [t for t, _ in layouts]
    first = tok.encode(prefix + suffixes[0], add_special_tokens=False)
    ids, state, groups, ends = d1.pack_set(
        tok, prefix, suffixes, "QUESTION:\n", [o for _, o in layouts], first
    )
    # the first question shares the prefix's encoding; the others follow, header first
    assert state == len(tok.encode(prefix, add_special_tokens=False))
    assert groups[:state] == [-1] * state
    lengths = [len(tok.encode(suffixes[0], add_special_tokens=False))] + [
        len(tok.encode("QUESTION:\n" + s, add_special_tokens=False))
        for s in suffixes[1:]
    ]
    assert [groups.count(k) for k in range(1, 5)] == lengths
    assert len(ids) == state + sum(lengths)
    assert [e[-1] for e in ends] == list(np.cumsum(lengths) + state - 1)
    assert len(ends[2]) == 3  # a multiple choice reads one slot per option
    assert tok.decode(ids[ends[0][0]]) == "</think>"


def test_set_masks_truth_table():
    # [state 3][q1 2][q2 2]
    masks = d1.set_masks(7, 3, [-1, -1, -1, 1, 1, 2, 2], window=2)
    full = np.array(masks["full_attention"])
    expected = np.ones((7, 7), bool)
    expected[3:5, 5:7] = expected[5:7, 3:5] = False
    assert (full == expected).all()
    sliding = np.array(masks["sliding_attention"])
    q, k = np.arange(7)[:, None], np.arange(7)[None]
    assert (sliding == ((k <= q) & (k > q - 2))).all()
    assert masks["conv"] is None
    assert d1.set_masks(4, 4, [-1] * 4, window=8)["sliding_attention"] == "causal"


def test_calibration_through_and_level():
    knots = [[0.2, 0.1], [0.6, 0.5], [0.9, 0.95]]
    assert through(0.0, knots) == 0.1 and through(1.0, knots) == 0.95
    assert math.isclose(through(0.4, knots), 0.3)
    assert through(0.6, knots) == 0.5
    cal = Calibration(default=2.0, tables={"estimate": knots})
    assert math.isclose(cal.level("estimate", 0.3), 0.4)
    assert cal.apply(prompt.Noul("q"), [2.0, 4.0]) == [1.0, 2.0]


def test_pointer_spans():
    logits = np.full((1, 2, 8), -10.0, np.float32)
    logits[0, 0, 2] = logits[0, 1, 4] = 10.0
    at, found = pointer.spans(logits, max_tokens=4)
    assert at.tolist() == [[2, 4]] and found[0] > 0.99
    at, _ = pointer.spans(logits, max_tokens=2)  # the end is out of reach
    assert at[0, 1] - at[0, 0] < 2
    logits[0, 0, 0] = 30.0  # "nothing" at BOS
    at, found = pointer.spans(logits, max_tokens=4)
    assert at.tolist() == [[0, 0]] and found[0] < 0.5
    ties = np.zeros((1, 2, 5), np.float32)
    ties[0, :, 0] = -np.inf
    assert pointer.spans(ties, max_tokens=4)[0].tolist() == [[1, 1]]


def test_state_positions_read_json_paths(tokenizer):
    tok = d1_tokenizer(tokenizer)
    state = {"name": 'Zoe "Z" Muller', "city": "Bern"}
    prefix = prompt.prefix_text(tok, state, tok.bos_token)
    text = prefix + "x"
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = pointer.state_positions(tok, state, prefix, text, encoded=encoded)
    start, end, path = positions.paths[0]
    assert path == "/name" and positions.text[start:end] == 'Zoe \\"Z\\" Muller'
    held = [
        i
        for i, s in enumerate(positions.offsets)
        if s is not None and s[0] < end and start < s[1]
    ]
    # a span over the string's tokens reads the caller's characters, not the escapes
    read = positions.read(held[0], held[-1])
    assert read["path"] == "/name" and read["text"] == 'Zoe "Z" Muller'
    assert positions.offsets[0] is None and positions.pointer_mask[0]


def one_hot(index, size=1000):
    row = [0.0] * size
    row[index] = 1.0
    return row


def test_box_answer_and_boxes_order():
    edges = [one_hot(100), one_hot(300), one_hot(200), one_hot(600)]
    out = box.answer([[0.9, 0.1], *edges])
    assert out["box"] == pytest.approx(
        {"top": 0.1, "bottom": 0.3, "left": 0.2, "right": 0.6}
    )
    assert out["confidence"] == 1.0 and out["found"] == 0.9
    assert box.answer([[0.2, 0.8], *edges])["box"] is None
    second = [one_hot(100), one_hot(300), one_hot(50), one_hot(150)]
    blocks = [[0.9, 0.1], *edges, [0.95, 0.05], *second, [0.4, 0.6], *edges]
    many = box.answer_many(blocks)
    assert many["count"] == 2 and not many["at_least"]
    lefts = [b["box"]["left"] for b in many["boxes"]]
    assert lefts == pytest.approx([0.05, 0.2])  # reading order
    assert many["presence"] == [0.9, 0.9, 0.4]  # kept decreasing


def test_score_readouts():
    probs = [0.3, 0.15, 0.55]
    assert api.score_level(probs, "median") == 2
    assert api.score_level(probs, "argmax") == 2
    q = prompt.Score("q", ["a", "b", "c"])
    assert api.answer(q, probs, "mean")["score"] == pytest.approx(1.25)
    assert api.answer(q, [0.5, 0.3, 0.2])["score"] == 0.0
    choice = prompt.Choice("q", {"a": None, "b": None, "idk": "I don't know"})
    assert api.answer(choice, [0.4, 0.35, 0.25])["choice"] == "a"
    assert api.answer(choice, [0.4, 0.35, 0.25], abstain_below=0.6)["choice"] == "idk"


def test_estimate_reads_calibrated_levels():
    q = prompt.as_question(
        {"type": "number", "instructions": "q", "range": [0, 10], "step": 1}
    )
    assert q.grid == Grid(0, 10, 11)
    probs = [0.0, 0.1, 0.5, 0.4] + [0.0] * 7
    out = api.answer(q, probs)
    assert out["number"] == 2.0 and out["interval"] == [1.0, 3.0]


def test_resampled_keeps_a_tone():
    rate, target = 44100, 16000
    t = np.arange(rate) / rate
    out = resampled(np.sin(2 * np.pi * 440 * t).astype(np.float32), rate, target)
    assert out.dtype == np.float32 and len(out) == target
    expect = np.sin(2 * np.pi * 440 * np.arange(target) / target)
    assert np.abs(out[200:-200] - expect[200:-200]).max() < 1e-2
    stereo = np.stack([out, out])
    assert np.array_equal(resampled(stereo, target, target), out)


def tiny_case(name, tokenizer):
    cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())
    config = next(c for c in cases["cases"] if c["id"] == name)["config"]
    config["text_config"]["vocab_size"] = len(tokenizer)
    config["image_token_id"] = tokenizer.convert_tokens_to_ids("<|image|>")
    config["video_token_id"] = tokenizer.convert_tokens_to_ids("<|video|>")
    config.update(
        d1_attention="set", d1_set_size=4, d1_codes=2, d1_numbers=True, d1_prompt=None
    )
    return config


def tiny_processor(tokenizer):
    from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import (
        Lfm3VlProcessor,
        Lfm3VlVideoProcessor,
    )
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    pixels = dict(patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    tokenizer.chat_template = TEMPLATE
    return Lfm3VlProcessor(
        Qwen3VLImageProcessor(min_pixels=64, max_pixels=1024, **pixels),
        tokenizer,
        Lfm3VlVideoProcessor(min_pixels=64, max_pixels=4096, min_frames=2, **pixels),
        chat_template=TEMPLATE,
    )


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    from mlx_vlm.models import d1_lfm3_vl

    mx.random.seed(7)
    model = d1_lfm3_vl.Model(
        d1_lfm3_vl.ModelConfig.from_dict(tiny_case("lfm3_vl", tokenizer))
    )
    model.eval()
    tables = {
        "found:text": [[0.0, 0.0], [1.0, 1.0]],
        "multiple": [[0.0, 0.1], [1.0, 0.9]],
    }
    model.attach_processor(
        tiny_processor(tokenizer), Calibration(default=1.3, tables=tables)
    )
    return model


def check_probabilities(p):
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-5)


def test_tiny_model_answers_every_type(tiny_model):
    questions = {
        **QUESTIONS,
        **VALUES,
        "late": {
            "type": "number",
            "instructions": "How late?",
            "range": [0, 2],
            "step": 0.5,
        },
    }
    out = tiny_model.system_one(INVOICE, questions)
    answers = out["answers"]
    assert list(answers) == list(questions)
    assert 0.0 <= answers["refund"]["noul"] <= 1.0
    assert answers["team"]["choice"] in QUESTIONS["team"]["criteria"]
    check_probabilities(answers["team"]["probabilities"])
    assert set(answers["topics"]["probabilities"]) == set(
        QUESTIONS["topics"]["criteria"]
    )
    assert answers["urgency"]["score"] in (0.0, 1.0, 2.0)
    check_probabilities(answers["urgency"]["probabilities"])
    for name in VALUES:
        assert set(answers[name]) >= {"type", "span", "found", VALUES[name]["type"]}
    assert answers["late"]["number"] in (0.0, 0.5, 1.0, 1.5, 2.0)
    assert out["usage"]["output_tokens"] == 0
    alone = tiny_model.system_one(INVOICE, questions, set_size=1)
    # each group repeats the state
    assert alone["usage"]["input_tokens"] > out["usage"]["input_tokens"]
    batch = tiny_model.system_one_batch([(TICKET, QUESTIONS), (INVOICE, VALUES)])
    assert batch[0]["answers"] == tiny_model.system_one(TICKET, QUESTIONS)["answers"]
    with pytest.raises(ValueError, match="score_readout"):
        tiny_model.system_one(TICKET, QUESTIONS, score_readout="mode")
    with pytest.raises(ValueError, match="'oops'"):
        tiny_model.system_one(TICKET, {"oops": {"type": "noul"}})


def test_tiny_model_reads_pictures_and_video(tiny_model):
    from PIL import Image

    picture = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (20, 28, 3), np.uint8)
    )
    questions = {
        "count": {
            "type": "number",
            "instructions": "How many?",
            "range": [0, 4],
            "step": 1,
        },
        "where": {"type": "box", "instructions": "Where?"},
        "all": {"type": "boxes", "instructions": "Every one.", "max": 2},
    }
    out = tiny_model.system_one(None, questions, media=[picture])
    where, every = out["answers"]["where"], out["answers"]["all"]
    assert set(where) == {"type", "box", "found", "confidence"}
    assert set(every) == {"type", "boxes", "count", "at_least", "presence"}
    assert len(every["presence"]) == 2
    frames = np.random.default_rng(1).integers(0, 255, (4, 16, 16, 3), np.uint8)
    out = tiny_model.system_one(
        "A clip.",
        {"refund": QUESTIONS["refund"]},
        media=[{"type": "video", "video": frames, "fps": 2.0}],
    )
    assert 0.0 <= out["answers"]["refund"]["noul"] <= 1.0
    with pytest.raises(ValueError, match="fps"):
        tiny_model.system_one(
            None, {"r": QUESTIONS["refund"]}, media=[{"type": "video", "video": frames}]
        )
    with pytest.raises(ValueError, match="exactly one picture"):
        tiny_model.system_one(None, {"where": questions["where"]})


def test_tiny_model_refuses_generation(tiny_model):
    with pytest.raises(ValueError, match="system_one"):
        tiny_model(mx.array([[1, 2, 3]]))
    with pytest.raises(ValueError, match="system_one"):  # how generate prefills
        tiny_model.get_input_embeddings(mx.array([[1, 2, 3]]))


def test_decide_cli(tiny_model, monkeypatch, tmp_path, capsys):
    from mlx_vlm import decide

    monkeypatch.setattr(decide, "load", lambda _: (tiny_model, None))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"note": TICKET}))
    refund = json.dumps({"refund": QUESTIONS["refund"]})
    decide.main(["--model", "m", "--state-file", str(state), "--questions", refund])
    assert 0.0 <= json.loads(capsys.readouterr().out)["answers"]["refund"]["noul"] <= 1
    missing = str(tmp_path / "no.jpg")
    with pytest.raises(SystemExit) as exit_info:
        decide.main(["--model", "m", "--image", missing, "--questions", refund])
    assert exit_info.value.code == 2
    assert "no.jpg" in capsys.readouterr().err


def test_tiny_omni_answers_text(tokenizer):
    from mlx_vlm.models import d1_lfm3_omni

    config = tiny_case("lfm3_omni", tokenizer)
    config["audio_token_id"] = config["video_token_id"] + 1
    config["audio_out_token_id"] = config["video_token_id"] + 2
    model = d1_lfm3_omni.Model(d1_lfm3_omni.ModelConfig.from_dict(config))
    model.attach_processor(tiny_processor(tokenizer))
    out = model.system_one(TICKET, QUESTIONS)
    assert out["answers"]["team"]["choice"] in QUESTIONS["team"]["criteria"]
    with pytest.raises(ValueError, match="system_one"):
        model.generate_audio([1, 2])


def test_config_defaults_and_checks(tokenizer):
    from mlx_vlm.models import d1_lfm3_vl

    config = tiny_case("lfm3_vl", tokenizer)
    for key in ("d1_set_size", "d1_codes", "d1_numbers"):
        config.pop(key)
    loaded = d1_lfm3_vl.ModelConfig.from_dict({**config, "d1_box_edges": None})
    assert (loaded.d1_set_size, loaded.d1_codes, loaded.d1_numbers) == (4, 1, False)
    assert (loaded.d1_prompt, loaded.d1_box_edges) == (1, "median")
    assert loaded.d1_vision_max_pixels == 1024 * 1024
    with pytest.raises(ValueError, match="set"):
        d1_lfm3_vl.ModelConfig.from_dict({**config, "d1_attention": "causal"})
    with pytest.raises(ValueError, match="d1_set_size"):
        d1_lfm3_vl.ModelConfig.from_dict({**config, "d1_set_size": 129})


def real(env):
    path = os.environ.get(env)
    return pytest.mark.skipif(not path, reason=f"set {env} to a local checkpoint")


@real("D1_PATH")
def test_d1_2b_card_answers():
    from mlx_vlm import load

    model, _ = load(os.environ["D1_PATH"])
    out = model.system_one(TICKET, QUESTIONS)
    a = out["answers"]
    assert round(a["refund"]["noul"], 2) == 1.0
    assert a["team"]["choice"] == "billing"
    assert a["topics"]["choices"] == ["billing", "delivery"]
    assert a["urgency"]["legend"][str(int(a["urgency"]["score"]))] == "medium"
    assert out["usage"]["input_tokens"] == 189
    a = model.system_one(INVOICE, VALUES)["answers"]
    assert [a[k][k2] for k, k2 in [("payee", "text"), ("total", "number")]] == [
        "ACME SA",
        1234.5,
    ]
    assert a["due"]["date"] == "2026-04-03" and a["vat"]["number"] is None


@real("D1_OMNI_PATH")
def test_d1_omni_2b_card_answers():
    from mlx_vlm import load

    model, _ = load(os.environ["D1_OMNI_PATH"])
    questions = {k: QUESTIONS[k] for k in ("refund", "team", "topics")}
    a = model.system_one(TICKET, questions)["answers"]
    assert round(a["refund"]["noul"], 2) == 0.94
    assert a["team"]["choice"] == "billing"
    assert a["topics"]["choices"] == ["billing", "delivery"]
    rate = 16000
    clip = np.random.default_rng(0).normal(0, 0.003, 4 * rate).astype(np.float32)
    for start in (0.5, 1.5, 2.5):
        n = np.arange(3200)
        clip[int(start * rate) : int(start * rate) + 3200] += 0.3 * np.sin(
            2 * np.pi * 880 * n / rate
        )
    out = model.system_one(
        None,
        {
            "beeps": {
                "type": "number",
                "instructions": "How many beeps are there?",
                "range": [0, 10],
                "step": 1,
            },
            "speech": {
                "type": "noul",
                "instructions": "Does someone speak in the recording?",
            },
        },
        media=[{"type": "audio", "audio": clip, "sampling_rate": rate}],
    )
    assert out["answers"]["beeps"]["number"] == 3.0
    assert out["answers"]["speech"]["noul"] < 0.5
    assert out["usage"]["input_tokens"] == 113
