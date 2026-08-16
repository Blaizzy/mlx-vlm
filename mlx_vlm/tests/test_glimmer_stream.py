"""Stream-loop tests for MuseGlimmerStreamState (the single state machine).

These exercise the exact endpoint pattern — chunk-by-chunk feed — and cover
every regression found across review rounds:
- thinking-only turns (answer must not be dropped)
- quoted "to=self<|message|>" and "to=user<|message|>" mid-answer
- split markers / headers across chunk boundaries
- end-marker tails must not leak
- multiple tool blocks + final answer
- multi-chunk plain prose must not be glued
"""

from mlx_vlm.server.glimmer_stream import MuseGlimmerStreamState


class FakeGlimmerProcessor:
    pass


FakeGlimmerProcessor.__module__ = "mlx_vlm.models.muse_glimmer.processing_muse_glimmer"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

TOOL_BLOCK = (
    "<atem:function_calls>\n"
    '<atem:invoke name="bash">\n'
    '<atem:parameter name="command">ls</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


def _stream(text, size=1, state=None):
    """Feed ``text`` in chunks of ``size``; return (reasoning, content)."""
    state = state or MuseGlimmerStreamState()
    reasoning, content = [], []
    for i in range(0, len(text), size):
        d = state.feed(text[i : i + size], last=(i + size >= len(text)))
        if d.reasoning:
            reasoning.append(d.reasoning)
        if d.content:
            content.append(d.content)
    return "".join(reasoning), "".join(content)


# --- thinking + answer ---------------------------------------------------


def test_answer_header_stripped():
    r, c = _stream("to=user<|message|>The result is 42.")
    assert r == ""
    assert c == "The result is 42."


def test_thinking_then_answer():
    r, c = _stream("to=self<|message|>plan<|eom|>to=user<|message|>blue")
    assert r == "plan"
    assert c == "blue"


def test_thinking_only_turn_answer_not_dropped():
    r, c = _stream(
        "to=self<|message|>The user wants a color. Blue is typical.<|eom|>"
        "to=user<|message|>blue"
    )
    assert r == "The user wants a color. Blue is typical."
    assert c == "blue"


def test_multiple_thinking_blocks():
    r, c = _stream(
        "to=self<|message|>first<|eom|>to=self<|message|>second<|eom|>"
        "to=user<|message|>answer"
    )
    assert "first" in r and "second" in r
    assert c == "answer"


# --- tool calls ----------------------------------------------------------


def test_tool_block_fully_suppressed():
    r, c = _stream("to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>done")
    assert "<atem" not in c
    assert "done" in c


def test_tool_block_then_multichunk_answer():
    text = (
        "to=self<|message|>check<|eom|>"
        "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>The result is 42."
    )
    r, c = _stream(text, size=7)
    assert r == "check"
    assert "The result is 42." in c
    assert "<atem" not in c and "to=" not in c


def test_two_tool_blocks_then_answer():
    block2 = TOOL_BLOCK.replace(">ls</", ">pwd</")
    text = (
        "to=self<|message|>plan<|eom|>"
        "to=bash<|message|>"
        + TOOL_BLOCK
        + "<|eom|>"
        + block2
        + "to=user<|message|>Done. Both ran."
    )
    r, c = _stream(text, size=9)
    assert r == "plan"
    assert "Done. Both ran." in c
    assert "<atem" not in c


# --- quoted markers must survive -----------------------------------------


def test_quoted_to_self_mid_answer():
    text = (
        "to=self<|message|>first<|eom|>"
        "to=user<|message|>The routing header is to=self<|message|> and it "
        "opens thinking"
    )
    r, c = _stream(text, size=5)
    assert "first" in r
    assert "The routing header is to=self<|message|> and it opens thinking" in c


def test_quoted_to_user_header_mid_answer():
    text = "to=user<|message|>The header to=user<|message|> is quoted text."
    r, c = _stream(text, size=4)
    assert "The header to=user<|message|> is quoted text." in c


def test_quoted_tool_start_mid_answer():
    text = "to=user<|message|>The doc says <atem:function_calls> is the tag. Done."
    r, c = _stream(text, size=6)
    assert "The doc says <atem:function_calls> is the tag. Done." in c


# --- chunk-boundary splits ------------------------------------------------


def test_split_header_across_chunks():
    text = "to=user<|message|>hello world"
    r, c = _stream(text, size=3)
    assert c == "hello world"


def test_split_to_self_quoted_across_chunks():
    text = (
        "to=self<|message|>first<|eom|>"
        "to=user<|message|>The marker to=self<|message|> is quoted."
    )
    r, c = _stream(text, size=7)
    assert "first" in r
    assert "The marker to=self<|message|> is quoted." in c


def test_split_end_marker_no_tail_leak():
    text = "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>final answer"
    r, c = _stream(text, size=5)
    assert "final answer" in c
    # No partial "</atem:function_calls>" tail may survive.
    assert "nction_calls>" not in c
    assert "alls>" not in c
    assert "s>" not in c


def test_split_tool_start_handled():
    text = "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>answer"
    r, c = _stream(text, size=4)
    assert "answer" in c
    assert "<atem" not in c


# --- prose integrity -------------------------------------------------------


def test_multichunk_prose_not_glued():
    text = "to=user<|message|>The color is blue and it looks nice."
    r, c = _stream(text, size=4)
    assert c == "The color is blue and it looks nice."


def test_indentation_preserved():
    text = "to=user<|message|>def f():\n    return 1"
    r, c = _stream(text, size=3)
    assert "def f():\n    return 1" in c


def test_stray_eom_becomes_space():
    text = "to=user<|message|>answer one<|eom|>answer two"
    r, c = _stream(text)
    assert "answer one answer two" in c


def test_last_flag_flushes_held_text():
    text = "to=user<|message|>the answer ends with to"
    r, c = _stream(text)
    assert c == "the answer ends with to"


# --- regression tests from round-4 review ---------------------------------


def test_delta_has_thinking_closed_field():
    """Anthropic endpoint reads thinking_closed on every delta."""
    state = MuseGlimmerStreamState()
    d1 = state.feed("to=self<|message|>plan")
    assert d1.thinking_closed is False
    d2 = state.feed("<|eom|>to=user<|message|>blue", last=True)
    assert d2.thinking_closed is True


def test_eom_then_tool_block_header_stripped():
    """<|eom|> in normal state is a routing boundary: a following header or
    tool block must be recognized (not leak into content)."""
    text = (
        "to=user<|message|>answer one<|eom|>"
        "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>done"
    )
    r, c = _stream(text, size=7)
    assert "answer one" in c
    assert "done" in c
    assert "<atem" not in c
    assert "to=bash" not in c


def test_eom_then_user_header():
    text = "to=user<|message|>answer one<|eom|>to=user<|message|>answer two"
    r, c = _stream(text, size=5)
    assert "answer one" in c
    assert "answer two" in c
    assert "to=user" not in c


def test_newline_after_tool_end_header_stripped():
    text = "to=bash<|message|>" + TOOL_BLOCK + "\nto=user<|message|>done"
    r, c = _stream(text, size=7)
    assert "done" in c
    assert "to=user" not in c


def test_multiple_thinking_blocks_not_glued():
    text = (
        "to=self<|message|>alpha<|eom|>"
        "to=self<|message|>beta<|eom|>"
        "to=user<|message|>answer"
    )
    r, c = _stream(text, size=11)
    assert "alpha" in r and "beta" in r
    assert "\n" in r  # blocks separated, not "alphabeta"


def test_response_output_items_from_text_glimmer():
    """The responses-endpoint final parse must strip glimmer headers from
    post-tool remaining text."""
    from mlx_vlm.server.responses_state import _response_output_items_from_text
    from mlx_vlm.tool_parsers import atem

    full_text = (
        "to=self<|message|>check<|eom|>"
        "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>The result is 42."
    )
    items, clean_text, reasoning, finish = _response_output_items_from_text(
        full_text,
        "msg_1",
        atem,
        [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        {},
        processor=FakeGlimmerProcessor(),
    )
    assert "The result is 42." in clean_text
    assert "to=user" not in clean_text
    assert "to=bash" not in clean_text


def test_process_tool_calls_quoted_to_self_in_answer_survives():
    """A quoted to=self<|message|> in the post-tool answer must not be
    truncated by the envelope cleanup."""
    from mlx_vlm.server.responses_state import process_tool_calls
    from mlx_vlm.tool_parsers import atem

    out = (
        "to=self<|message|>check<|eom|>"
        "to=bash<|message|>"
        + TOOL_BLOCK
        + "to=user<|message|>the marker to=self<|message|> is quoted"
    )
    tc = process_tool_calls(out, atem, TOOLS)
    assert "the marker to=self<|message|> is quoted" in tc["remaining_text"]


# --- round-5 regression tests ---------------------------------------------


def test_full_form_header_stripped():
    """<|start|>assistant to=<name><|message|> (the template's full form)
    must be stripped in the stream, not leaked."""
    text = "<|start|>assistant to=user<|message|>hello world"
    r, c = _stream(text, size=5)
    assert "hello world" in c
    assert "<|start|>" not in c
    assert "to=user" not in c


def test_full_form_header_before_tool_block():
    text = (
        "<|start|>assistant to=bash<|message|>"
        + TOOL_BLOCK
        + "<|start|>assistant to=user<|message|>done"
    )
    r, c = _stream(text, size=7)
    assert "done" in c
    assert "<atem" not in c
    assert "<|start|>" not in c


def test_eom_separator_preserved_exact():
    """The <|eom|> separator space is content, not dropped (exact text)."""
    text = "to=user<|message|>answer one<|eom|>answer two"
    r, c = _stream(text, size=5)
    assert c == "answer one answer two"


def test_eom_then_header_exact():
    text = "to=user<|message|>answer one<|eom|>to=user<|message|>answer two"
    r, c = _stream(text, size=5)
    assert c == "answer one answer two"


def test_eom_then_plain_to_word_not_routing():
    """ "answer one<|eom|>to the store" — "to " (no =) is not routing."""
    text = "to=user<|message|>answer one<|eom|>to the store"
    r, c = _stream(text, size=3)
    assert c == "answer one to the store"


def test_stream_final_consistency_quoted_header():
    """Streamed and final output must agree: quoted to=user header survives
    in both."""
    from mlx_vlm.server.responses_state import _split_thinking

    text = "to=user<|message|>the marker to=user<|message|> is quoted"
    r, c = _stream(text, size=4)
    assert "the marker to=user<|message|> is quoted" in c
    _, fc = _split_thinking(text, processor=FakeGlimmerProcessor())
    assert "the marker to=user<|message|> is quoted" in fc


def test_eot_stripped():
    text = "to=user<|message|>hello<|eot|>"
    r, c = _stream(text, size=3)
    assert "<|eot|>" not in c
    assert "hello" in c


def test_multi_thinking_blocks_single_newline():
    text = (
        "to=self<|message|>alpha<|eom|>"
        "to=self<|message|>beta<|eom|>"
        "to=user<|message|>answer"
    )
    r, c = _stream(text, size=3)
    assert "alpha\nbeta" in r, f"got {r!r}"
    assert "\n\n" not in r


# --- round-6 regression tests ---------------------------------------------


def test_whitespace_then_marker_after_eom():
    """Whitespace arriving before a routing marker must not flip the segment
    flag (newline token before to=self re-open)."""
    text = (
        "to=self<|message|>a<|eom|>\n\n"
        "to=self<|message|>b<|eom|>"
        "to=user<|message|>c"
    )
    r, c = _stream(text, size=1)
    assert "a" in r and "b" in r
    assert c.strip() == "c"
    assert "to=self" not in c and "to=user" not in c


def test_whitespace_then_header_after_tool_end():
    text = "to=bash<|message|>" + TOOL_BLOCK + "\nto=user<|message|>done"
    r, c = _stream(text, size=1)
    assert "done" in c
    assert "to=user" not in c
    assert "<atem" not in c


def test_eom_then_header_final_parse_consistency():
    """The final parse must strip a header after <|eom|> (stream does)."""
    from mlx_vlm.server.responses_state import _split_thinking

    text = "to=user<|message|>answer one<|eom|>to=user<|message|>answer two"
    r, c = _stream(text, size=5)
    assert c == "answer one answer two"
    _, fc = _split_thinking(text, processor=FakeGlimmerProcessor())
    assert "to=user" not in fc
    assert "answer one" in fc and "answer two" in fc


def test_truncated_thinking_final_parse_is_content():
    """Truncated reasoning (no <|eom|>) goes to content in the FINAL parse
    (the stored message). The stream may show live reasoning deltas, but the
    authoritative response must be content."""
    from mlx_vlm.server.responses_state import _split_thinking

    text = "to=self<|message|>planning cut off"
    _, fc = _split_thinking(text, processor=FakeGlimmerProcessor())
    assert "planning cut off" in fc
    r, c = _split_thinking(text, processor=FakeGlimmerProcessor())
    assert r is None


def test_eot_split_across_chunks():
    text = "to=user<|message|>hello<|eot|>"
    r, c = _stream(text, size=1)
    assert "<|eot|>" not in c
    assert "hello" in c


def test_eom_no_double_space():
    text = "to=user<|message|>answer one <|eom|>answer two"
    r, c = _stream(text, size=3)
    assert "answer one answer two" in c
    assert "  " not in c


def test_flush_strips_partial_eom():
    text = "to=self<|message|>plan<|eom"
    r, c = _stream(text, size=3)
    assert "<|eom" not in c


# --- round-7 regression tests ---------------------------------------------


def test_full_form_header_size1():
    """The full-form header must not leak at size-1 chunking (feed boundary
    after the "<|start|>assistant " prefix)."""
    text = "<|start|>assistant to=user<|message|>hello world"
    r, c = _stream(text, size=1)
    assert "hello world" in c
    assert "<|start|>" not in c
    assert "to=user" not in c


def test_full_form_header_before_tool_size1():
    text = (
        "<|start|>assistant to=bash<|message|>"
        + TOOL_BLOCK
        + "<|start|>assistant to=user<|message|>done"
    )
    r, c = _stream(text, size=1)
    assert "done" in c
    assert "<atem" not in c
    assert "<|start|>" not in c


def test_post_tool_eom_header_final_parse():
    """The final parse must strip a header after <|eom|> in the post-tool
    remaining text (matching the stream)."""
    from mlx_vlm.server.responses_state import process_tool_calls
    from mlx_vlm.tool_parsers import atem

    out = (
        "to=self<|message|>check<|eom|>"
        "to=bash<|message|>" + TOOL_BLOCK + "to=user<|message|>answer one<|eom|>"
        "to=user<|message|>answer two"
    )
    tc = process_tool_calls(out, atem, TOOLS)
    assert "answer one" in tc["remaining_text"]
    assert "answer two" in tc["remaining_text"]
    # The full pipeline (final parse) must yield clean text.
    from mlx_vlm.server.responses_state import _split_thinking

    _, final_content = _split_thinking(
        tc["remaining_text"], processor=FakeGlimmerProcessor()
    )
    assert "answer one answer two" in final_content
    assert "to=user" not in final_content


def test_truncated_marker_final_parse():
    """Truncated markers (generation cut mid-marker) must not leak into the
    final parse content."""
    from mlx_vlm.server.responses_state import _split_thinking

    _, fc = _split_thinking(
        "to=self<|message|>plan<|eom", processor=FakeGlimmerProcessor()
    )
    assert "<|eom" not in fc
    _, fc2 = _split_thinking(
        "to=user<|message|>hello<|eot", processor=FakeGlimmerProcessor()
    )
    assert "<|eot" not in fc2


def test_quoted_self_pair_mid_answer_final_parse():
    """A complete quoted "to=self<|message|>…<|eom|>" pair mid-answer must
    survive in the final parse (not be collected as reasoning)."""
    from mlx_vlm.server.responses_state import _split_thinking

    text = (
        "to=self<|message|>blockone<|eom|>"
        "to=user<|message|>the marker to=self<|message|> is <|eom|>quoted"
    )
    r, c = _split_thinking(text, processor=FakeGlimmerProcessor())
    assert "blockone" in (r or "")
    # The quoted pair survives as content (the <|eom|> is a continuation
    # marker and becomes a space); it must NOT be collected as reasoning.
    assert "the marker to=self<|message|> is quoted" in c
    assert "blockone" not in c
