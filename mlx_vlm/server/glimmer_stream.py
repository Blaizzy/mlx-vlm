"""Muse Glimmer streaming state machine.

Single self-contained state machine for Glimmer's recipient-routing output,
replacing the ``ThinkingStreamState`` + ``suppress_tool_call_content`` pairing
(which disagreed about stream state). Owns ALL region logic in one buffer:

  NORMAL   --"to=self<|message|>" at a routing boundary--> THINKING
  NORMAL   --"<atem:function_calls>"-->                   TOOL
  THINKING --"<|eom|>"-->                                 NORMAL
  TOOL     --"</atem:function_calls>"-->                  NORMAL

Rules (each learned from review-found regressions):

* Content is only emitted once it is unambiguous — partial markers/headers
  are held in the buffer until resolved (or flushed at ``last=True``).
* ``to=<name><|message|>`` headers are stripped only at segment starts
  (beginning of the response, or right after a close marker). A *quoted*
  header mid-answer survives.
* ``to=self<|message|>`` only re-opens thinking at a routing boundary;
  quoted occurrences mid-answer flow as content.
* The tool region suppresses everything, and the end marker is consumed
  position-aware so a marker split across chunks never leaks its tail.
* ``<|eom|>`` is a continuation boundary: substituted with a single space.

The final response is recomputed from the full text by ``_split_thinking`` /
``process_tool_calls``; this class only shapes the streamed deltas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_HEADER_RE = re.compile(
    r"^\s*(?:<\|start\|>assistant )?to=[A-Za-z0-9_.-]+<\|message\|>"
)
_GLIMMER_EOM = "<|eom|>"
_GLIMMER_SELF_OPEN = "to=self<|message|>"
_GLIMMER_TOOL_START = "<atem:function_calls>"
_GLIMMER_TOOL_END = "</atem:function_calls>"
_GLIMMER_EOT = "<|eot|>"


@dataclass
class GlimmerStreamDelta:
    reasoning: Optional[str] = None
    content: Optional[str] = None
    thinking_closed: bool = False


class MuseGlimmerStreamState:
    """Streaming state machine for Muse Glimmer recipient-routing output."""

    def __init__(self):
        self.buffer = ""
        # "normal" | "thinking" | "tool"
        self.state = "normal"
        # True when the next content is at a routing segment start (start of
        # stream, or right after a close marker). Only at a segment start may
        # a "to=<name><|message|>" header be stripped or a "to=self" block
        # open; quoted occurrences mid-answer must survive.
        self.segment_started = True
        # Whether reasoning was emitted in a previous feed call — used to
        # separate consecutive thinking blocks ("a\nb", not "ab").
        self._reasoning_emitted = False
        self._pending_reasoning_sep = False
        # Last content char emitted (for separator dedup across feed calls).
        self._last_content_end = ""

    # -- public ------------------------------------------------------------
    def feed(self, text: str, last: bool = False) -> GlimmerStreamDelta:
        """Consume a chunk of decoded text; return clean deltas."""
        self.buffer += text or ""
        reasoning: List[str] = []
        content: List[str] = []
        thinking_closed = False

        while self.buffer:
            if self.state == "thinking":
                idx = self.buffer.find(_GLIMMER_EOM)
                if idx < 0:
                    emit, held = self._split_partial(
                        self.buffer, (_GLIMMER_EOM, _GLIMMER_TOOL_START)
                    )
                    if emit:
                        self._append_reasoning(reasoning, emit)
                    self.buffer = held
                    # Either everything was emitted, or the tail is a held
                    # partial marker awaiting more input — wait either way.
                    break
                if idx:
                    self._append_reasoning(reasoning, self.buffer[:idx])
                self.buffer = self.buffer[idx + len(_GLIMMER_EOM) :].lstrip("\n")
                self.state = "normal"
                self.segment_started = True
                thinking_closed = True
                continue

            if self.state == "tool":
                idx = self.buffer.find(_GLIMMER_TOOL_END)
                if idx < 0:
                    # Hold partial end-marker tails so they never leak.
                    hold = self._partial_marker_len(self.buffer, (_GLIMMER_TOOL_END,))
                    if hold:
                        break
                    # Tool body without any end marker: suppress, drop.
                    self.buffer = ""
                    break
                # Position-aware: consume through the end marker; anything
                # after it (in this chunk) is the next segment start.
                self.buffer = self.buffer[idx + len(_GLIMMER_TOOL_END) :].lstrip("\n")
                self.state = "normal"
                self.segment_started = True
                continue

            # ---- NORMAL state ----
            # At a segment start, leading whitespace/newlines before a header
            # or thinking-open marker are part of the routing, not content.
            # This includes a *partial* routing marker (e.g. "<atem:fu" of
            # "<atem:function_calls>") that arrives split across chunks.
            if self.segment_started and self.buffer[:1].isspace():
                stripped = self.buffer.lstrip()
                if self._starts_routing(stripped):
                    self.buffer = stripped
                    if not self.buffer:
                        break
                    continue
                if stripped == "":
                    # Whitespace only so far — a routing marker may follow in
                    # a later chunk; hold without flipping the segment flag.
                    break
                # The head of a routing marker may be split ("to", "t",
                # "<|start|>", "<atem:") — hold those too.
                if self._is_routing_head(stripped):
                    break
                self.segment_started = False

            # "to=self<|message|>" opens thinking — check before the generic
            # header strip (which would otherwise consume it as a header).
            if self.segment_started and self.buffer.startswith(_GLIMMER_SELF_OPEN):
                if self._reasoning_emitted:
                    # Consecutive thinking block: separate from the previous.
                    self._pending_reasoning_sep = True
                self.buffer = self.buffer[len(_GLIMMER_SELF_OPEN) :].lstrip("\n")
                self.state = "thinking"
                self.segment_started = False
                continue

            # Strip a leading recipient header, but ONLY at a genuine segment
            # start (stream start or right after a close marker). A quoted
            # "to=user<|message|>" mid-answer must survive.
            if self.segment_started:
                m = _HEADER_RE.match(self.buffer)
                if m:
                    self.buffer = self.buffer[m.end() :]
                    continue

            # Re-open thinking? Only at a routing boundary, and only for the
            # complete marker (a split "to=self<|m" mid-answer stays content).
            idx_self = self.buffer.find(_GLIMMER_SELF_OPEN)
            idx_tool = self.buffer.find(_GLIMMER_TOOL_START)
            idx_eom = self.buffer.find(_GLIMMER_EOM)
            idx_eot = self.buffer.find(_GLIMMER_EOT)

            candidates = [
                (i, kind)
                for i, kind in (
                    (idx_self, "self"),
                    (idx_tool, "tool"),
                    (idx_eom, "eom"),
                    (idx_eot, "eot"),
                )
                if i >= 0
            ]
            candidates.sort()

            if not candidates:
                # Emit what cannot grow into a marker or header prefix. A
                # header in progress ("to=", "to=u", "to=us<|m", ...) is held
                # whole until it resolves or diverges.
                if self._is_header_in_progress(self.buffer):
                    break
                emit, self.buffer = self._split_partial(
                    self.buffer,
                    (
                        _GLIMMER_SELF_OPEN,
                        _GLIMMER_TOOL_START,
                        _GLIMMER_TOOL_END,
                        _GLIMMER_EOM,
                        _GLIMMER_EOT,
                        "to=",
                    ),
                )
                if emit:
                    content.append(emit)
                    if emit.strip():
                        self.segment_started = False
                break

            idx, kind = candidates[0]
            if idx:
                # Text before the structural marker: emit (already held-safe).
                before = self.buffer[:idx]
                # Leading header check already done above; emit rest verbatim.
                if before:
                    content.append(before)
                    # Whitespace-only output (e.g. an <|eom|> substitution)
                    # does not start a real content segment.
                    if before.strip():
                        self.segment_started = False
                self.buffer = self.buffer[idx:]
                continue

            if kind == "self":
                # The leading-text branch above already emitted everything
                # before this marker, so idx == 0 here. Re-open thinking only
                # at a genuine segment start (stream start or after a close
                # marker); a quoted marker mid-answer flows as content.
                if not self.segment_started:
                    # Quoted "to=self<|message|>" mid-answer: emit verbatim
                    # and treat as content.
                    emit, self.buffer = self._split_partial(
                        self.buffer,
                        (
                            _GLIMMER_SELF_OPEN,
                            _GLIMMER_TOOL_START,
                            _GLIMMER_EOM,
                        ),
                    )
                    if emit:
                        content.append(emit)
                        self.segment_started = False
                    break
                if self._reasoning_emitted:
                    # Consecutive thinking block: separate from the previous.
                    self._pending_reasoning_sep = True
                self.buffer = self.buffer[len(_GLIMMER_SELF_OPEN) :].lstrip("\n")
                self.state = "thinking"
                self.segment_started = False
                continue

            if kind == "tool":
                # A tool block at a routing segment start is a real tool
                # call; a quoted "<atem:function_calls>" mid-answer is not.
                if not self.segment_started:
                    emit, self.buffer = self._split_partial(
                        self.buffer,
                        (
                            _GLIMMER_SELF_OPEN,
                            _GLIMMER_TOOL_START,
                            _GLIMMER_EOM,
                        ),
                    )
                    if emit:
                        content.append(emit)
                    break
                self.buffer = self.buffer[len(_GLIMMER_TOOL_START) :]
                self.state = "tool"
                self.segment_started = False
                continue

            if kind == "eom":
                # Stray/continuation marker in normal state. It is a routing
                # boundary: the substituted separator space is content, and
                # the following text starts a fresh segment. Avoid doubling
                # the space when the preceding emitted text already ends in
                # whitespace.
                self.buffer = re.sub(r"^\s*<\|eom\|>\s*", " ", self.buffer, count=1)
                prev_ws = self._last_content_end.isspace()
                if self.buffer.startswith(" ") and not prev_ws:
                    content.append(" ")
                self.buffer = self.buffer.lstrip()
                self.segment_started = True
                continue

            if kind == "eot":
                # End-of-turn marker. Substitute a separator and treat the
                # following text as a fresh segment (matches the final parse).
                self.buffer = re.sub(r"^\s*<\|eot\|>\s*", " ", self.buffer, count=1)
                prev_ws = self._last_content_end.isspace()
                if self.buffer.startswith(" ") and not prev_ws:
                    content.append(" ")
                self.buffer = self.buffer.lstrip()
                self.segment_started = True
                continue

        if last and self.buffer:
            held, self.buffer = self.buffer, ""
            if self.state == "thinking":
                # Truncated reasoning (no <|eom|>) — the final parse emits
                # this as content, so the stream must too (channel agreement).
                # Strip a trailing partial marker prefix first.
                content.append(self._strip_partial_tail(held))
            elif self.state == "tool":
                pass  # truncated tool body: drop
            else:
                content.append(self._strip_partial_tail(held))

        joined_content = "".join(content)
        if joined_content:
            self._last_content_end = joined_content[-1]
        joined_reasoning = "\n".join(p for p in reasoning if p) or None
        if joined_reasoning:
            self._reasoning_emitted = True

        return GlimmerStreamDelta(
            reasoning=joined_reasoning,
            content=joined_content or None,
            thinking_closed=thinking_closed,
        )

    # -- helpers -----------------------------------------------------------
    def _append_reasoning(self, reasoning: List[str], text: str) -> None:
        """Append a reasoning fragment, honoring a pending block separator."""
        if not text:
            return
        if self._pending_reasoning_sep:
            reasoning.append("\n" + text)
            self._pending_reasoning_sep = False
        else:
            reasoning.append(text)

    @staticmethod
    def _strip_partial_tail(text: str) -> str:
        """Strip a trailing partial MARKER prefix (e.g. "hello<|e" cut
        mid-marker) so no raw fragment leaks at flush. Only "<"-starting
        markers qualify: a bare "to"/"to=" head is prose at flush (no next
        chunk) and stays."""
        for marker in (
            _GLIMMER_TOOL_START,
            _GLIMMER_TOOL_END,
            _GLIMMER_EOM,
            _GLIMMER_EOT,
            "<|start|>assistant",
        ):
            for length in range(min(len(marker), len(text)), 0, -1):
                if text.endswith(marker[:length]):
                    text = text[:-length]
                    break
        return text

    @staticmethod
    def _starts_routing(text: str) -> bool:
        """True if ``text`` starts with (or is a prefix of) a routing marker:
        a recipient header ("to=…", bare or full form), the tool block, or
        the thinking open.

        A bare "t"/"to" (no "=") is NOT treated as routing — ordinary words
        like "top" or "to the store" must not suppress an <|eom|> separator.
        """
        stripped = text.lstrip()
        if stripped.startswith("to="):
            return True
        if stripped.startswith("<|start|>assistant"):
            return True
        for marker in (_GLIMMER_SELF_OPEN, _GLIMMER_TOOL_START):
            if marker.startswith(stripped) and stripped.startswith("<"):
                return True
        return False

    @staticmethod
    def _is_routing_head(text: str) -> bool:
        """True if ``text`` is a prefix that could grow into a routing
        marker ("t"/"to"/"to=" for headers, "<|start|>" for the full form,
        "<atem:" for the tool block)."""
        if text in ("t", "to", "to="):
            return True
        if "<|start|>assistant".startswith(text) or "<|start|>".startswith(text):
            return True
        if _GLIMMER_TOOL_START.startswith(text):
            return True
        return False

    @staticmethod
    def _is_header_in_progress(text: str) -> bool:
        """True if ``text`` starts like a recipient header that is not yet
        complete ("to=", "to=u", "to=us", "to=us<|m", the full-form
        "<|start|>assistant to=…", ...), optionally after leading whitespace.
        Also True for a partial full-form prefix ("<|sta", "<|start|>ass").
        """
        stripped = text.lstrip()
        if "<|start|>assistant".startswith(stripped):
            return True  # partial full-form prefix in progress
        if stripped.startswith("<|start|>assistant"):
            if not stripped.startswith("<|start|>assistant "):
                return True  # full-form prefix still in progress
            stripped = stripped[len("<|start|>assistant ") :]
            if not stripped:
                return True  # awaiting the header head ("to=…")
            # After the prefix, a bare "t"/"to"/"to=" head opens a header.
            if stripped in ("t", "to") or stripped.startswith("to="):
                return True
            if not stripped.startswith("to="):
                return False
        if not stripped.startswith("to="):
            return False
        rest = stripped[3:]
        # Name chars, then an optional partial "<|message|>".
        name_end = 0
        while (
            name_end < len(rest)
            and rest[name_end]
            in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        ):
            name_end += 1
        after = rest[name_end:]
        if not after:
            return True  # name still in progress
        if "<|message|>".startswith(after) and after != "<|message|>":
            return True  # closing marker still in progress
        return False

    @staticmethod
    def _partial_marker_len(text: str, markers: Tuple[str, ...]) -> int:
        """Longest trailing suffix of ``text`` that is a prefix of a marker."""
        hold = 0
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for length in range(max_len, 0, -1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
                    break
        return hold

    def _split_partial(self, text: str, markers: Tuple[str, ...]) -> Tuple[str, str]:
        """Split into (emittable, held) where held may grow into a marker or
        header prefix."""
        hold = self._partial_marker_len(text, markers)
        # A completed "to=" may still be the head of a recipient header
        # ("to=user<|message|>") — hold it (and its partial heads) until the
        # next chunk decides.
        if "to=" in markers:
            if text.endswith("to="):
                hold = max(hold, 3)
            elif text.endswith("to") and (len(text) == 2 or text[-3].isspace()):
                hold = max(hold, 2)
            elif text.endswith("t") and (len(text) == 1 or text[-2].isspace()):
                hold = max(hold, 1)
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""


def make_glimmer_stream_state(processor):
    """Return a MuseGlimmerStreamState for a Glimmer processor, else None.

    Endpoints use this to decide between the Glimmer state machine (which
    handles thinking + tool suppression internally) and the legacy
    ThinkingStreamState + suppress_tool_call_content pairing.
    """
    if processor is None:
        return None
    if getattr(type(processor), "__module__", "").startswith(
        "mlx_vlm.models.muse_glimmer"
    ):
        return MuseGlimmerStreamState()
    return None
