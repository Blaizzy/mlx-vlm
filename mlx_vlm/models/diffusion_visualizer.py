"""Live unmasking visualization for masked-diffusion language models.

Single source of truth for the terminal rendering shared by the diffusion
models: display-width-exact word wrapping, a flash-free in-place canvas
redrawer with redraw throttling and alternate-screen escalation, and a
token-canvas visualizer used by the nemotron_labs_diffusion and llada2_moe
models. The diffusion_gemma package builds its text-stream visualizer on
the same ``_CanvasRedrawer``.

Rendering lives with the models; the shared generation engine in
``mlx_vlm.generate.diffusion`` only reuses the terminal text helpers for its
fallback draft-text redrawer.
"""

import re
import shutil
import time
import unicodedata
from typing import Any, Optional


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def escape_carriage_returns(text: str) -> str:
    return text.replace("\r", "\\r")


def clip_display_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""

    if "\n" in text:
        return "\n".join(
            clip_display_width(line, max_width) for line in text.split("\n")
        )

    out = []
    width = 0
    clipped = False
    for char in text:
        if unicodedata.combining(char):
            char_width = 0
        else:
            char_width = 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
        if width + char_width > max_width:
            clipped = True
            break
        out.append(char)
        width += char_width

    if clipped and max_width >= 3:
        while out and display_width("".join(out)) > max_width - 3:
            out.pop()
        out.append("...")

    return "".join(out)


def _take_display_width(text: str, width: int) -> str:
    taken = ""
    taken_width = 0
    for char in text:
        char_width = display_width(char)
        if taken_width + char_width > width:
            break
        taken += char
        taken_width += char_width
    return taken or text[:1]


def _wrap_text(text: str, width: int) -> str:
    """Word-wrap ``text`` so every line fits in ``width`` display columns.

    Wrapping must be display-width exact: the redrawer counts one terminal row
    per line, so a line that overflows would desynchronize the cursor-up
    redraws and corrupt the animation.
    """
    wrapped_lines = []
    for raw_line in text.split("\n"):
        line = ""
        line_width = 0
        for word in re.findall(r" *[^ ]+| +", raw_line):
            word_width = display_width(word)
            if line and line_width + word_width > width:
                wrapped_lines.append(line.rstrip(" "))
                line = ""
                line_width = 0
                word = word.lstrip(" ")
                word_width = display_width(word)
                if not word:
                    continue

            if line_width + word_width <= width:
                line += word
                line_width += word_width
                continue

            if line:
                wrapped_lines.append(line.rstrip(" "))
            while word_width > width:
                head = _take_display_width(word, width)
                wrapped_lines.append(head)
                word = word[len(head) :]
                word_width = display_width(word)
            line = word
            line_width = word_width
        wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


class _CanvasRedrawer:
    """In-place canvas redrawer with alternate-screen support.

    Overwrites the previous frame line by line in a single buffered write, so
    frames update without a clear-then-reprint flash. When the canvas grows
    taller than the terminal, switches to the alternate screen buffer (hiding
    the cursor) and shows the tail of the canvas; ``finish()`` restores the
    screen.
    """

    def __init__(self, min_interval: float = 0.05):
        self.rows = 0
        self.alternate_screen = False
        self.min_interval = min_interval
        self._last_draw = 0.0
        self._last_canvas = None

    def throttled(self) -> bool:
        return time.perf_counter() - self._last_draw < self.min_interval

    def _frame_start(self) -> str:
        if self.alternate_screen:
            return "\033[H"
        if self.rows <= 0:
            return "\r"
        return "\r" + "\033[1A" * (self.rows - 1)

    def clear(self) -> None:
        if self.rows <= 0 and not self.alternate_screen:
            return
        print(self._frame_start() + "\033[0J", end="", flush=True)
        self.rows = 0

    def draw(
        self, text: str, *, wrap_width: Optional[int] = None, force: bool = False
    ) -> None:
        now = time.perf_counter()
        if not force and now - self._last_draw < self.min_interval:
            return
        terminal_size = shutil.get_terminal_size((120, 20))
        width = max(20, terminal_size.columns - 1)
        if wrap_width is not None and wrap_width > 0:
            width = min(width, wrap_width)
        canvas = _wrap_text(text, width)
        if not force and canvas == self._last_canvas:
            return

        lines = canvas.split("\n")
        max_rows = max(1, terminal_size.lines - 2)
        controls = []
        if len(lines) >= max_rows and not self.alternate_screen:
            controls.append("\033[?1049h\033[?25l\033[H\033[2J")
            self.alternate_screen = True
            self.rows = 0
        if self.alternate_screen and len(lines) > max_rows:
            lines = lines[-max_rows:]

        # Overwrite each previous row in place (erase line, write new
        # content), then erase whatever remains below the new frame. Emitting
        # the whole frame as one write avoids flicker.
        controls.append(self._frame_start())
        frame = "\n".join(f"\033[2K{line}" for line in lines) + "\033[0J"
        print("".join(controls) + frame, end="", flush=True)
        self.rows = len(lines)
        self._last_draw = now
        self._last_canvas = canvas

    def finish(self) -> None:
        if self.alternate_screen:
            print("\033[?25h\033[?1049l", end="", flush=True)
            self.alternate_screen = False
            self.rows = 0
        else:
            self.clear()
        self._last_canvas = None


class DiffusionUnmaskingVisualizer:
    """Throttled in-place terminal view of a diffusion canvas being unmasked.

    ``visualize`` takes the ``(1, length)`` token canvas of the sequence
    generated so far. Only tokens that changed since the previous frame are
    re-decoded; everything after the first end-of-sequence token renders as
    ``[MASK]``. All printing is skipped when ``active`` is false, so callers
    only need to guard work done to build the canvas itself.
    """

    def __init__(
        self,
        *,
        active: bool,
        mask_id: int,
        eos_token_ids,
        tokenizer: Optional[Any] = None,
        skip_special_tokens: bool = False,
        min_interval: float = 0.1,
    ):
        self.active = active
        self.mask_id = mask_id
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.redrawer = _CanvasRedrawer(min_interval=min_interval)
        self.token_ids = None
        self.pieces = None

    def finish(self) -> None:
        if not self.active:
            return
        self.redrawer.finish()

    def _decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return str(token_id)
        piece = self.tokenizer.decode(
            [token_id], skip_special_tokens=self.skip_special_tokens
        )
        return escape_carriage_returns(piece) or " "

    def visualize(self, tokens: Any, force: bool = False) -> None:
        if not self.active:
            return
        if not force and self.redrawer.throttled():
            return

        token_ids = tokens[0].tolist()
        pieces = self.pieces
        previous_token_ids = self.token_ids
        if (
            pieces is None
            or previous_token_ids is None
            or len(previous_token_ids) != len(token_ids)
        ):
            pieces = ["[MASK]"] * len(token_ids)
            previous_token_ids = [self.mask_id] * len(token_ids)

        found_eos = False
        for i, token_id in enumerate(token_ids):
            previous_token_id = previous_token_ids[i]
            if found_eos:
                if previous_token_id != self.mask_id:
                    pieces[i] = "[MASK]"
                continue
            if token_id == self.mask_id:
                if previous_token_id != self.mask_id:
                    pieces[i] = "[MASK]"
            elif token_id in self.eos_token_ids:
                if previous_token_id != token_id:
                    pieces[i] = self._decode_token(token_id) or "<eos>"
                found_eos = True
            elif previous_token_id != token_id:
                pieces[i] = self._decode_token(token_id)

        self.pieces = pieces
        self.token_ids = token_ids
        self.redrawer.draw("".join(pieces), force=force)
