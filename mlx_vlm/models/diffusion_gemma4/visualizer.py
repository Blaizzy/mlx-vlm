"""Live unmasking visualization for DiffusionGemma4 generation.

Renders the full sequence generated so far — finalized text from completed
canvases followed by the in-flight canvas with ``[Mask]`` placeholders —
wrapped to the terminal width and redrawn in place on every denoising step.
When the canvas grows taller than the terminal, the view switches to the
alternate screen buffer and restores it when generation finishes.

Like the nemotron and llada masked-diffusion visualizers, all rendering lives
with the model; the shared generation engine only forwards draft frames here.
"""

import shutil
import sys
import time
import unicodedata
from typing import Any, Dict, Optional


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def _take_display_width(text: str, width: int) -> str:
    taken = ""
    taken_width = 0
    for char in text:
        char_width = _display_width(char)
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
        for word in raw_line.split(" "):
            word_width = _display_width(word)
            separator = 1 if line else 0
            if line_width + separator + word_width <= width:
                line += (" " if separator else "") + word
                line_width += separator + word_width
                continue
            if line:
                wrapped_lines.append(line)
            while word_width > width:
                head = _take_display_width(word, width)
                wrapped_lines.append(head)
                word = word[len(head) :]
                word_width = _display_width(word)
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

    def draw(self, text: str, *, wrap_width: Optional[int] = None) -> None:
        now = time.perf_counter()
        if now - self._last_draw < self.min_interval:
            return
        terminal_size = shutil.get_terminal_size((120, 20))
        width = max(20, terminal_size.columns - 1)
        if wrap_width is not None and wrap_width > 0:
            width = min(width, wrap_width)
        canvas = _wrap_text(text, width)
        if canvas == self._last_canvas:
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


class DiffusionGemma4Visualizer:
    """Composes the full-sequence canvas and drives the redrawer."""

    def __init__(self, wrap_width: int = 0):
        self.wrap_width = wrap_width
        self.redrawer = _CanvasRedrawer()
        self.live_text = ""

    def _draw_canvas(self, draft_text: str = "") -> None:
        canvas = self.live_text + draft_text
        if not canvas:
            return
        self.redrawer.draw(
            canvas.replace("\r", "\\r"),
            wrap_width=self.wrap_width if self.wrap_width else None,
        )

    def handle_draft(self, response: Any) -> None:
        self._draw_canvas(response.draft_text)

    def handle_text(self, text: str) -> bool:
        self.live_text += text
        if text:
            self._draw_canvas()
        return True

    def finish(self, text: str) -> None:
        self.redrawer.finish()
        if text:
            print(text, end="", flush=True)


def make_unmasking_visualizer(
    kwargs: Dict[str, Any], verbose: bool
) -> Optional[DiffusionGemma4Visualizer]:
    """Build the live unmasking visualizer for a generation call.

    Like the nemotron visualizer's ``visualize and sys.stdout.isatty()`` gate,
    the live view is on by default for verbose terminal runs. The flag is
    written back into the stream kwargs so the engine yields draft frames;
    explicit ``diffusion_show_unmasking=False`` disables the view.
    """
    if verbose and sys.stdout.isatty():
        kwargs.setdefault("diffusion_show_unmasking", True)
    if not kwargs.get("diffusion_show_unmasking", False):
        return None
    if not sys.stdout.isatty():
        return None
    return DiffusionGemma4Visualizer(
        wrap_width=int(kwargs.get("diffusion_unmasking_width", 0) or 0)
    )


def install_output_handler_patch() -> None:
    """Patch the engine's ``DiffusionOutputHandler`` to use model visualizers.

    Mirrors ``install_auto_processor_patch``: the shared generation engine is
    left untouched on disk, and the model package installs its display hook at
    import time. Models that expose ``make_unmasking_visualizer`` get their
    own visualizer; everything else keeps the stock handler behavior.
    """
    from ...generate import diffusion as engine

    handler = engine.DiffusionOutputHandler
    if getattr(handler, "_model_visualizer_patched", False):
        return
    handler._model_visualizer_patched = True

    original_init = handler.__init__
    original_handle_draft = handler.handle_draft
    original_handle_text = handler.handle_text
    original_finish = handler.finish

    def patched_init(self, model, kwargs, verbose):
        make_visualizer = getattr(model, "make_unmasking_visualizer", None)
        # Build before the stock init so the visualizer can default
        # ``diffusion_show_unmasking`` into the stream kwargs.
        self._model_visualizer = (
            make_visualizer(kwargs, verbose) if make_visualizer is not None else None
        )
        original_init(self, model, kwargs, verbose)
        if self._model_visualizer is not None:
            self.redrawer = None  # the model visualizer owns all rendering

    def patched_handle_draft(self, response):
        if self._model_visualizer is not None:
            if self.verbose:
                self._model_visualizer.handle_draft(response)
            return
        original_handle_draft(self, response)

    def patched_handle_text(self, text):
        if self._model_visualizer is not None:
            if self.verbose:
                return self._model_visualizer.handle_text(text)
            return False
        return original_handle_text(self, text)

    def patched_finish(self, text):
        if self._model_visualizer is not None:
            if self.verbose:
                self._model_visualizer.finish(text)
            return
        original_finish(self, text)

    handler.__init__ = patched_init
    handler.handle_draft = patched_handle_draft
    handler.handle_text = patched_handle_text
    handler.finish = patched_finish


__all__ = [
    "DiffusionGemma4Visualizer",
    "install_output_handler_patch",
    "make_unmasking_visualizer",
]

install_output_handler_patch()
