"""Live unmasking visualization for masked-diffusion language models.

Shared by the nemotron_labs_diffusion and llada2_moe models: decoded token
pieces are tracked across redraws (masked positions render as ``[MASK]``),
word-wrapped to the terminal width, and redrawn in place with cursor-up
controls. When the canvas grows taller than the terminal, the view switches
to the alternate screen buffer and restores it when generation finishes.

Rendering lives with the models; the shared generation engine in
``mlx_vlm.generate.diffusion`` keeps its own draft-text redrawer untouched.
"""

import shutil
import time
from typing import Any, Optional


def _wrap_text(text: str, width: int) -> str:
    lines = []
    while len(text) > width:
        split_at = text.rfind(" ", 0, width + 1)
        if split_at <= 0:
            split_at = width
        lines.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    if text:
        lines.append(text)
    return "\n".join(lines)


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
        self.min_interval = min_interval
        self.alternate_screen = False
        self.rows = 0
        self.last_draw = 0.0
        self.token_ids = None
        self.pieces = None
        self.canvas = ""

    def _clear(self) -> None:
        if not self.active:
            return
        if self.alternate_screen:
            print("\033[H\033[2J", end="", flush=True)
            self.rows = 0
            return
        if self.rows == 0:
            return
        controls = ["\r\033[2K"]
        for _ in range(self.rows - 1):
            controls.append("\033[1A\r\033[2K")
        print("".join(controls), end="", flush=True)
        self.rows = 0

    def finish(self) -> None:
        if not self.active:
            return
        if self.alternate_screen:
            print("\033[H\033[2J\033[?25h\033[?1049l", end="", flush=True)
            self.alternate_screen = False
            self.rows = 0
        else:
            self._clear()

    def _decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return str(token_id)
        piece = self.tokenizer.decode(
            [token_id], skip_special_tokens=self.skip_special_tokens
        )
        return piece.replace("\n", "\\n") or " "

    def visualize(self, tokens: Any, force: bool = False) -> None:
        if not self.active:
            return
        now = time.perf_counter()
        if not force and now - self.last_draw < self.min_interval:
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

        terminal_size = shutil.get_terminal_size((120, 20))
        terminal_width = max(20, terminal_size.columns - 1)
        canvas = _wrap_text("".join(pieces), terminal_width)
        if not force and canvas == self.canvas:
            return
        rows = max(1, canvas.count("\n") + 1)
        if rows >= max(1, terminal_size.lines - 2) and not self.alternate_screen:
            print("\033[?1049h\033[?25l\033[H\033[2J", end="", flush=True)
            self.alternate_screen = True
        self._clear()
        print(canvas, end="", flush=True)
        self.rows = rows
        self.last_draw = now
        self.canvas = canvas
