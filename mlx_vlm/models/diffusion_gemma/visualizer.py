"""Live unmasking visualization for DiffusionGemma4 generation.

Renders the full sequence generated so far — finalized text from completed
canvases followed by the in-flight canvas with ``[Mask]`` placeholders —
wrapped to the terminal width and redrawn in place on every denoising step.
When the canvas grows taller than the terminal, the view switches to the
alternate screen buffer and restores it when generation finishes.

Like the nemotron and llada masked-diffusion visualizers, all rendering lives
with the model; the shared generation engine only forwards draft frames here.
The rendering primitives — display-width-exact wrapping and the flash-free
in-place redrawer — are shared with the other diffusion models and live in
``mlx_vlm.models.diffusion_visualizer``.
"""

import sys
from typing import Any, Dict, Optional

from ..diffusion_visualizer import (
    _CanvasRedrawer,
    _display_width,
    _escape_carriage_returns,
    _take_display_width,
    _wrap_text,
)


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
            _escape_carriage_returns(canvas),
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
    "_CanvasRedrawer",
    "_display_width",
    "_escape_carriage_returns",
    "_take_display_width",
    "_wrap_text",
    "DiffusionGemma4Visualizer",
    "install_output_handler_patch",
    "make_unmasking_visualizer",
]

install_output_handler_patch()
