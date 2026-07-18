#!/usr/bin/env python
"""Entry point for mlx_vlm CLI."""

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "generate",
        "generate_image",
        "convert",
        "chat",
        "chat_ui",
        "server",
    }

    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")

    # Device selection must happen before the submodule import: generation
    # streams are created at import time from the default device.
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]
        if device not in ("cpu", "gpu"):
            raise ValueError(f"--device must be cpu or gpu, got {device}")
        import mlx.core as mx

        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    if subcommand == "generate_image":
        sys.argv[1:1] = ["--output-modality", "image"]
        subcommand = "generate"
    submodule = importlib.import_module(f"mlx_vlm.{subcommand}")
    submodule.main()
