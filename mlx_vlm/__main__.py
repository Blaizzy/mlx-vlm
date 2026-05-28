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
        "video_generate",
    }

    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    if subcommand == "generate_image":
        sys.argv[1:1] = ["--output-modality", "image"]
        subcommand = "generate"
    submodule = importlib.import_module(f"mlx_vlm.{subcommand}")
    submodule.main()
