#!/usr/bin/env python
"""Entry point for mlx_vlm CLI."""

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "generate": "generate",
        "generate_image": "generate",
        "convert": "convert",
        "chat": "chat",
        "chat_ui": "chat_ui",
        "server": "server",
        "video_generate": "video_generate",
        "configure_clients": "configure_clients",
        "configure-clients": "configure_clients",
    }

    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {set(subcommands)}")
    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"CLI requires a subcommand in {set(subcommands)}")
    if subcommand == "generate_image":
        sys.argv[1:1] = ["--output-modality", "image"]
    submodule = importlib.import_module(f"mlx_vlm.{subcommands[subcommand]}")
    submodule.main()
