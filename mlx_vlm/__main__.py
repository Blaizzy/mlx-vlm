#!/usr/bin/env python
"""Entry point for mlx_vlm CLI."""

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "generate",
        "generate_image",
        "generate_video",
        "convert",
        "chat",
        "chat_ui",
        "server",
        "moe_offload",
    }

    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    if subcommand in {"generate_image", "generate_video"}:
        output_modality = subcommand.removeprefix("generate_")
        sys.argv[1:1] = ["--output-modality", output_modality]
        subcommand = "generate"
    submodule = importlib.import_module(f"mlx_vlm.{subcommand}")
    submodule.main()
