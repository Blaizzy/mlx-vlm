#!/usr/bin/env python
"""Entry point for mlx_vlm CLI."""

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "generate",
        "convert",
        "chat",
        "chat_ui",
        "server",
        "video_generate",
        "smolvlm_video_generate",
    }

    if len(sys.argv) < 2:
        print(
            f"Usage: mlx_vlm <subcommand> [options] or python -m mlx_vlm <subcommand> [options]"
        )
        print(f"Available subcommands: {', '.join(sorted(subcommands))}")
        sys.exit(1)

    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        print(f"Error: Unknown subcommand '{subcommand}'")
        print(f"Available subcommands: {', '.join(sorted(subcommands))}")
        sys.exit(1)

    # Dynamically import and run the submodule
    submodule = importlib.import_module(f"mlx_vlm.{subcommand}")
    submodule.main()
