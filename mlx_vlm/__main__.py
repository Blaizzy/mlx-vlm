import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "convert",
        "generate",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    submodule = importlib.import_module(f"mlx_vlm.{subcommand}")
    submodule.main()