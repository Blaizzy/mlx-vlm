"""CLI entry point for image editing."""

from .generate.edit_image import parse_image_edit_arguments, run_image_edit_cli


def main() -> None:
    run_image_edit_cli(parse_image_edit_arguments())


if __name__ == "__main__":
    main()
