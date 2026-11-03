"""Modality-independent MTP head checkpoint layout.

Trainable-weight saves keep ``mtp_head.*`` out of the target weights and write
the head as a standalone drafter beside them, so the target loads strictly.
"""

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

MTP_PREFIX = "mtp_head."


def mtp_head_dir(weights_file):
    """Return the drafter directory saved beside a trainable-weights file."""
    path = Path(weights_file)
    step, sep, _ = path.name.partition("_")
    return path.with_name(f"{step}_mtp" if sep and step.isdigit() else "mtp")


def split_mtp_weights(weights):
    """Split flat weights into ``(main, head)`` with the head prefix removed."""
    main = {k: v for k, v in weights.items() if not k.startswith(MTP_PREFIX)}
    head = {
        k.removeprefix(MTP_PREFIX): v
        for k, v in weights.items()
        if k.startswith(MTP_PREFIX)
    }
    return main, head


def save_mtp_head(head, weights, directory):
    """Write a trained MTP head as a standalone drafter loadable by ``load_drafter``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(directory / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    config = asdict(head.config)
    config["text_config"]["mtp_num_hidden_layers"] = len(head.layers)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return directory
