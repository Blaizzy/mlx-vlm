import json
from pathlib import Path
from typing import Optional

import mlx.core as mx


class CompactProposalHead:
    """Proposal-only quantized LM-head rows with real-vocabulary remapping.

    This intentionally is not an ``nn.Module``. The drafter owns it as opaque
    runtime state, so loading MTP checkpoint weights cannot traverse or mutate
    the separately-derived proposal tensors.
    """

    def __init__(
        self,
        *,
        weight: mx.array,
        scales: mx.array,
        biases: Optional[mx.array],
        vocab_ids: mx.array,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ):
        if weight.ndim != 2 or scales.ndim != 2:
            raise ValueError("compact proposal tensors must be rank 2")
        rows = int(weight.shape[0])
        if int(scales.shape[0]) != rows:
            raise ValueError("compact proposal weight/scales row mismatch")
        if biases is not None and int(biases.shape[0]) != rows:
            raise ValueError("compact proposal weight/biases row mismatch")
        if vocab_ids.ndim != 1 or int(vocab_ids.shape[0]) != rows:
            raise ValueError("compact proposal vocab mapping row mismatch")
        if bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"unsupported compact proposal bit width: {bits}")
        if group_size <= 0:
            raise ValueError("compact proposal group size must be positive")

        ids = [int(value) for value in vocab_ids.tolist()]
        if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
            raise ValueError(
                "compact proposal vocab IDs must be unique and non-negative"
            )

        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.vocab_ids = vocab_ids.astype(mx.int32)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)

    @property
    def output_dims(self) -> int:
        return int(self.weight.shape[0])

    @property
    def input_dims(self) -> int:
        return int(self.weight.shape[1]) * 32 // self.bits

    def logits(self, hidden: mx.array) -> mx.array:
        if int(hidden.shape[-1]) != self.input_dims:
            raise ValueError(
                f"compact proposal hidden size {hidden.shape[-1]} != {self.input_dims}"
            )
        return mx.quantized_matmul(
            hidden,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def propose(self, hidden: mx.array) -> mx.array:
        compact_ids = mx.argmax(self.logits(hidden), axis=-1)
        return self.vocab_ids[compact_ids]


def load_compact_proposal_head(path: str | Path) -> CompactProposalHead:
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    weights_path = root / "compact_head.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"compact proposal sidecar requires config.json and "
            f"compact_head.safetensors: {root}"
        )

    config = json.loads(config_path.read_text())
    tensors = mx.load(str(weights_path))
    required = {"weight", "scales", "vocab_ids"}
    missing = required.difference(tensors)
    if missing:
        raise ValueError(f"compact proposal sidecar is missing: {sorted(missing)}")

    head = CompactProposalHead(
        weight=tensors["weight"],
        scales=tensors["scales"],
        biases=tensors.get("biases"),
        vocab_ids=tensors["vocab_ids"],
        group_size=int(config["group_size"]),
        bits=int(config["bits"]),
        mode=config.get("mode", "affine"),
    )
    expected_rows = int(config.get("rows", head.output_dims))
    expected_hidden = int(config.get("hidden_size", head.input_dims))
    if head.output_dims != expected_rows or head.input_dims != expected_hidden:
        raise ValueError("compact proposal sidecar shape differs from config")
    return head
