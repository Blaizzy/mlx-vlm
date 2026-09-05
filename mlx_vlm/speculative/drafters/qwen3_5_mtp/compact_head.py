import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any, Mapping, Optional

import mlx.core as mx

COMPACT_HEAD_MODEL_TYPE = "qwen3_5_compact_proposal_head"
COMPACT_HEAD_SCHEMA_VERSION = 1
_SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
logger = logging.getLogger(__name__)


def vocab_ids_sha256(vocab_ids) -> str:
    """Hash real token IDs using an explicit little-endian int32 encoding."""
    digest = hashlib.sha256()
    for value in vocab_ids:
        digest.update(struct.pack("<I", int(value)))
    return digest.hexdigest()


def tokenizer_vocab_sha256(tokenizer) -> str:
    """Return a stable digest of a tokenizer's token-to-ID mapping."""
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("target tokenizer does not provide get_vocab()")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping):
        raise ValueError("target tokenizer vocabulary must be a mapping")
    items = sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _config_value(config: Any, name: str):
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


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
        biases: mx.array,
        vocab_ids: mx.array,
        group_size: int,
        bits: int,
        mode: str = "affine",
        full_vocab_size: Optional[int] = None,
        target_model_type: Optional[str] = None,
        target_tokenizer_sha256: Optional[str] = None,
    ):
        if mode != "affine":
            raise ValueError("compact proposal head must use affine quantization")
        if bits not in _SUPPORTED_BITS:
            raise ValueError(f"unsupported compact proposal bit width: {bits}")
        if group_size <= 0:
            raise ValueError("compact proposal group size must be positive")
        if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
            raise ValueError("compact proposal tensors must be rank 2")
        if weight.dtype != mx.uint32:
            raise ValueError("compact proposal weight must use packed uint32 values")
        if not mx.issubdtype(scales.dtype, mx.floating):
            raise ValueError("compact proposal scales must be floating point")
        if not mx.issubdtype(biases.dtype, mx.floating):
            raise ValueError("compact proposal biases must be floating point")
        if scales.dtype != biases.dtype or scales.shape != biases.shape:
            raise ValueError("compact proposal scales/biases shape or dtype mismatch")

        rows = int(weight.shape[0])
        if rows <= 0:
            raise ValueError("compact proposal head must contain at least one row")
        if int(scales.shape[0]) != rows:
            raise ValueError("compact proposal weight/scales row mismatch")
        if vocab_ids.ndim != 1 or int(vocab_ids.shape[0]) != rows:
            raise ValueError("compact proposal vocab mapping row mismatch")
        if not mx.issubdtype(vocab_ids.dtype, mx.integer):
            raise ValueError("compact proposal vocab IDs must be integers")

        packed_bits = int(weight.shape[1]) * 32
        if packed_bits % bits:
            raise ValueError("compact proposal packed weight width is invalid")
        input_dims = packed_bits // bits
        if input_dims % group_size:
            raise ValueError(
                "compact proposal hidden size must be divisible by group size"
            )
        expected_groups = input_dims // group_size
        if int(scales.shape[1]) != expected_groups:
            raise ValueError(
                "compact proposal scales/biases width differs from quantization"
            )

        ids = [int(value) for value in vocab_ids.tolist()]
        if any(value < 0 for value in ids):
            raise ValueError("compact proposal vocab IDs must be non-negative")
        if any(left >= right for left, right in zip(ids, ids[1:])):
            raise ValueError(
                "compact proposal vocab IDs must be unique and strictly increasing"
            )
        if full_vocab_size is not None:
            full_vocab_size = int(full_vocab_size)
            if full_vocab_size <= 0:
                raise ValueError("target vocabulary size must be positive")
            if ids[-1] >= full_vocab_size:
                raise ValueError("compact proposal vocab ID exceeds target vocabulary")

        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.vocab_ids = vocab_ids.astype(mx.int32)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)
        self.full_vocab_size = full_vocab_size
        self.target_model_type = target_model_type
        self.target_tokenizer_sha256 = target_tokenizer_sha256
        self._input_dims = input_dims
        self._maximum_vocab_id = ids[-1]
        self.last_proposal_implementation = "not_run"

    @property
    def output_dims(self) -> int:
        return int(self.weight.shape[0])

    @property
    def input_dims(self) -> int:
        return self._input_dims

    def validate_target(self, target_config, tokenizer=None) -> None:
        """Reject a sidecar that cannot match the selected target model."""
        hidden_size = _config_value(target_config, "hidden_size")
        vocab_size = _config_value(target_config, "vocab_size")
        model_type = _config_value(target_config, "model_type")
        if hidden_size is None or vocab_size is None:
            raise ValueError("target config must provide hidden_size and vocab_size")
        if int(hidden_size) != self.input_dims:
            raise ValueError(
                f"compact proposal hidden size {self.input_dims} does not match "
                f"target hidden size {hidden_size}"
            )
        if self.full_vocab_size is not None and int(vocab_size) != self.full_vocab_size:
            raise ValueError(
                f"compact proposal vocabulary size {self.full_vocab_size} does not "
                f"match target vocabulary size {vocab_size}"
            )
        if self._maximum_vocab_id >= int(vocab_size):
            raise ValueError("compact proposal vocab ID exceeds target vocabulary")
        if (
            self.target_model_type is not None
            and model_type is not None
            and str(model_type) != self.target_model_type
        ):
            raise ValueError(
                f"compact proposal target type {self.target_model_type} does not "
                f"match selected target type {model_type}"
            )
        if tokenizer is not None and self.target_tokenizer_sha256 is not None:
            observed = tokenizer_vocab_sha256(tokenizer)
            if observed != self.target_tokenizer_sha256:
                raise ValueError(
                    "compact proposal tokenizer vocabulary does not match target"
                )

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
        self.last_proposal_implementation = "quantized_matmul_argmax"
        return self.vocab_ids[compact_ids]


def load_compact_proposal_head(
    path: str | Path,
    *,
    target_config=None,
    tokenizer=None,
) -> CompactProposalHead:
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    weights_path = root / "compact_head.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "compact proposal sidecar requires config.json and "
            f"compact_head.safetensors: {root}"
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid compact proposal config: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError("compact proposal config must be a JSON object")
    if config.get("schema_version") != COMPACT_HEAD_SCHEMA_VERSION:
        raise ValueError("unsupported compact proposal schema version")
    if config.get("model_type") != COMPACT_HEAD_MODEL_TYPE:
        raise ValueError("unsupported compact proposal model type")

    tensors = mx.load(str(weights_path))
    required = {"weight", "scales", "biases", "vocab_ids"}
    if set(tensors) != required:
        missing = sorted(required.difference(tensors))
        extra = sorted(set(tensors).difference(required))
        raise ValueError(
            f"compact proposal tensor keys differ: missing={missing}, extra={extra}"
        )

    try:
        target = config.get("target") or {}
        selection = config.get("selection") or {}
        group_size = int(config["group_size"])
        bits = int(config["bits"])
        expected_rows = int(config["rows"])
        expected_hidden = int(config["hidden_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid compact proposal config values") from error
    if not isinstance(target, dict) or not isinstance(selection, dict):
        raise ValueError("compact proposal target and selection must be objects")

    head = CompactProposalHead(
        weight=tensors["weight"],
        scales=tensors["scales"],
        biases=tensors["biases"],
        vocab_ids=tensors["vocab_ids"],
        group_size=group_size,
        bits=bits,
        mode=config.get("mode", "affine"),
        full_vocab_size=target.get("vocab_size"),
        target_model_type=target.get("model_type"),
        target_tokenizer_sha256=target.get("tokenizer_vocab_sha256"),
    )

    if head.output_dims != expected_rows or head.input_dims != expected_hidden:
        raise ValueError("compact proposal sidecar shape differs from config")
    expected_mapping_hash = selection.get("mapping_sha256")
    if expected_mapping_hash is not None:
        observed_mapping_hash = vocab_ids_sha256(head.vocab_ids.tolist())
        if observed_mapping_hash != expected_mapping_hash:
            raise ValueError("compact proposal vocabulary mapping hash differs")
    if target_config is not None:
        head.validate_target(target_config, tokenizer=tokenizer)
        if not target:
            logger.warning(
                "Compact proposal sidecar has no target identity metadata; "
                "only shape and token-range compatibility were checked."
            )
    return head
