"""Live, server-side settings for mlx-vlm (T2 reloadable knobs)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

TEXT_KINDS: Tuple[str, ...] = ("text_generation",)
VISION_KINDS: Tuple[str, ...] = ("image_generation", "image_edit")

KV_SCHEMES: Tuple[str, ...] = ("uniform", "turboquant")
DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0

logger = logging.getLogger("mlx_vlm.server")

# name, type, default, reload kinds, allowed (None = free), help
KNOBS: Tuple[
    Tuple[str, str, Any, Tuple[str, ...], Optional[Tuple[str, ...]], str], ...
] = (
    (
        "kv_bits",
        "float_or_none",
        None,
        TEXT_KINDS,
        None,
        "KV bits; null = model default.",
    ),
    (
        "kv_quant_scheme",
        "str",
        "uniform",
        TEXT_KINDS,
        KV_SCHEMES,
        "KV quantization scheme.",
    ),
    (
        "kv_group_size",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "KV quantization group size.",
    ),
    ("kv_key_bits", "float_or_none", None, TEXT_KINDS, None, "Split KV key bits."),
    ("kv_value_bits", "float_or_none", None, TEXT_KINDS, None, "Split KV value bits."),
    (
        "kv_key_scheme",
        "str_or_none",
        None,
        TEXT_KINDS,
        KV_SCHEMES,
        "Split KV key scheme.",
    ),
    (
        "kv_value_scheme",
        "str_or_none",
        None,
        TEXT_KINDS,
        KV_SCHEMES,
        "Split KV value scheme.",
    ),
    (
        "quantized_kv_start",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "Quantized KV start layer.",
    ),
    ("apc_enabled", "bool", False, TEXT_KINDS, None, "Prefix caching on/off."),
    (
        "apc_disk_path",
        "str_or_none",
        None,
        TEXT_KINDS,
        None,
        "APC disk tier directory.",
    ),
    ("apc_block_size", "int", 16, TEXT_KINDS, None, "APC block size (tokens)."),
    ("apc_num_blocks", "int", 2048, TEXT_KINDS, None, "APC block pool capacity."),
    (
        "apc_disk_max_gb",
        "float_or_none",
        None,
        TEXT_KINDS,
        None,
        "APC disk tier cap (GB).",
    ),
    (
        "max_kv_size",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "Requested context budget (tokens).",
    ),
    (
        "token_queue_timeout",
        "float_or_none",
        DEFAULT_TOKEN_QUEUE_TIMEOUT,
        TEXT_KINDS,
        None,
        "Token queue wait timeout in seconds; null disables the timeout.",
    ),
    (
        "spec_draft_model",
        "str_or_none",
        None,
        TEXT_KINDS,
        None,
        "Speculative drafting model path.",
    ),
    (
        "spec_draft_kind",
        "str_or_none",
        None,
        TEXT_KINDS,
        None,
        "Speculative draft kind (auto if unset).",
    ),
    (
        "vision_cache_size",
        "int",
        20,
        VISION_KINDS,
        None,
        "Vision feature cache capacity.",
    ),
)

_KNOB_SPEC: Dict[str, Dict[str, Any]] = {
    name: {
        "type": kind,
        "default": default,
        "reload_kinds": kinds,
        "allowed": list(allowed) if allowed is not None else None,
        "help": help_text,
    }
    for (name, kind, default, kinds, allowed, help_text) in KNOBS
}

# Knobs that are naturally live (applied per request) and must never trigger a
# model reload, so they are excluded from the cache-key fingerprint.
_LIVE_KNOBS: Tuple[str, ...] = ("max_kv_size", "token_queue_timeout")


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = float(raw)
    return default if value == 0 else value


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_token_queue_timeout() -> Optional[float]:
    raw = os.environ.get("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "")
    if raw == "":
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_TOKEN_QUEUE_TIMEOUT=%r; falling back to %ss.",
            raw,
            DEFAULT_TOKEN_QUEUE_TIMEOUT,
        )
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    return timeout if timeout > 0 else None


@dataclass
class RuntimeConfig:
    kv_bits: Optional[float] = None
    kv_quant_scheme: str = "uniform"
    kv_group_size: Optional[int] = None
    kv_key_bits: Optional[float] = None
    kv_value_bits: Optional[float] = None
    kv_key_scheme: Optional[str] = None
    kv_value_scheme: Optional[str] = None
    quantized_kv_start: Optional[int] = None
    apc_enabled: bool = False
    apc_disk_path: Optional[str] = None
    apc_block_size: int = 16
    apc_num_blocks: int = 2048
    apc_disk_max_gb: Optional[float] = None
    max_kv_size: Optional[int] = None
    token_queue_timeout: Optional[float] = DEFAULT_TOKEN_QUEUE_TIMEOUT
    spec_draft_model: Optional[str] = None
    spec_draft_kind: Optional[str] = None
    vision_cache_size: int = 20

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _env_defaults: Dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        cfg = cls(
            kv_bits=_env_float("KV_BITS", None),
            kv_quant_scheme=os.environ.get("KV_QUANT_SCHEME", "uniform"),
            kv_group_size=_env_int("KV_GROUP_SIZE", None),
            kv_key_bits=_env_float("KV_KEY_BITS", None),
            kv_value_bits=_env_float("KV_VALUE_BITS", None),
            kv_key_scheme=os.environ.get("KV_KEY_SCHEME") or None,
            kv_value_scheme=os.environ.get("KV_VALUE_SCHEME") or None,
            quantized_kv_start=_env_int("QUANTIZED_KV_START", None),
            apc_enabled=os.environ.get("APC_ENABLED", "0").lower()
            in ("1", "true", "yes"),
            apc_disk_path=os.environ.get("APC_DISK_PATH") or None,
            apc_block_size=int(os.environ.get("APC_BLOCK_SIZE", "16")),
            apc_num_blocks=int(os.environ.get("APC_NUM_BLOCKS", "2048")),
            apc_disk_max_gb=_env_float("APC_DISK_MAX_GB", None),
            max_kv_size=_env_int("MAX_KV_SIZE", None),
            token_queue_timeout=_env_token_queue_timeout(),
            spec_draft_model=os.environ.get("MLX_VLM_DRAFT_MODEL") or None,
            spec_draft_kind=os.environ.get("MLX_VLM_DRAFT_KIND") or None,
            vision_cache_size=int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20")),
        )
        cfg._env_defaults = {name: getattr(cfg, name) for name in _KNOB_SPEC}
        return cfg

    def schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "type": spec["type"],
                "default": spec["default"],
                "reload_kinds": list(spec["reload_kinds"]),
                "allowed": spec["allowed"],
                "help": spec["help"],
            }
            for name, spec in _KNOB_SPEC.items()
        ]

    def current(self) -> Dict[str, Any]:
        with self._lock:
            return {name: getattr(self, name) for name in _KNOB_SPEC}

    def apply_changes(
        self, payload: Dict[str, Any], op: str = "merge"
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if op == "replace":
            with self._lock:
                for name in _KNOB_SPEC:
                    restored = self._env_defaults.get(name, _KNOB_SPEC[name]["default"])
                    setattr(self, name, restored)
        return self._apply_validated(payload)

    def _apply_validated(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            raise TypeError("settings payload must be a JSON object")
        applied: Dict[str, Any] = {}
        rejected: List[Dict[str, Any]] = []
        with self._lock:
            for name, raw in payload.items():
                if name not in _KNOB_SPEC:
                    rejected.append({"name": name, "reason": "unknown knob"})
                    continue
                spec = _KNOB_SPEC[name]
                try:
                    value = _coerce(spec["type"], raw, spec.get("allowed"))
                    if (
                        name == "token_queue_timeout"
                        and value is not None
                        and value < 0
                    ):
                        value = None
                except (TypeError, ValueError) as exc:
                    rejected.append({"name": name, "reason": str(exc)})
                    continue
                setattr(self, name, value)
                applied[name] = value
        return applied, rejected

    def reload_kinds(self, applied: Dict[str, Any]) -> Set[str]:
        kinds: Set[str] = set()
        for name in applied:
            kinds.update(_KNOB_SPEC[name]["reload_kinds"])
        return kinds

    def fingerprint(self, kinds: Optional[Iterable[str]] = None) -> str:
        kind_set = set(kinds) if kinds is not None else None
        items: List[Tuple[str, str]] = []
        with self._lock:
            for name in _KNOB_SPEC:
                if name in _LIVE_KNOBS:
                    continue
                spec = _KNOB_SPEC[name]
                if kind_set is not None and not (set(spec["reload_kinds"]) & kind_set):
                    continue
                if (
                    name.startswith("apc_")
                    and name != "apc_enabled"
                    and not self.apc_enabled
                ):
                    continue
                value = getattr(self, name)
                items.append((name, "" if value is None else str(value)))
        blob = json.dumps(sorted(items), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _coerce(kind: str, raw: Any, allowed: Optional[List[str]] = None) -> Any:
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off", ""):
                return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if kind == "int":
        return int(raw)
    if kind == "int_or_none":
        if raw is None or raw == "":
            return None
        return int(raw)
    if kind == "float_or_none":
        if raw is None or raw == "":
            return None
        value = float(raw)
        return None if value == 0 else value
    if kind == "str":
        if raw is None:
            raise ValueError("expected a string, got null")
        value = str(raw)
    elif kind == "str_or_none":
        if raw is None or raw == "":
            return None
        value = str(raw)
    else:
        raise ValueError(f"unsupported knob type {kind!r}")
    if allowed is not None and value not in allowed:
        raise ValueError(f"expected one of {allowed}, got {value!r}")
    return value


def knob_specs() -> Iterator[Tuple[str, Dict[str, Any]]]:
    yield from _KNOB_SPEC.items()
