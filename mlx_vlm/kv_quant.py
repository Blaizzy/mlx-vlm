from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

TURBOQUANT_SCHEME = "turboquant"
UNIFORM_SCHEME = "uniform"


@dataclass(frozen=True)
class KVQuantSpec:
    scheme: str
    bits: float
    group_size: int

    @property
    def is_turboquant(self) -> bool:
        return self.scheme == TURBOQUANT_SCHEME


@dataclass(frozen=True)
class KVQuantPolicy:
    bits: float
    key: KVQuantSpec
    value: KVQuantSpec

    @property
    def is_homogeneous(self) -> bool:
        return self.key.scheme == self.value.scheme

    @property
    def is_turboquant(self) -> bool:
        return self.key.is_turboquant and self.value.is_turboquant

    @property
    def scheme(self) -> str:
        if not self.is_homogeneous:
            raise ValueError(
                "policy mixes quantization schemes; inspect key/value separately"
            )
        return self.key.scheme

    @property
    def group_size(self) -> int:
        return self.key.group_size

    @property
    def has_split_override(self) -> bool:
        return not _is_default_split(self.bits, self.key.bits, self.value.bits)

    def to_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "scheme": self.key.scheme,
        }
        if self.has_split_override:
            config["key_bits"] = self.key.bits
            config["value_bits"] = self.value.bits
        if not self.is_homogeneous:
            config["key_scheme"] = self.key.scheme
            config["value_scheme"] = self.value.scheme
        return config

    def fingerprint(self, quantized_kv_start: Any) -> str:
        key_bits = self.key.bits if self.has_split_override else None
        value_bits = self.value.bits if self.has_split_override else None
        descriptor = kv_quant_fingerprint(
            self.bits,
            self.group_size,
            self.key.scheme,
            quantized_kv_start,
            key_bits,
            value_bits,
        )
        if not self.is_homogeneous:
            descriptor += f"-ks{self.key.scheme}-vs{self.value.scheme}"
        return descriptor


def _is_default_split(bits: float, key_bits: float, value_bits: float) -> bool:
    fractional = not math.isclose(bits, round(bits), abs_tol=1e-6)
    expected_key = math.floor(bits) if fractional else bits
    expected_value = math.ceil(bits) if fractional else bits
    return math.isclose(key_bits, expected_key, abs_tol=1e-6) and math.isclose(
        value_bits, expected_value, abs_tol=1e-6
    )


def kv_quant_fingerprint(
    kv_bits: Any,
    kv_group_size: Any,
    kv_quant_scheme: Any,
    quantized_kv_start: Any,
    kv_key_bits: Any = None,
    kv_value_bits: Any = None,
) -> str:
    descriptor = f"kv{kv_bits}-{kv_group_size}-{kv_quant_scheme}-{quantized_kv_start}"
    if kv_key_bits is not None or kv_value_bits is not None:
        descriptor += f"-k{kv_key_bits}-v{kv_value_bits}"
    return descriptor


def from_legacy(
    kv_bits: Optional[float],
    kv_quant_scheme: Optional[str] = None,
    kv_group_size: int = 64,
    kv_key_bits: Optional[float] = None,
    kv_value_bits: Optional[float] = None,
    kv_key_scheme: Optional[str] = None,
    kv_value_scheme: Optional[str] = None,
) -> Optional[KVQuantPolicy]:
    if kv_bits is None:
        return None

    from .turboquant import resolve_kv_bits, turboquant_enabled

    if turboquant_enabled(kv_bits, kv_quant_scheme):
        bits, default_key_bits, default_value_bits = resolve_kv_bits(
            kv_bits, kv_key_bits, kv_value_bits
        )
        scheme = TURBOQUANT_SCHEME
    else:
        bits = float(kv_bits)
        default_key_bits = float(kv_key_bits) if kv_key_bits is not None else bits
        default_value_bits = float(kv_value_bits) if kv_value_bits is not None else bits
        scheme = UNIFORM_SCHEME

    group_size = int(kv_group_size)
    key_scheme = _validate_scheme(kv_key_scheme, scheme)
    value_scheme = _validate_scheme(kv_value_scheme, scheme)

    key_bits = _bits_for_scheme(key_scheme, scheme, default_key_bits, kv_key_bits, bits)
    value_bits = _bits_for_scheme(
        value_scheme, scheme, default_value_bits, kv_value_bits, bits
    )

    return KVQuantPolicy(
        bits=bits,
        key=KVQuantSpec(scheme=key_scheme, bits=key_bits, group_size=group_size),
        value=KVQuantSpec(scheme=value_scheme, bits=value_bits, group_size=group_size),
    )


def _validate_scheme(scheme: Optional[str], fallback: str) -> str:
    if scheme is None:
        return fallback
    if scheme not in (UNIFORM_SCHEME, TURBOQUANT_SCHEME):
        raise ValueError(
            f"unknown KV quantization scheme {scheme!r}; "
            f"expected {UNIFORM_SCHEME!r} or {TURBOQUANT_SCHEME!r}"
        )
    return scheme


def _bits_for_scheme(
    scheme: str,
    base_scheme: str,
    default_bits: float,
    explicit_bits: Optional[float],
    nominal_bits: float,
) -> float:
    if explicit_bits is not None:
        bits = float(explicit_bits)
    elif scheme == base_scheme:
        bits = default_bits
    else:
        bits = float(nominal_bits)
    if scheme == UNIFORM_SCHEME and not float(bits).is_integer():
        raise ValueError(f"uniform KV quantization requires integer bits, got {bits}")
    return bits


def from_config(config: Optional[Dict[str, Any]]) -> Optional[KVQuantPolicy]:
    if not config:
        return None
    return from_legacy(
        config.get("bits"),
        config.get("scheme"),
        int(config.get("group_size", 64)),
        config.get("key_bits"),
        config.get("value_bits"),
        config.get("key_scheme"),
        config.get("value_scheme"),
    )
