"""Read tensor-only torch ZIP checkpoints without importing torch or NumPy.

The restricted unpickler creates metadata, never source classes. Only dense,
contiguous tensor storages and OrderedDict are accepted. The intermediate
safetensors file preserves source bytes; MLX handles BF16 conversion later.
"""

import collections
import io
import json
import math
import pickle
import struct
import zipfile
from dataclasses import dataclass

DTYPES = {
    "FloatStorage": ("F32", 4),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8),
    "IntStorage": ("I32", 4),
    "BoolStorage": ("BOOL", 1),
    "ByteStorage": ("U8", 1),
}


@dataclass
class Storage:
    kind: str
    key: str
    size: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def _tensor(storage, offset, shape, stride, *unused):
    return Tensor(storage, offset, shape, stride)


class TensorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in DTYPES:
            return name
        if module == "torch._utils" and name in (
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
        ):
            return _tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda tensor, *args: tensor
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, value):
        if (
            not isinstance(value, tuple)
            or len(value) != 5
            or value[0] != "storage"
            or value[1] not in DTYPES
        ):
            raise pickle.UnpicklingError("Unsupported checkpoint storage")
        return Storage(value[1], str(value[2]), value[4])


def read_metadata(archive):
    names = archive.namelist()
    candidates = [n for n in names if n.endswith("/data.pkl")]
    if len(candidates) != 1:
        raise ValueError("Expected a tensor ZIP checkpoint with one data.pkl")
    prefix = candidates[0].removesuffix("data.pkl")
    if (
        prefix + "byteorder" in names
        and archive.read(prefix + "byteorder") != b"little"
    ):
        raise ValueError("Only little-endian checkpoints are supported")
    data = TensorUnpickler(io.BytesIO(archive.read(candidates[0]))).load()
    if "state_dict" in data:
        data = data["state_dict"]
    elif "model" in data and "model_config" in data:
        data = data["model"]
    # The released structure generator contains non-tensor training metadata.
    data.pop("__mixhavior__", None)
    if not isinstance(data, dict) or not all(
        isinstance(v, Tensor) for v in data.values()
    ):
        raise ValueError("Checkpoint must contain only named dense tensors")
    return prefix, data


def to_safetensors(source, destination):
    """Stream a tensor-only checkpoint to a byte-exact safetensors file."""
    with zipfile.ZipFile(source) as archive:
        prefix, tensors = read_metadata(archive)
        header, offset = {}, 0
        for name, t in tensors.items():
            stride = 1
            for size, actual in zip(reversed(t.shape), reversed(t.stride)):
                if size > 1 and actual != stride:
                    raise ValueError(f"Non-contiguous tensor: {name}")
                stride *= size
            dtype, width = DTYPES[t.storage.kind]
            length = math.prod(t.shape) * width
            if t.offset < 0 or t.offset + math.prod(t.shape) > t.storage.size:
                raise ValueError(f"Tensor outside storage: {name}")
            header[name] = {
                "dtype": dtype,
                "shape": list(t.shape),
                "data_offsets": [offset, offset + length],
            }
            offset += length
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with open(destination, "wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for name, t in tensors.items():
                width = DTYPES[t.storage.kind][1]
                remaining = math.prod(t.shape) * width
                with archive.open(prefix + "data/" + t.storage.key) as raw:
                    raw.seek(t.offset * width)
                    while remaining:
                        chunk = raw.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise ValueError(f"Truncated storage: {name}")
                        out.write(chunk)
                        remaining -= len(chunk)
    return len(tensors)
