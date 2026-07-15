from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten


def materialize_mx_arrays(tree: Any) -> Any:
    arrays = [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]
    if arrays:
        mx.eval(*arrays)
    return tree
