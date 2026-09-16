"""Tests for the component-major APC storage layer (phase 5 of #1629)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.apc_storage import KVBlockHandle


@pytest.fixture(autouse=True)
def _seeded():
    mx.random.seed(0)


def test_kv_block_handle_empty():
    handle = KVBlockHandle()
    assert handle.resident_bytes() == 0
