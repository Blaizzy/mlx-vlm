"""Tests for the component-major APC storage layer (phase 5 of #1629)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCBlock, APCManager
from mlx_vlm.apc_storage import APCNode, KVBlockHandle


@pytest.fixture(autouse=True)
def _seeded():
    mx.random.seed(0)


def _fake_kv(num_layers=2, seq_len=32, heads=2, head_dim=4):
    keys = [mx.random.normal((1, heads, seq_len, head_dim)) for _ in range(num_layers)]
    values = [
        mx.random.normal((1, heads, seq_len, head_dim)) for _ in range(num_layers)
    ]
    mx.eval(keys + values)
    return keys, values


def test_kv_block_handle_roundtrip():
    keys, values = _fake_kv(num_layers=3, seq_len=16)
    handle = KVBlockHandle(keys, values)
    assert handle.resident_bytes() == sum(t.nbytes for t in keys + values)
    handle.release()
    assert handle.keys is None and handle.values is None
    assert handle.resident_bytes() == 0


def test_kv_block_handle_empty():
    handle = KVBlockHandle()
    assert handle.resident_bytes() == 0


def test_apcblock_is_node_and_delegates():
    block = APCBlock(block_id=0)
    assert isinstance(block, APCNode)
    assert block.components == {}
    assert block.keys is None and block.values is None
    assert block.resident_bytes() == 0

    keys, values = _fake_kv(num_layers=2, seq_len=16)
    block.block_hash = 123
    block.parent_hash = 7
    block.token_ids = tuple(range(16))
    block.ref_cnt = 2
    block.set_kv(keys, values)

    assert set(block.components) == {"kv"}
    assert block.keys[0] is keys[0] and block.values[1] is values[1]
    assert block.resident_bytes() == sum(t.nbytes for t in keys + values)
    assert block.node_key == 123 and block.parent_key == 7
    assert block.prefix_len == 16 and block.lock_count == 2

    block.release_components()
    assert block.components == {}
    assert block.keys is None and block.resident_bytes() == 0


def test_store_populates_kv_component_and_accounting():
    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    token_ids = list(range(2 * block_size))
    keys, values = _fake_kv(num_layers=2, seq_len=2 * block_size)

    stored = manager.store_kv_blocks(token_ids, keys, values)
    assert len(stored) == 2

    assert isinstance(stored[0].kv_handle(), KVBlockHandle)

    assert stored[0].keys[0].shape == (1, 2, block_size, 4)

    total = manager.resident_bytes()
    assert total > 0
    assert manager.stats_snapshot()["resident_bytes"] == total

    manager.release(stored)
    manager.clear()
    assert manager.resident_bytes() == 0
    assert all(b.components == {} for b in manager.pool)
