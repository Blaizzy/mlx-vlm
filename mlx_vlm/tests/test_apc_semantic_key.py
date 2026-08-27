"""Tests for the complete semantic APC cache key (phase 6 of #1629)."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.apc import (
    _hash_payload,
    apc_disk_namespace,
    hash_image_payload,
    model_key_dependencies,
    semantic_extra_hash,
    tenant_scoped_hash,
)
from mlx_vlm.apc_adapters import ADAPTER_SCHEMA_VERSION


def test_reduces_to_tenant_image_when_no_extra_inputs():
    assert semantic_extra_hash(tenant="t", image_hash=123) == tenant_scoped_hash(
        "t", 123
    )

    got = semantic_extra_hash(
        tenant="tenant-a",
        image_hash=123,
        media={"audio": None, "video": None, "masks": None},
    )
    assert got == tenant_scoped_hash("tenant-a", 123)


def test_media_inputs_change_the_key():
    base = semantic_extra_hash(tenant="t", image_hash=7)
    audio = mx.ones((1, 4, 8))
    with_audio = semantic_extra_hash(tenant="t", image_hash=7, media={"audio": audio})
    assert with_audio != base

    assert with_audio == semantic_extra_hash(
        tenant="t", image_hash=7, media={"audio": mx.ones((1, 4, 8))}
    )

    assert with_audio != semantic_extra_hash(
        tenant="t", image_hash=7, media={"audio": mx.zeros((1, 4, 8))}
    )


def test_media_folding_is_key_order_independent():
    a, v = mx.ones((2, 2)), mx.zeros((3, 3))
    k1 = semantic_extra_hash(image_hash=1, media={"audio": a, "video": v})
    k2 = semantic_extra_hash(image_hash=1, media={"video": v, "audio": a})
    assert k1 == k2


def test_distinct_semantic_slots_do_not_collide():
    arr = mx.ones((2, 3))
    as_audio = semantic_extra_hash(image_hash=1, media={"audio": arr})
    as_video = semantic_extra_hash(image_hash=1, media={"video": arr})
    assert as_audio != as_video


def test_model_processor_hook_contributes_and_is_defensive():
    base = semantic_extra_hash(image_hash=5)

    contributor = SimpleNamespace(apc_key_dependencies=lambda: ["adapter-x"])
    assert semantic_extra_hash(image_hash=5, model=contributor) != base

    plain = SimpleNamespace(foo=1)
    boom = SimpleNamespace(
        apc_key_dependencies=lambda: (_ for _ in ()).throw(ValueError)
    )
    not_callable = SimpleNamespace(apc_key_dependencies=5)
    assert semantic_extra_hash(image_hash=5, model=plain) == base
    assert semantic_extra_hash(image_hash=5, model=boom) == base
    assert semantic_extra_hash(image_hash=5, model=not_callable) == base
    assert model_key_dependencies(None, None) == ()


def test_hash_payload_none_list_and_ref():
    assert _hash_payload(None) is None
    assert _hash_payload([]) is None
    assert _hash_payload(["a.png", "b.png"]) == _hash_payload(["a.png", "b.png"])
    assert _hash_payload("x") == hash_image_payload(image_ref="x")


def test_disk_namespace_is_stable_and_fingerprinted():
    ns = apc_disk_namespace("org/model", kv_bits=4, kv_group_size=64)

    assert ns.startswith("org/model#s%d-" % ADAPTER_SCHEMA_VERSION)
    assert ns == apc_disk_namespace("org/model", kv_bits=4, kv_group_size=64)

    assert ns != apc_disk_namespace("org/other", kv_bits=4, kv_group_size=64)
    assert ns != apc_disk_namespace("org/model", kv_bits=8, kv_group_size=64)
    assert ns != apc_disk_namespace("org/model", kv_bits=None)


def test_disk_namespace_distinguishes_adapters_and_weights():
    base = apc_disk_namespace("org/model", kv_bits=4)
    a = apc_disk_namespace("org/model", adapter_path="adapter-a", kv_bits=4)
    b = apc_disk_namespace("org/model", adapter_path="adapter-b", kv_bits=4)
    assert base != a and a != b
    assert base != apc_disk_namespace(
        "org/model", weights_fingerprint="rev-2", kv_bits=4
    )


def test_tensor_hash_is_shape_aware_and_lossless():
    same_bytes_diff_shape = _hash_payload(mx.array([[1, 2, 3, 4]])) != _hash_payload(
        mx.array([[1, 2], [3, 4]])
    )
    assert same_bytes_diff_shape
    x = mx.array([1.0], dtype=mx.float32)
    y = mx.array([1.0001], dtype=mx.float32)
    assert _hash_payload(x) != _hash_payload(y)


def test_processed_feature_tensor_changes_the_key():
    def audio_key(input_features):
        payload = input_features if input_features is not None else None
        return semantic_extra_hash(image_hash=0, media={"audio": payload})

    assert audio_key(mx.zeros((1, 80, 10))) != audio_key(mx.ones((1, 80, 10)))
