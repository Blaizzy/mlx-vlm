"""Tests for the complete semantic APC cache key (phase 6 of #1629)."""

from __future__ import annotations

from types import SimpleNamespace

from mlx_vlm.apc import (
    _hash_payload,
    hash_image_payload,
    model_key_dependencies,
    semantic_extra_hash,
)


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
