from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.speculative.drafters.deepseek_v4_mtp.deepseek_v4_mtp import (
    DeepseekV4MTPDraftModel,
)
from mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant import (
    Gemma4AssistantDraftModel,
)
from mlx_vlm.speculative.drafters.mtp_base import AutoregressiveMTPDraftModel
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
    Qwen4ExpMTPDraftModel,
)


@pytest.mark.parametrize("cls", [Qwen4ExpMTPDraftModel, DeepseekV4MTPDraftModel])
def test_stateful_capabilities_are_inherited_from_lifecycle_owner(cls):
    assert "supports_continuous_batching" not in cls.__dict__
    assert cls.supports_continuous_batching is True
    assert cls.requires_verified_token_reconciliation is True
    assert cls.filter_batch is AutoregressiveMTPDraftModel.filter_batch
    assert callable(cls.accept_verified_tokens_batch)


def test_shared_kv_assistant_declares_stateless_batching():
    assert Gemma4AssistantDraftModel.supports_continuous_batching is True
    assert Gemma4AssistantDraftModel.requires_verified_token_reconciliation is False


def test_base_row_filter_preserves_owned_seed_cache_and_position_alignment():
    drafter = AutoregressiveMTPDraftModel(SimpleNamespace())
    drafter._seed_token = mx.array([[10], [20], [30]])
    drafter._seed_hidden = mx.array([[[1]], [[2]], [[3]]])
    drafter._next_position = mx.array([4, 5, 6])
    keys = mx.arange(6).reshape(3, 1, 2, 1)
    drafter._cache = [SimpleNamespace(keys=keys, values=keys + 10)]
    drafter.filter_batch([2, 0])
    assert drafter._seed_token.tolist() == [[30], [10]]
    assert drafter._seed_hidden.tolist() == [[[3]], [[1]]]
    assert drafter._next_position.tolist() == [6, 4]
    assert drafter._cache[0].keys.tolist() == keys[[2, 0]].tolist()
