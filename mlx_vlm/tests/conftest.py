import json
import os
from pathlib import Path

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


@pytest.fixture(
    params=[
        case
        for case in json.loads(
            Path(__file__).with_name("model_cases.json").read_text()
        )["cases"]
        if "decision" in case["checks"]
    ],
    ids=lambda case: case["id"],
)
def decision_case(request):
    import copy

    return copy.deepcopy(request.param)


@pytest.fixture
def decision_model(decision_case):
    from mlx_vlm.tests.test_models import _model_for_case

    return _model_for_case(decision_case)


@pytest.fixture
def decision_processor(decision_case):
    from types import SimpleNamespace

    from mlx_vlm.tests.test_processors import _DecisionTokenizer

    return SimpleNamespace(
        tokenizer=_DecisionTokenizer(decision_case["decision"]["tokenizer_vocab_size"])
    )
