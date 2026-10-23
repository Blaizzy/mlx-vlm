import json
import os
from pathlib import Path

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()


@pytest.fixture(
    params=[
        case
        for case in json.loads(
            Path(__file__).with_name("extraction_cases.json").read_text()
        )["cases"]
        if "decision_engine" in case
    ],
    ids=lambda case: case["id"],
)
def incremental_decision(request):
    from mlx_vlm.tests.test_extraction_models import _decision_fixture

    return request.param, _decision_fixture(request.param)
