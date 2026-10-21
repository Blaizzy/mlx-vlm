import os

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Reuse compiled kernels across variants without serializing a whole suite."""
    # xdist records loadgroup separately before setting each worker's dist to no.
    if config.getoption("dist", default="no") != "loadgroup" and not getattr(
        config.option, "loadgroup", False
    ):
        return
    for item in items:
        group = f"{item.path.stem}.{getattr(item, 'originalname', None) or item.name}"
        params = getattr(getattr(item, "callspec", None), "params", {})
        case = params.get("case")
        if isinstance(case, dict) and "module" in case:
            group += f".{case['module']}"
        elif isinstance(params.get("family"), str):
            group += f".{params['family']}"
        item.add_marker(pytest.mark.xdist_group(group))


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()
