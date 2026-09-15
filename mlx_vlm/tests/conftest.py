import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


@pytest.fixture
def template_processor():
    """Actual pinned templates and the Transformers renderer; no model or vocab."""
    from transformers.utils.chat_template_utils import render_jinja_template

    def load(name):
        template = (
            Path(__file__).parent / "fixtures" / "chat_templates" / f"{name}.jinja"
        ).read_text()

        def render(messages, *, tokenize=False, **kwargs):
            assert tokenize is False
            return render_jinja_template(
                [messages], chat_template=template, bos_token="<bos>", **kwargs
            )[0][0]

        return SimpleNamespace(chat_template=template, apply_chat_template=render)

    return load


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()
