from __future__ import annotations


class DummyTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        # Return a sentinel string so we can assert it was used.
        return "TEMPLATED_PROMPT"


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyConfig:
    def __init__(self):
        self.model_type = "gemma4"


class DummyModel:
    def __init__(self):
        self.config = DummyConfig()


def test_stream_generate_applies_chat_template_for_gemma4(monkeypatch):
    # Import inside test to avoid import-time side effects.
    import importlib

    gen = importlib.import_module("mlx_vlm.generate")

    proc = DummyProcessor()
    model = DummyModel()

    captured = {}

    def fake_prepare_inputs(*args, **kwargs):
        captured["prompts"] = kwargs.get("prompts")
        return {"input_ids": None, "pixel_values": None, "attention_mask": None}

    # Prevent actual generation and avoid dependency on MLX arrays
    def fake_generate_step(*args, **kwargs):
        if False:
            yield None

    monkeypatch.setattr(gen, "prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr(gen, "generate_step", fake_generate_step)

    # Call generator; it will hit prepare_inputs before trying to generate.
    g = gen.stream_generate(model, proc, "Hello", image=None, audio=None, max_tokens=1)
    try:
        next(g)
    except StopIteration:
        pass
    except Exception:
        # We only care that prepare_inputs saw the templated prompt.
        pass

    assert captured.get("prompts") == "TEMPLATED_PROMPT"


def test_stream_generate_can_skip_chat_template(monkeypatch):
    import importlib

    gen = importlib.import_module("mlx_vlm.generate")

    proc = DummyProcessor()
    model = DummyModel()
    captured = {}

    def fake_prepare_inputs(*args, **kwargs):
        captured["prompts"] = kwargs.get("prompts")
        return {"input_ids": None, "pixel_values": None, "attention_mask": None}

    monkeypatch.setattr(gen, "prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr(gen, "generate_step", lambda *a, **k: iter(()))

    g = gen.stream_generate(
        model,
        proc,
        "Hello",
        image=None,
        audio=None,
        max_tokens=1,
        skip_chat_template=True,
    )
    try:
        next(g)
    except StopIteration:
        pass
    except Exception:
        pass

    assert captured.get("prompts") == "Hello"

