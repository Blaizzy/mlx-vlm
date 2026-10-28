"""Sandboxed Jinja rendering for model chat templates."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

_CHAT_TEMPLATE_ENV = None


def _get_chat_template_env():
    """Return a shared sandboxed Jinja environment for chat templates.

    Matches Hugging Face transformers: ``ImmutableSandboxedEnvironment`` with
    ``trim_blocks`` / ``lstrip_blocks``, ``loopcontrols``, and the ``tojson``,
    ``raise_exception``, and ``strftime_now`` helpers.
    """
    global _CHAT_TEMPLATE_ENV
    if _CHAT_TEMPLATE_ENV is not None:
        return _CHAT_TEMPLATE_ENV

    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise ImportError("jinja2 is required for apply_chat_template") from exc

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # Override Jinja's HTML-escaping tojson, matching Hugging Face transformers.
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format):
        return datetime.now().strftime(format)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    _CHAT_TEMPLATE_ENV = env
    return env


def compile_chat_template(chat_template: str):
    """Compile a chat template with a sandboxed Jinja environment."""
    return _get_chat_template_env().from_string(chat_template)


def render_chat_template(chat_template: str, **context: Any) -> str:
    """Render a chat template with a sandboxed Jinja environment."""
    return compile_chat_template(chat_template).render(**context)
