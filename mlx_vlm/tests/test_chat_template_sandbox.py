"""Sandboxed Jinja rendering for processor chat templates."""

from __future__ import annotations

import importlib
from datetime import datetime

import pytest
from jinja2 import Template
from jinja2.exceptions import SecurityError, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mlx_vlm.models.chat_template import render_chat_template

NORMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)

NORMAL_MESSAGES = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
]

PROCESSOR_SPECS = (
    ("mlx_vlm.models.molmo2.processing", "Molmo2Processor"),
    ("mlx_vlm.models.molmo.processing_molmo", "MolmoProcessor"),
    (
        "mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl",
        "Ernie4_5_VLProcessor",
    ),
    ("mlx_vlm.models.kimi_vl.processing_kimi_vl", "KimiVLProcessor"),
    ("mlx_vlm.models.phi3_v.processing_phi3_v", "Phi3VProcessor"),
    (
        "mlx_vlm.models.locateanything.processing_locateanything",
        "LocateAnythingProcessor",
    ),
)

ATTR_ESCAPE_TEMPLATES = (
    "{{ ''.__class__.__mro__ }}",
    "{{ ''.__class__.__subclasses__ }}",
)


def _processor(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return object.__new__(getattr(module, class_name))


def test_normal_chat_template_renders_identically():
    expected = Template(NORMAL_CHAT_TEMPLATE).render(
        messages=NORMAL_MESSAGES,
        add_generation_prompt=True,
    )
    assert expected == "User: hello\nAssistant: hi there\nAssistant: "
    assert (
        render_chat_template(
            NORMAL_CHAT_TEMPLATE,
            messages=NORMAL_MESSAGES,
            add_generation_prompt=True,
        )
        == expected
    )


@pytest.mark.parametrize("template", ATTR_ESCAPE_TEMPLATES)
def test_attribute_escape_is_rejected(template: str):
    Template(template).render()
    with pytest.raises(SecurityError):
        render_chat_template(template)


def test_chat_template_env_is_sandboxed():
    from mlx_vlm.models.chat_template import _get_chat_template_env

    assert isinstance(_get_chat_template_env(), ImmutableSandboxedEnvironment)


def test_tojson_raise_exception_and_strftime_now_keep_working():
    assert (
        render_chat_template("{{ data|tojson }}", data={"a": "<b>"}) == '{"a": "<b>"}'
    )
    with pytest.raises(TemplateError, match="boom"):
        render_chat_template('{{ raise_exception("boom") }}')
    assert render_chat_template(
        '{{ strftime_now("%Y-%m-%d") }}'
    ) == datetime.now().strftime("%Y-%m-%d")


def test_loopcontrols_continue_is_supported():
    rendered = render_chat_template(
        "{% for x in items %}"
        "{% if x == 2 %}{% continue %}{% endif %}"
        "{{ x }}"
        "{% endfor %}",
        items=[1, 2, 3],
    )
    assert rendered == "13"


@pytest.mark.parametrize("module_name,class_name", PROCESSOR_SPECS)
def test_processors_render_normal_templates_identically(module_name, class_name):
    expected = Template(NORMAL_CHAT_TEMPLATE).render(
        messages=NORMAL_MESSAGES,
        add_generation_prompt=True,
    )
    rendered = _processor(module_name, class_name).apply_chat_template(
        NORMAL_MESSAGES,
        chat_template=NORMAL_CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert rendered == expected


@pytest.mark.parametrize("module_name,class_name", PROCESSOR_SPECS)
@pytest.mark.parametrize("template", ATTR_ESCAPE_TEMPLATES)
def test_processors_reject_attribute_escape(module_name, class_name, template):
    with pytest.raises(SecurityError):
        _processor(module_name, class_name).apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template=template,
            tokenize=False,
        )
