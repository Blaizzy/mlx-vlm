import importlib.util
from pathlib import Path


def _load_parser():
    parser_path = (
        Path(__file__).resolve().parents[2] / "computer_use" / "action_parser.py"
    )
    spec = importlib.util.spec_from_file_location(
        "computer_use_action_parser", parser_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_action_response


def test_parse_action_response_accepts_json_null():
    parse_action_response = _load_parser()

    parsed = parse_action_response(
        '{"action": "CLICK", "value": null, "position": [0.4, 0.81]}'
    )

    assert parsed == {"action": "CLICK", "value": None, "position": [0.4, 0.81]}


def test_parse_action_response_accepts_python_style_dict():
    parse_action_response = _load_parser()

    parsed = parse_action_response(
        "{'action': 'CLICK', 'value': None, 'position': [0.4, 0.81]}"
    )

    assert parsed == {"action": "CLICK", "value": None, "position": [0.4, 0.81]}


def test_parse_action_response_extracts_dict_from_fenced_output():
    parse_action_response = _load_parser()

    parsed = parse_action_response(
        '```json\n{"action": "WAIT", "value": null, "position": null}\n```'
    )

    assert parsed == {"action": "WAIT", "value": None, "position": None}
