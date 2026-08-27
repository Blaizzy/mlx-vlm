import ast
import json


def parse_action_response(response):
    """Parse a GUI action emitted as JSON or a Python-style dict string."""
    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        raise TypeError(
            f"Expected a string or dict response, got {type(response).__name__}"
        )

    response = response.strip()
    if response.startswith("```"):
        lines = response.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and start < end:
        response = response[start : end + 1]

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(response)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected action response to parse to a dict, got {type(parsed).__name__}"
        )

    return parsed
