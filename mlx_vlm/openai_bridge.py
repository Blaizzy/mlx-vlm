"""Small terminal client for calling the OpenAI Responses API."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TIMEOUT = 60.0


class BridgeConfigurationError(ValueError):
    """Raised when bridge configuration is missing or invalid."""


class BridgeRequestError(RuntimeError):
    """Raised when the Responses API cannot be reached or rejects a request."""


@dataclass(frozen=True)
class BridgeConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT


def load_config(environ: dict[str, str] | None = None) -> BridgeConfig:
    env = os.environ if environ is None else environ
    api_key = env.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise BridgeConfigurationError(
            "OPENAI_API_KEY is required; set it in the environment, not in a script."
        )

    base_url = env.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BridgeConfigurationError(
            "OPENAI_BASE_URL must be an absolute http(s) URL."
        )

    model = env.get("OPENAI_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise BridgeConfigurationError("OPENAI_MODEL must not be empty.")

    try:
        timeout = float(env.get("OPENAI_TIMEOUT", str(DEFAULT_TIMEOUT)))
    except ValueError as exc:
        raise BridgeConfigurationError("OPENAI_TIMEOUT must be a positive number.") from exc
    if timeout <= 0:
        raise BridgeConfigurationError("OPENAI_TIMEOUT must be a positive number.")

    return BridgeConfig(api_key, base_url, model, timeout)


def build_request(
    prompt: str,
    config: BridgeConfig,
    *,
    system: str | None = None,
) -> Request:
    if not prompt.strip():
        raise BridgeConfigurationError("Prompt must not be empty.")
    content = [{"type": "input_text", "text": prompt}]
    body: dict[str, Any] = {
        "model": config.model,
        "input": [{"role": "user", "content": content}],
    }
    if system:
        body["instructions"] = system
    return Request(
        f"{config.base_url}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise BridgeRequestError("Responses API returned no output text.")
    return "".join(parts)


def call_responses(
    prompt: str,
    config: BridgeConfig,
    *,
    system: str | None = None,
) -> dict[str, Any]:
    request = build_request(prompt, config, system=system)
    try:
        with urlopen(request, timeout=config.timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise BridgeRequestError(
            f"OpenAI Responses API returned HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise BridgeRequestError(f"Unable to reach OpenAI Responses API: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BridgeRequestError("OpenAI Responses API returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise BridgeRequestError("OpenAI Responses API returned an unexpected response.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the OpenAI Responses API from a local MLX terminal workflow."
    )
    parser.add_argument("--prompt", help="Prompt text; reads stdin when omitted.")
    parser.add_argument("--system", help="Optional system/developer instruction.")
    parser.add_argument("--model", help="Override OPENAI_MODEL for this request.")
    parser.add_argument(
        "--json", action="store_true", help="Print the complete API response as JSON."
    )
    args = parser.parse_args()

    try:
        config = load_config()
        if args.model:
            config = BridgeConfig(
                config.api_key, config.base_url, args.model, config.timeout
            )
        prompt = args.prompt if args.prompt is not None else sys.stdin.read()
        payload = call_responses(prompt, config, system=args.system)
        print(json.dumps(payload) if args.json else _response_text(payload))
    except (BridgeConfigurationError, BridgeRequestError) as exc:
        parser.error(str(exc))
