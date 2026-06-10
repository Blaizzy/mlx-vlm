from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable

from .model_catalog import local_model_infos


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_CONTEXT_WINDOW = 131_072
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_PROVIDER_ID = "mlx-vlm"
DEFAULT_PROVIDER_NAME = "MLX-VLM Local"
CLIENTS = ("pi", "hermes", "opencode")


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _default_pi_config() -> Path:
    return _expand_path(
        os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")
    ) / "models.json"


def _default_hermes_config() -> Path:
    return _expand_path(os.environ.get("HERMES_HOME", "~/.hermes")) / "config.yaml"


def _default_opencode_config() -> Path:
    return _expand_path(
        os.environ.get("OPENCODE_CONFIG", "~/.config/opencode/opencode.json")
    )


def _model_name(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1]
    return re.sub(r"[-_]+", " ", name).strip() or model_id


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak-{int(time.time())}")
    shutil.copy2(path, backup_path)
    return backup_path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def _json_yaml(value: str) -> str:
    return json.dumps(value)


def _yaml_unquote(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1]
        except json.JSONDecodeError:
            return value.strip("'\"")
    return value


def _find_top_level_block(lines: list[str], key: str) -> tuple[int, int] | None:
    key_pattern = re.compile(rf"^{re.escape(key)}\s*:")
    start = next((idx for idx, line in enumerate(lines) if key_pattern.match(line)), -1)
    if start < 0:
        return None

    top_level_key = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if top_level_key.match(lines[idx]):
            end = idx
            break
    return start, end


def _extract_hermes_default_model(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _find_top_level_block(lines, "model")
    if block is None:
        return ""
    for line in lines[block[0] + 1 : block[1]]:
        match = re.match(r"\s+default\s*:\s*(.*)\s*$", line)
        if match:
            return _yaml_unquote(match.group(1))
    return ""


def _hermes_model_block(default_model: str, base_url: str, api_key: str) -> list[str]:
    lines = [
        "model:\n",
        f"  default: {_json_yaml(default_model)}\n",
        "  provider: custom\n",
        f"  base_url: {_json_yaml(base_url)}\n",
    ]
    if api_key:
        lines.append(f"  api_key: {_json_yaml(api_key)}\n")
    return lines


def _hermes_custom_provider_entry(
    *,
    provider_name: str,
    base_url: str,
    api_key: str,
    default_model: str,
    model_ids: Iterable[str],
    context_window: int,
) -> list[str]:
    lines = [
        f"- name: {_json_yaml(provider_name)}\n",
        f"  base_url: {_json_yaml(base_url)}\n",
        "  api_mode: chat_completions\n",
        f"  model: {_json_yaml(default_model)}\n",
    ]
    if api_key:
        lines.insert(2, f"  api_key: {_json_yaml(api_key)}\n")
    if model_ids:
        lines.append("  models:\n")
        for model_id in model_ids:
            lines.extend(
                [
                    f"    {_json_yaml(model_id)}:\n",
                    f"      context_length: {context_window}\n",
                ]
            )
    return lines


def _split_yaml_list_entries(lines: list[str]) -> list[list[str]]:
    entries: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"\s*-\s+", line):
            if current:
                entries.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
            elif line.strip():
                entries.append([line])
    if current:
        entries.append(current)
    return entries


def _custom_provider_entry_matches(
    entry: list[str], *, provider_name: str, base_url: str
) -> bool:
    normalized_base_url = base_url.rstrip("/")
    for line in entry:
        name_match = re.match(r"\s*-?\s*name\s*:\s*(.*)\s*$", line)
        if name_match and _yaml_unquote(name_match.group(1)) == provider_name:
            return True
        url_match = re.match(r"\s*base_url\s*:\s*(.*)\s*$", line)
        if (
            url_match
            and _yaml_unquote(url_match.group(1)).rstrip("/") == normalized_base_url
        ):
            return True
    return False


def patch_hermes_config_text(
    text: str,
    *,
    model_ids: list[str],
    default_model: str,
    base_url: str,
    api_key: str,
    provider_name: str,
    context_window: int,
) -> str:
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        lines[-1] = lines[-1] + "\n"

    model_block = _find_top_level_block(lines, "model")
    replacement = _hermes_model_block(default_model, base_url, api_key)
    if model_block is None:
        lines = replacement + (["\n"] if lines else []) + lines
    else:
        lines = lines[: model_block[0]] + replacement + lines[model_block[1] :]

    provider_entry = _hermes_custom_provider_entry(
        provider_name=provider_name,
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        model_ids=model_ids,
        context_window=context_window,
    )
    custom_block = _find_top_level_block(lines, "custom_providers")
    if custom_block is None:
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(["custom_providers:\n", *provider_entry])
    else:
        body = lines[custom_block[0] + 1 : custom_block[1]]
        kept_entries = [
            entry
            for entry in _split_yaml_list_entries(body)
            if not _custom_provider_entry_matches(
                entry, provider_name=provider_name, base_url=base_url
            )
        ]
        new_block = ["custom_providers:\n"]
        for entry in kept_entries:
            new_block.extend(entry)
        new_block.extend(provider_entry)
        lines = lines[: custom_block[0]] + new_block + lines[custom_block[1] :]

    return "".join(lines)


def pi_config(
    existing: dict,
    *,
    model_ids: list[str],
    base_url: str,
    api_key: str,
    provider_id: str,
    provider_name: str,
    context_window: int,
    max_tokens: int,
) -> dict:
    data = dict(existing)
    providers = dict(data.get("providers") or {})
    providers[provider_id] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": api_key,
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": False,
            "maxTokensField": "max_tokens",
        },
        "models": [
            {
                "id": model_id,
                "name": f"{_model_name(model_id)} ({provider_name})",
                "contextWindow": context_window,
                "maxTokens": max_tokens,
            }
            for model_id in model_ids
        ],
    }
    data["providers"] = providers
    return data


def opencode_config(
    existing: dict,
    *,
    model_ids: list[str],
    base_url: str,
    api_key: str,
    provider_id: str,
    provider_name: str,
) -> dict:
    data = dict(existing)
    data.setdefault("$schema", "https://opencode.ai/config.json")
    providers = dict(data.get("provider") or {})
    providers[provider_id] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": provider_name,
        "options": {
            "baseURL": base_url,
            "apiKey": api_key,
        },
        "models": {
            model_id: {"name": _model_name(model_id)} for model_id in model_ids
        },
    }
    data["provider"] = providers
    return data


def resolve_default_model(
    model_ids: list[str],
    explicit_default: str | None,
    hermes_config: Path,
) -> str:
    by_lower = {model_id.lower(): model_id for model_id in model_ids}
    if explicit_default:
        resolved = by_lower.get(explicit_default.lower())
        if not resolved:
            raise ValueError(
                f"Default model {explicit_default!r} is not available in the HF cache"
            )
        return resolved

    current = _extract_hermes_default_model(hermes_config)
    if current and current.lower() in by_lower:
        return by_lower[current.lower()]
    return model_ids[0]


def _client_error(unknown: Iterable[str] | None = None) -> argparse.ArgumentTypeError:
    valid = ", ".join(CLIENTS)
    if unknown:
        return argparse.ArgumentTypeError(
            f"unknown client(s): {', '.join(sorted(unknown))}. Expected one of: {valid}"
        )
    return argparse.ArgumentTypeError(f"expected one of: {valid}")


def _parse_client(value: str) -> tuple[str, ...]:
    client = value.strip().lower()
    if client not in CLIENTS:
        raise _client_error([client] if client else None)
    return (client,)


def _parse_clients(value: str) -> tuple[str, ...]:
    requested = tuple(
        item.strip().lower() for item in value.split(",") if item.strip()
    )
    unknown = sorted(set(requested) - set(CLIENTS))
    if not requested or unknown:
        raise _client_error(unknown or None)
    return tuple(dict.fromkeys(requested))


def configure_clients(args: argparse.Namespace) -> list[tuple[str, Path]]:
    model_infos = local_model_infos(sort=True)
    model_ids = [model["id"] for model in model_infos]
    if not model_ids:
        raise ValueError(
            "No supported MLX models were found in the Hugging Face cache."
        )

    default_model = resolve_default_model(
        model_ids, args.default_model, args.hermes_config
    )
    updated: list[tuple[str, Path]] = []

    def maybe_backup(path: Path) -> None:
        if not args.dry_run and args.backup:
            _backup(path)

    if "pi" in args.clients:
        path = args.pi_config
        next_config = pi_config(
            _read_json(path),
            model_ids=model_ids,
            base_url=args.base_url,
            api_key=args.api_key,
            provider_id=args.provider_id,
            provider_name=args.provider_name,
            context_window=args.context_window,
            max_tokens=args.max_tokens,
        )
        if not args.dry_run:
            maybe_backup(path)
            _write_json(path, next_config)
        updated.append(("pi", path))

    if "hermes" in args.clients:
        path = args.hermes_config
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        next_text = patch_hermes_config_text(
            text,
            model_ids=model_ids,
            default_model=default_model,
            base_url=args.base_url,
            api_key=args.api_key,
            provider_name=args.provider_name,
            context_window=args.context_window,
        )
        if not args.dry_run:
            maybe_backup(path)
            _atomic_write_text(path, next_text)
        updated.append(("hermes", path))

    if "opencode" in args.clients:
        path = args.opencode_config
        next_config = opencode_config(
            _read_json(path),
            model_ids=model_ids,
            base_url=args.base_url,
            api_key=args.api_key,
            provider_id=args.provider_id,
            provider_name=args.provider_name,
        )
        if not args.dry_run:
            maybe_backup(path)
            _write_json(path, next_config)
        updated.append(("opencode", path))

    print(
        f"{'Would configure' if args.dry_run else 'Configured'} "
        f"{len(updated)} client(s) with {len(model_ids)} HF cache model(s)."
    )
    print(f"Server: {args.base_url}")
    print(f"Provider: {args.provider_id}")
    print(f"Default model: {default_model}")
    for client, path in updated:
        print(f"{client}: {path}")
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set up pi, Hermes, and opencode to use the local MLX-VLM "
            "OpenAI-compatible server and cached HF models."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible server base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key to write for clients that require one.",
    )
    parser.add_argument(
        "--provider-id",
        default=DEFAULT_PROVIDER_ID,
        help=(
            "Provider id/name used by pi and opencode "
            f"(default: {DEFAULT_PROVIDER_ID})."
        ),
    )
    parser.add_argument(
        "--provider-name",
        default=DEFAULT_PROVIDER_NAME,
        help=f"Display name for the local provider (default: {DEFAULT_PROVIDER_NAME}).",
    )
    parser.add_argument(
        "--client",
        dest="clients",
        metavar="CLIENT",
        type=_parse_client,
        default=CLIENTS,
        help=(
            "Client to configure: pi, hermes, or opencode. "
            "Omit to configure all clients."
        ),
    )
    parser.add_argument(
        "--clients",
        dest="clients",
        type=_parse_clients,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--default-model",
        help="Default Hermes model. Must be present in the HF cache model list.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Context window metadata for generated client catalogs "
            f"(default: {DEFAULT_CONTEXT_WINDOW})."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max output token metadata for pi models (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--pi-config",
        type=_expand_path,
        default=_default_pi_config(),
        help="Path to pi models.json.",
    )
    parser.add_argument(
        "--hermes-config",
        type=_expand_path,
        default=_default_hermes_config(),
        help="Path to Hermes config.yaml.",
    )
    parser.add_argument(
        "--opencode-config",
        type=_expand_path,
        default=_default_opencode_config(),
        help="Path to opencode config JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be configured without writing files.",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not create .bak-* backups before overwriting existing files.",
    )
    parser.set_defaults(backup=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        configure_clients(args)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
