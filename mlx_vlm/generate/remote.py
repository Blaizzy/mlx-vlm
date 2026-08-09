"""Optional server-aware execution for the generate CLI.

When an mlx-vlm server is already running, the CLI can send a plain text or
vision request to it instead of cold-loading the model into this process. The
local path remains authoritative for every option the OpenAI-compatible server
does not expose.
"""

import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_PREFILL_STEP_SIZE = 2048


class RemoteServerError(RuntimeError):
    """A reachable server could not satisfy a CLI request."""


def resolve_base_url(args):
    """Resolve --base-url, then $MLX_VLM_BASE_URL, then the local default."""
    return (
        getattr(args, "base_url", None)
        or os.environ.get("MLX_VLM_BASE_URL")
        or DEFAULT_BASE_URL
    )


def _auth_header(args):
    key = getattr(args, "api_key", None) or os.environ.get("MLX_VLM_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _http_error_message(exc):
    try:
        detail = exc.read().decode("utf-8", "replace").strip()
    except OSError:
        detail = ""
    suffix = f": {detail}" if detail else ""
    return f"mlx-vlm server returned HTTP {exc.code}{suffix}"


def server_available(base_url, args=None, timeout=1.5):
    """Return whether an mlx-vlm server answers its models endpoint.

    A connection failure permits automatic local fallback. A reachable server
    that returns an authentication or other HTTP error is surfaced instead of
    being silently bypassed by a separate local model load.
    """
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models",
        headers=_auth_header(args) if args is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise RemoteServerError(_http_error_message(exc)) from exc
    except OSError:
        return False


def eligible_for_server(args):
    """Whether every requested generate feature maps to the server API."""
    if getattr(args, "local", False):
        return False
    if getattr(args, "output_modality", "text") != "text":
        return False
    if not getattr(args, "model", None) or getattr(args, "prompt", None) is None:
        return False

    # --prompt is nargs="+" (a list of words) or the string default. A list
    # of message dicts is an internal representation, not a plain CLI prompt.
    prompt = args.prompt
    if isinstance(prompt, list) and not all(isinstance(part, str) for part in prompt):
        return False

    # Keep this conservative. These options either require local state or are
    # server-start configuration rather than request-level API fields.
    disqualifying = (
        getattr(args, "adapter_path", None),
        getattr(args, "draft_model", None),
        getattr(args, "quantize_activations", False),
        getattr(args, "chat", False),
        getattr(args, "audio", None),
        getattr(args, "video", None),
        getattr(args, "eos_tokens", None),
        getattr(args, "processor_kwargs", None),
        getattr(args, "gen_kwargs", None),
        getattr(args, "thinking_mode", None),
        getattr(args, "force_download", False),
        getattr(args, "trust_remote_code", False),
        getattr(args, "max_kv_size", None),
        getattr(args, "kv_bits", None),
        getattr(args, "kv_key_bits", None),
        getattr(args, "kv_value_bits", None),
        getattr(args, "kv_key_scheme", None),
        getattr(args, "kv_value_scheme", None),
    )
    if any(disqualifying):
        return False
    if getattr(args, "revision", "main") != "main":
        return False
    prefill_step_size = getattr(args, "prefill_step_size", None)
    return prefill_step_size in (None, DEFAULT_PREFILL_STEP_SIZE)


def _image_url(path):
    """Return an HTTP(S) image URL unchanged or encode a local path as data."""
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme in {"http", "https", "data"}:
        return path
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(os.path.expanduser(path), "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _prompt_text(args):
    """Join argparse's nargs prompt representation into one text message."""
    prompt = args.prompt
    return " ".join(prompt) if isinstance(prompt, list) else str(prompt)


def build_messages(args):
    """Map the CLI prompt, system message, and images to chat messages."""
    messages = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})

    text = _prompt_text(args)
    images = getattr(args, "image", None) or []
    if isinstance(images, str):
        images = [images]
    if images:
        content = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_url(path)}}
            for path in images
        )
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


def sampling_params(args):
    """Map request-level CLI controls to the server schema."""
    params = {}
    for cli_name, api_name in (
        ("temperature", "temperature"),
        ("max_tokens", "max_tokens"),
        ("repetition_penalty", "repetition_penalty"),
        ("repetition_context_size", "repetition_context_size"),
        ("frequency_penalty", "frequency_penalty"),
        ("frequency_context_size", "frequency_context_size"),
        ("presence_penalty", "presence_penalty"),
        ("presence_context_size", "presence_context_size"),
        ("seed", "seed"),
        ("resize_shape", "resize_shape"),
        ("thinking_budget", "thinking_budget"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            params[api_name] = value
    # The local CLI always passes this boolean into its chat template, including
    # its false default. Do the same remotely rather than inheriting a server
    # process's unrelated default.
    params["enable_thinking"] = getattr(args, "enable_thinking", False)
    if getattr(args, "thinking_budget", None) is not None:
        for cli_name in ("thinking_start_token", "thinking_end_token"):
            value = getattr(args, cli_name, None)
            if value is not None:
                params[cli_name] = value
    return params


def stream_chat(base_url, model, messages, params, args=None):
    """POST a streaming chat request and yield text deltas from its SSE body."""
    body = {"model": model, "messages": messages, "stream": True, **params}
    headers = {"Content-Type": "application/json", **_auth_header(args)}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except (IndexError, json.JSONDecodeError, KeyError, TypeError):
                continue
            content = delta.get("content")
            if content:
                yield content


def run_on_server(args):
    """Run one eligible request remotely, else return ``False`` for local use."""
    forced = getattr(args, "server", False)
    if not eligible_for_server(args):
        if forced:
            raise RemoteServerError(
                "--server cannot be used with this combination of generate options"
            )
        return False

    base_url = resolve_base_url(args)
    if not server_available(base_url, args):
        if forced:
            raise RemoteServerError(f"No mlx-vlm server is available at {base_url}")
        return False

    verbose = getattr(args, "verbose", False)
    if verbose:
        print(f"Using server at {base_url} (model: {args.model}).", file=sys.stderr)

    received = False
    chunks = []
    try:
        for chunk in stream_chat(
            base_url, args.model, build_messages(args), sampling_params(args), args
        ):
            received = True
            if verbose:
                print(chunk, end="", flush=True)
            else:
                chunks.append(chunk)
        if verbose:
            if received:
                print()
        else:
            print("".join(chunks))
        return True
    except urllib.error.HTTPError as exc:
        raise RemoteServerError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        # A request that never produced output may race a just-stopped server;
        # retrying locally is safe only in that case.
        if forced or received:
            raise RemoteServerError(f"mlx-vlm server request failed: {exc}") from exc
        if verbose:
            print(f"Server request failed ({exc}); loading locally.", file=sys.stderr)
        return False
