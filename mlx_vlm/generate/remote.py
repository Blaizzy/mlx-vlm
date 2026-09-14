"""Server-aware execution for ``mlx_vlm generate``.

With ``--base-url`` (or ``$MLX_VLM_BASE_URL``) set, forward the request to a
running mlx-vlm server and stream the reply; otherwise generate locally.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Options that change the model itself and cannot be honored over the chat API.
_LOCAL_ONLY = ("adapter_path", "draft_model", "quantize_activations")


def resolve_base_url(args):
    """--base-url, then $MLX_VLM_BASE_URL, else None. None means run locally."""
    return getattr(args, "base_url", None) or os.environ.get("MLX_VLM_BASE_URL")


def _auth_header(args):
    key = getattr(args, "api_key", None) or os.environ.get("MLX_VLM_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _prompt_text(args):
    """--prompt is nargs='+' (a list of words) or the string default."""
    prompt = args.prompt
    return " ".join(prompt) if isinstance(prompt, list) else str(prompt)


def build_messages(args):
    """Build chat messages; image paths/URLs pass through for the server to resolve."""
    messages = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})

    text = _prompt_text(args)
    images = getattr(args, "image", None) or []
    if isinstance(images, str):
        images = [images]
    if images:
        content = [{"type": "text", "text": text}]
        content += [
            {"type": "image_url", "image_url": {"url": os.path.expanduser(path)}}
            for path in images
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


def sampling_params(args):
    """Build the request body: OpenAI-spec fields, plus mlx-vlm extras only when
    engaged, so a default request stays spec-compliant and portable."""
    params = {}
    # OpenAI spec fields.
    for name in (
        "temperature",
        "max_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value

    # mlx-vlm extras — added only when actually engaged, never at their defaults.
    if getattr(args, "repetition_penalty", None) is not None:
        params["repetition_penalty"] = args.repetition_penalty
    for penalty, ctx in (
        ("repetition_penalty", "repetition_context_size"),
        ("frequency_penalty", "frequency_context_size"),
        ("presence_penalty", "presence_context_size"),
    ):
        if getattr(args, penalty, None) is not None:
            value = getattr(args, ctx, None)
            if value is not None:
                params[ctx] = value
    if getattr(args, "enable_thinking", False):
        params["enable_thinking"] = True
    if getattr(args, "thinking_budget", None) is not None:
        params["thinking_budget"] = args.thinking_budget
        for name in ("thinking_start_token", "thinking_end_token"):
            value = getattr(args, name, None)
            if value is not None:
                params[name] = value
    return params


def stream_chat(base_url, model, messages, params, args=None):
    """POST a streaming chat completion; yield text deltas from the SSE body."""
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
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            content = delta.get("content")
            if content:
                yield content


def _local_only_conflict(args):
    for name in _LOCAL_ONLY:
        if getattr(args, name, None):
            return name.replace("_", "-")
    return None


def run_on_server(args):
    """Serve remotely when a base URL is set: True if handled, False for local."""
    base_url = resolve_base_url(args)
    if not base_url:
        return False

    conflict = _local_only_conflict(args)
    if conflict:
        raise SystemExit(f"--{conflict} requires local execution; drop --base-url.")

    messages = build_messages(args)
    params = sampling_params(args)
    if getattr(args, "verbose", False):
        print(f"Using server at {base_url}.", file=sys.stderr)
    try:
        for chunk in stream_chat(base_url, args.model, messages, params, args):
            print(chunk, end="", flush=True)
        print()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:200]
        except Exception:
            detail = ""
        raise SystemExit(f"server at {base_url} returned HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"server at {base_url} is unreachable ({exc.reason}); "
            "start it or drop --base-url."
        )
    return True
