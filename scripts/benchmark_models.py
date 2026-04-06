#!/usr/bin/env python3
"""Benchmark models for agentic use on bastion.

Usage: python3 scripts/benchmark_models.py [--models model1,model2,...]

Tests: generation speed, prefill speed, tool calling, code quality,
instruction following, and context capacity.
"""

import json
import sys
import time

import requests

BASE = "http://100.106.192.127:8080"

MODELS = [
    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "inferencerlabs/Qwen3.5-35B-A3B-MLX-5.5bit",
    "unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def api_call(model, input_text, max_tokens=50, tools=None, temperature=0, timeout=300):
    payload = {
        "model": model,
        "input": input_text,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/v1/responses", json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code == 200:
            return r.json(), elapsed
        else:
            return {"error": r.status_code, "detail": r.text[:100]}, elapsed
    except Exception as e:
        return {"error": str(e)}, time.time() - t0


def test_generation_speed(model):
    """Measure tok/s on a simple prompt."""
    prompt = "Write a detailed story about a wizard. " * 5
    d, elapsed = api_call(model, prompt, max_tokens=200, temperature=0.7)
    if "error" in d:
        return None, None
    u = d.get("usage", {})
    out = u.get("output_tokens", 0)
    tps = out / elapsed if elapsed > 0 else 0
    return tps, u.get("input_tokens", 0)


def test_prefill_speed(model):
    """Measure prefill tok/s on a large prompt."""
    prompt = "Fox dog cat bird fish. " * 2000  # ~12K tokens
    d, elapsed = api_call(model, prompt + " Answer: yes.", max_tokens=5)
    if "error" in d:
        return None
    inp = d.get("usage", {}).get("input_tokens", 0)
    return inp / elapsed if elapsed > 0 else 0


def test_tool_calling(model):
    """Test structured tool call generation."""
    d, _ = api_call(
        model,
        "Search the web for Python 3.14 release date",
        max_tokens=100,
        tools=TOOLS,
    )
    if "error" in d:
        return False
    return any(i.get("type") == "function_call" for i in d.get("output", []))


def test_code_quality(model):
    """Test code generation quality."""
    d, _ = api_call(
        model,
        "Write a Python function implementing binary search with type hints and docstring. Only code, no explanation.",
        max_tokens=300,
    )
    if "error" in d:
        return 0
    text = ""
    for i in d.get("output", []):
        if i.get("type") == "message":
            text += "".join(p.get("text", "") for p in i.get("content", []))
    score = 0
    if "def " in text and "binary" in text.lower():
        score += 1
    if "->" in text:
        score += 1  # type hints
    if '"""' in text or "'''" in text:
        score += 1  # docstring
    return score


def test_instruction_following(model):
    """Test precise instruction following."""
    d, _ = api_call(
        model,
        "Do exactly these 3 things:\n1. Output ALPHA\n2. Output 42\n3. Output OMEGA\n\nOne per line, no other text.",
        max_tokens=30,
    )
    if "error" in d:
        return 0
    text = ""
    for i in d.get("output", []):
        if i.get("type") == "message":
            text += "".join(p.get("text", "") for p in i.get("content", []))
    score = 0
    if "ALPHA" in text:
        score += 1
    if "42" in text:
        score += 1
    if "OMEGA" in text:
        score += 1
    return score


def run_benchmarks(models):
    results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"{'='*60}")

        r = {}

        # Gen speed
        print("  Generation speed...", end=" ", flush=True)
        tps, inp_tok = test_generation_speed(model)
        r["gen_tps"] = tps
        print(f"{tps:.1f} tok/s" if tps else "FAILED")

        # Prefill speed
        print("  Prefill speed...", end=" ", flush=True)
        prefill = test_prefill_speed(model)
        r["prefill_tps"] = prefill
        print(f"{prefill:.0f} tok/s" if prefill else "FAILED")

        # Tool calling
        print("  Tool calling...", end=" ", flush=True)
        tools_ok = test_tool_calling(model)
        r["tool_call"] = tools_ok
        print("YES" if tools_ok else "NO")

        # Code quality
        print("  Code quality...", end=" ", flush=True)
        code = test_code_quality(model)
        r["code_quality"] = code
        print(f"{code}/3")

        # Instruction following
        print("  Instruction following...", end=" ", flush=True)
        inst = test_instruction_following(model)
        r["instruction"] = inst
        print(f"{inst}/3")

        results[model] = r

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'Model':<45} {'Gen':>7} {'Prefill':>8} {'Tools':>6} {'Code':>5} {'Inst':>5}")
    print(f"  {'-'*45} {'-'*7} {'-'*8} {'-'*6} {'-'*5} {'-'*5}")
    for model, r in results.items():
        name = model.split("/")[-1][:44]
        gen = f"{r['gen_tps']:.0f}" if r["gen_tps"] else "FAIL"
        pre = f"{r['prefill_tps']:.0f}" if r["prefill_tps"] else "FAIL"
        tools = "YES" if r["tool_call"] else "NO"
        code = f"{r['code_quality']}/3"
        inst = f"{r['instruction']}/3"
        print(f"  {name:<45} {gen:>7} {pre:>8} {tools:>6} {code:>5} {inst:>5}")


if __name__ == "__main__":
    models = MODELS
    if len(sys.argv) > 1 and sys.argv[1] == "--models":
        models = sys.argv[2].split(",")
    run_benchmarks(models)
