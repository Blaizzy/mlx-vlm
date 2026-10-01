"""Opt-in real-model HTTP/APC test.

MLX_VLM_COMPACTION_TEST_MODEL=openbmb/MiniCPM5-2B pytest -s \
    mlx_vlm/tests/test_compaction_model.py
"""

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import httpx
import pytest

MODEL = os.environ.get("MLX_VLM_COMPACTION_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="Opt-in model inference test")


@contextmanager
def running_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ)
    env.update(
        APC_ENABLED="1",
        APC_DISK_ENABLED="0",
        APC_NUM_BLOCKS="2048",
        MLX_VLM_COMPACTION_KEY_FILE=str(tmp_path / "compaction.key"),
    )
    env.pop("MLX_VLM_SERVER_API_KEY", None)
    with (tmp_path / "server.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--model",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=180,
                headers={"X-APC-Tenant": "compaction-test"},
            ) as client:
                for _ in range(240):
                    assert process.poll() is None, (tmp_path / "server.log").read_text()
                    try:
                        if client.get("/health", timeout=1).is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.25)
                else:
                    pytest.fail("Model server did not become ready")
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def post(client, path, **body):
    start = time.monotonic()
    response = client.post(path, json={"model": MODEL, **body})
    assert response.is_success, response.text
    return response.json(), round(time.monotonic() - start, 3)


def test_real_compaction_apc_and_restart(tmp_path):
    from openai import OpenAI

    history = [
        {
            "role": "system",
            "content": "Follow the user's requirements. Answer factual questions concisely.",
        },
        {
            "role": "user",
            "content": "We are working on project ORCHID. The deployment port is 7319. Never edit secrets.env. Remember all three facts.",
        },
    ]
    for index in range(8):
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Inspection {index} finished. "
                    + "The routine build log contains no new decisions. " * 70,
                },
                {
                    "role": "user",
                    "content": f"Continue inspection {index + 1}, keeping the original requirements.",
                },
            ]
        )
    question = {
        "role": "user",
        "content": "What is the project name, deployment port, and file you must never edit? Give all three.",
    }
    history.append(question)
    options = {
        "temperature": 0,
        "enable_thinking": False,
        "max_output_tokens": 96,
        "store": False,
    }
    report = {"model": MODEL}
    with running_server(tmp_path) as client:
        original_count, _ = post(
            client, "/v1/responses/input_tokens", input=history, **options
        )
        baseline, report["baseline_seconds"] = post(
            client, "/v1/responses", input=history, **options
        )
        with OpenAI(
            base_url=str(client.base_url).rstrip("/") + "/v1",
            api_key="test",
            default_headers={"X-APC-Tenant": "compaction-test"},
        ) as sdk:
            start = time.monotonic()
            compact_result = sdk.responses.compact(
                model=MODEL,
                input=history,
                extra_body={
                    "keep_tokens": 256,
                    "max_output_tokens": 768,
                    "temperature": 0,
                    "enable_thinking": False,
                },
            )
            report["compaction_seconds"] = round(time.monotonic() - start, 3)
            compacted = compact_result.model_dump()
        assert compacted["output"][0]["type"] == "compaction"
        output = compacted["output"]
        compact_count, _ = post(
            client, "/v1/responses/input_tokens", input=output, **options
        )
        report.update(
            before_tokens=original_count["input_tokens"],
            after_tokens=compact_count["input_tokens"],
            summary_usage=compacted["usage"],
        )
        assert report["after_tokens"] < report["before_tokens"] * 0.6
        cold, report["first_continuation_seconds"] = post(
            client, "/v1/responses", input=output, **options
        )
        warm, report["warm_continuation_seconds"] = post(
            client, "/v1/responses", input=output, **options
        )
        report["first_usage"], report["warm_usage"] = cold["usage"], warm["usage"]
        assert (
            warm["usage"]["input_tokens_details"]["cached_tokens"]
            > cold["usage"]["input_tokens_details"]["cached_tokens"]
        )
        for response in [baseline, cold, warm]:
            text = response["output_text"]
            assert all(fact in text for fact in ("ORCHID", "7319", "secrets.env")), text
        report["answer"] = warm["output_text"]
        with OpenAI(
            base_url=str(client.base_url).rstrip("/") + "/v1",
            api_key="test",
            default_headers={"X-APC-Tenant": "compaction-test"},
        ) as sdk:
            sdk_response = sdk.responses.create(
                model=MODEL,
                input=compact_result.output,
                max_output_tokens=96,
                temperature=0,
                store=False,
                extra_body={"enable_thinking": False},
            )
            assert all(
                fact in sdk_response.output_text
                for fact in ("ORCHID", "7319", "secrets.env")
            )
            report["sdk_replay"] = "passed"
        # Both chaining patterns must resolve to exactly the same tokenized input.
        full_count, _ = post(
            client, "/v1/responses/input_tokens", input=history + output, **options
        )
        assert full_count == compact_count
        client.post("/v1/cache/reset").raise_for_status()
        after_reset, _ = post(client, "/v1/responses", input=output, **options)
        assert after_reset["usage"]["input_tokens_details"]["cached_tokens"] == 0
        assert all(
            fact in after_reset["output_text"]
            for fact in ("ORCHID", "7319", "secrets.env")
        )
        # Automatic compaction emits replayable state in streaming responses.
        start = time.monotonic()
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": history,
                **options,
                "stream": True,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 2000}
                ],
            },
        )
        assert response.is_success, response.text
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        completed = next(
            x["response"] for x in events if x["type"] == "response.completed"
        )
        assert completed["output"][0]["type"] == "compaction"
        assert all(
            fact in completed["output_text"]
            for fact in ("ORCHID", "7319", "secrets.env")
        ), completed["output_text"]
        report["automatic_seconds"] = round(time.monotonic() - start, 3)
        # Repeated compaction must incorporate the preceding summary and corrections.
        more = output + [
            {"role": "assistant", "content": "Confirmed."},
            {
                "role": "user",
                "content": "Correction: deployment port is now 8421. Preserve the other requirements.",
            },
            {
                "role": "assistant",
                "content": "Port updated to 8421. "
                + "Routine verification passed. " * 800,
            },
            question,
        ]
        again, _ = post(
            client,
            "/v1/responses/compact",
            input=more,
            keep_tokens=0,
            max_output_tokens=768,
            enable_thinking=False,
        )
        corrected, _ = post(client, "/v1/responses", input=again["output"], **options)
        assert all(
            fact in corrected["output_text"]
            for fact in ("ORCHID", "8421", "secrets.env")
        ), corrected["output_text"]
        report["corrected_answer"] = corrected["output_text"]
        # Pi/OpenCode/Hermes can keep their own summary format and ownership.
        # Exercise both ordinary protocol paths with a client-authored handoff.
        client_summary = [
            history[0],
            {
                "role": "user",
                "content": "Prior conversation summary: Project ORCHID; deployment port 7319; never edit secrets.env.",
            },
            {"role": "assistant", "content": "I will preserve those requirements."},
            question,
        ]
        for path in ("/v1/chat/completions", "/v1/messages"):
            client.post("/v1/cache/reset").raise_for_status()
            body = {
                "messages": client_summary,
                "max_tokens": 96,
                "temperature": 0,
                "enable_thinking": False,
            }
            if path.endswith("messages"):
                body.update(system=history[0]["content"], messages=client_summary[1:])
            first, _ = post(client, path, **body)
            repeated, _ = post(client, path, **body)
            if path.endswith("messages"):
                text = "".join(part.get("text", "") for part in repeated["content"])
                cache_key = lambda r: r["usage"].get("cache_read_input_tokens", 0)
            else:
                text = repeated["choices"][0]["message"]["content"]
                cache_key = lambda r: r["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ]
            assert all(fact in text for fact in ("ORCHID", "7319", "secrets.env")), text
            assert cache_key(repeated) > cache_key(first)
            report[path] = {"answer": text, "warm_usage": repeated["usage"]}
    with running_server(tmp_path) as client:
        restored, report["restart_continuation_seconds"] = post(
            client, "/v1/responses", input=output, **options
        )
        assert restored["usage"]["input_tokens_details"]["cached_tokens"] == 0
        assert all(
            fact in restored["output_text"]
            for fact in ("ORCHID", "7319", "secrets.env")
        ), restored["output_text"]
    (tmp_path / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Results and server log: {tmp_path}")
