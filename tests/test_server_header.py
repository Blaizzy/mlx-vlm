"""Regression test: mlx_vlm.server must emit exactly one Server response header.

Two Server headers (one from uvicorn's default + one from mlx-vlm's middleware)
violate RFC 7231 §7.4.2 singleton-field semantics and break strict HTTP clients
like aiohttp (used by LiteLLM and other production proxies).

This test starts the server briefly and asserts the response has exactly one
Server header, with mlx-vlm's informative version identifier.
"""
import subprocess
import sys
import time

import httpx
import pytest


@pytest.fixture(scope="module")
def running_server():
    """Start mlx_vlm.server on a high port; tear down at end of module."""
    port = 18750
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to come up
    for _ in range(30):
        try:
            httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(1)
    else:
        proc.terminate()
        pytest.skip("Server failed to start within 30s")
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


def test_single_server_header(running_server):
    """Response from /v1/models has exactly one Server header."""
    response = httpx.get(f"{running_server}/v1/models")
    server_headers = [
        v for k, v in response.headers.raw if k.lower() == b"server"
    ]
    assert len(server_headers) == 1, (
        f"Expected one Server header, got {len(server_headers)}: "
        f"{server_headers}"
    )


def test_server_header_includes_mlx_vlm_version(running_server):
    """The single Server header includes the mlx_vlm version identifier."""
    response = httpx.get(f"{running_server}/v1/models")
    server_value = response.headers.get("server", "")
    assert "mlx_vlm" in server_value.lower(), (
        f"Expected 'mlx_vlm' in Server header, got: {server_value!r}"
    )
