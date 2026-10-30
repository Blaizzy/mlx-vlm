"""Stage disjoint, byte-preserving Mistral expert shards over a private link."""

import argparse
import concurrent.futures
import json
import re
import shutil
import struct
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def serve(source, bind, port):
    files = {}
    for path in sorted(source.glob("*.safetensors")):
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            original = json.loads(f.read(length))
        files[path.name] = (path, length + 8, original)
    params = json.loads((source / "params.json").read_text())
    experts = params["moe"]["num_experts"]
    prepared = {}

    def prepare(rank, size, name):
        key = rank, size, name
        if key in prepared:
            return prepared[key]
        path, offset, original = files[name]
        start, end = experts * rank // size, experts * (rank + 1) // size
        selected, regions, nbytes = {}, [], 0
        for key, item in sorted(
            ((k, v) for k, v in original.items() if k != "__metadata__"),
            key=lambda kv: kv[1]["data_offsets"][0],
        ):
            expert = re.match(r"layers\.\d+\.experts\.(\d+)\.", key)
            if key.startswith("mtp") or (expert and not start <= int(expert[1]) < end):
                continue
            a, b = item["data_offsets"]
            info = dict(item, data_offsets=[nbytes, nbytes + b - a])
            if info["dtype"].startswith("F8_"):
                info["dtype"] = "U8"
            selected[key] = info
            if regions and regions[-1][0] + regions[-1][1] == offset + a:
                regions[-1] = (regions[-1][0], regions[-1][1] + b - a)
            else:
                regions.append((offset + a, b - a))
            nbytes += b - a
        header = json.dumps(selected, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        result = (
            path,
            struct.pack("<Q", len(header)) + header,
            regions,
            nbytes + len(header) + 8,
        )
        prepared[rank, size, name] = result
        return result

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            match = re.fullmatch(
                r"/(manifest|file)/(\d+)/(\d+)(?:/([^/]+))?", self.path
            )
            if not match:
                self.send_error(404)
                return
            kind, rank, size, name = match.groups()
            rank = int(rank)
            size = int(size)
            if not 0 <= rank < size <= experts:
                self.send_error(400)
                return
            if kind == "manifest":
                data = json.dumps(
                    {
                        "rank": rank,
                        "size": size,
                        "expert_start": experts * rank // size,
                        "expert_end": experts * (rank + 1) // size,
                        "source": str(source),
                        "files": {n: prepare(rank, size, n)[3] for n in files},
                        "metadata": {
                            n: (source / n).read_text()
                            for n in (
                                "params.json",
                                "tokenizer.json",
                                "tokenizer_config.json",
                                "generation_config.json",
                                "chat_template.jinja",
                                "processor_config.json",
                                "tekken.json",
                            )
                            if (source / n).exists()
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if name not in files:
                self.send_error(404)
                return
            path, header, regions, length = prepare(rank, size, name)
            self.send_response(200)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(header)
            with path.open("rb") as f:
                for offset, count in regions:
                    f.seek(offset)
                    while count:
                        chunk = f.read(min(8 * 1024**2, count))
                        if not chunk:
                            raise EOFError(path)
                        self.wfile.write(chunk)
                        count -= len(chunk)

    print(json.dumps({"serving": str(source), "bind": bind, "port": port}), flush=True)
    ThreadingHTTPServer((bind, port), Handler).serve_forever()


def stage(url, rank, size, output):
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{url}/manifest/{rank}/{size}", timeout=120) as r:
        manifest = json.load(r)
    if output.resolve() == Path(manifest["source"]).resolve():
        raise ValueError("Stage into a separate directory from the original checkpoint")
    marker = output / "rank.json"
    if marker.exists() and json.loads(marker.read_text()) != {
        k: v for k, v in manifest.items() if k != "metadata"
    }:
        raise ValueError("Output already contains a different checkpoint partition")
    required = sum(
        n
        for name, n in manifest["files"].items()
        if not (output / name).exists() or (output / name).stat().st_size != n
    )
    if shutil.disk_usage(output).free < required + 10 * 1024**3:
        raise RuntimeError("Insufficient disk space")
    for name, content in manifest.pop("metadata").items():
        (output / name).write_text(content)
    marker.write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()

    def download(item):
        name, length = item
        target = output / name
        if target.exists() and target.stat().st_size == length:
            return name, "cached"
        partial = target.with_suffix(".partial")
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(
                        f"{url}/file/{rank}/{size}/{name}", timeout=180
                    ) as r,
                    partial.open("wb") as f,
                ):
                    shutil.copyfileobj(r, f, 8 * 1024**2)
                if partial.stat().st_size != length:
                    raise IOError("Incomplete file")
                partial.replace(target)
                print(
                    json.dumps(
                        {
                            "file": name,
                            "bytes": length,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
                return name, length
            except Exception:
                if attempt == 2:
                    raise

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        list(executor.map(download, manifest["files"].items()))
    print(
        json.dumps(
            {
                "complete": True,
                "rank": rank,
                "bytes": sum(manifest["files"].values()),
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("source", type=Path)
    s.add_argument("--bind", required=True)
    s.add_argument("--port", type=int, default=8944)
    s = sub.add_parser("stage")
    s.add_argument("url")
    s.add_argument("rank", type=int)
    s.add_argument("size", type=int)
    s.add_argument("output", type=Path)
    a = vars(p.parse_args())
    command = a.pop("command")
    globals()[command](**a)
