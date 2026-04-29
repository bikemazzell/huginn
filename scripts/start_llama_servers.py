#!/usr/bin/env python3
"""Start the two local llama.cpp servers Huginn expects (chat + embeddings).

Defaults match config/runtime.yaml: chat on 127.0.0.1:1234, embeddings on 1235.
Blocks until both endpoints respond on /v1/models, then keeps running until
SIGINT/SIGTERM, at which point both child servers are terminated.

Usage:
    scripts/start_llama_servers.py --chat-model /path/to/qwen.gguf --chat-mmproj /path/to/mmproj.gguf
    HUGINN_CHAT_MODEL=/path/to/qwen.gguf HUGINN_CHAT_MMPROJ=/path/to/mmproj.gguf scripts/start_llama_servers.py

Override the binary with $LLAMA_SERVER_BIN (default: llama-server on PATH).
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBED_MODEL = (
    REPO_ROOT / "models" / "nomic-embed-text-v2-moe" / "nomic-embed-text-v2-moe.Q4_K_M.gguf"
)
CHAT_PORT = 1234
EMBED_PORT = 1235
HEALTH_TIMEOUT_SEC = 120
HEALTH_POLL_SEC = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat-model",
        default=os.environ.get("HUGINN_CHAT_MODEL"),
        help="Path to chat GGUF (env: HUGINN_CHAT_MODEL).",
    )
    parser.add_argument(
        "--chat-mmproj",
        default=os.environ.get("HUGINN_CHAT_MMPROJ"),
        help="Path to chat mmproj GGUF (env: HUGINN_CHAT_MMPROJ).",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("HUGINN_EMBED_MODEL", str(DEFAULT_EMBED_MODEL)),
        help="Path to embedding GGUF (env: HUGINN_EMBED_MODEL).",
    )
    parser.add_argument("--chat-port", type=int, default=CHAT_PORT)
    parser.add_argument("--embed-port", type=int, default=EMBED_PORT)
    parser.add_argument(
        "--chat-ctx", type=int, default=8192, help="Chat context length (-c)."
    )
    parser.add_argument(
        "--embed-ctx", type=int, default=512, help="Embed context length (-c)."
    )
    parser.add_argument(
        "--ngl", type=int, default=99, help="GPU layers offload (-ngl)."
    )
    return parser.parse_args()


def resolve_binary() -> str:
    binary = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    found = shutil.which(binary)
    if not found:
        sys.exit(
            f"error: {binary!r} not found on PATH. Set $LLAMA_SERVER_BIN or install llama.cpp."
        )
    return found


def validate_model(label: str, path: str | None) -> Path:
    if not path:
        sys.exit(f"error: --{label}-model is required (no default available).")
    p = Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"error: {label} model not found at {p}")
    return p


def spawn(
    binary: str,
    model: Path,
    *,
    mmproj: Path | None,
    port: int,
    ngl: int,
    ctx: int,
    embeddings: bool,
) -> subprocess.Popen[bytes]:
    cmd = [
        binary,
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        str(ngl),
        "-c",
        str(ctx),
    ]
    if embeddings:
        cmd += ["--embeddings", "--pooling", "cls"]
    else:
        if mmproj is not None:
            cmd += ["--mmproj", str(mmproj)]
        cmd += ["--jinja", "--reasoning", "off", "--reasoning-budget", "0"]
    print(f"[start_llama_servers] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_healthy(port: int, label: str, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[start_llama_servers] {label} ready on :{port}", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(HEALTH_POLL_SEC)
    raise TimeoutError(f"{label} on :{port} did not become healthy within timeout")


def terminate(procs: list[subprocess.Popen[bytes]]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + 10
    for p in procs:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main() -> int:
    args = parse_args()
    binary = resolve_binary()
    chat_model = validate_model("chat", args.chat_model)
    chat_mmproj = validate_model("chat mmproj", args.chat_mmproj)
    embed_model = validate_model("embed", args.embed_model)

    procs: list[subprocess.Popen[bytes]] = []
    interrupted = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"[start_llama_servers] caught signal {signum}, shutting down", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        procs.append(
            spawn(
                binary,
                chat_model,
                mmproj=chat_mmproj,
                port=args.chat_port,
                ngl=args.ngl,
                ctx=args.chat_ctx,
                embeddings=False,
            )
        )
        procs.append(
            spawn(
                binary,
                embed_model,
                mmproj=None,
                port=args.embed_port,
                ngl=args.ngl,
                ctx=args.embed_ctx,
                embeddings=True,
            )
        )

        deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
        wait_healthy(args.chat_port, "chat", deadline)
        wait_healthy(args.embed_port, "embed", deadline)
        print("[start_llama_servers] both servers ready. Ctrl-C to stop.", flush=True)

        while not interrupted:
            for p in procs:
                if p.poll() is not None:
                    print(
                        f"[start_llama_servers] child pid={p.pid} exited with {p.returncode}",
                        flush=True,
                    )
                    return p.returncode or 1
            time.sleep(0.5)
        return 0
    finally:
        terminate(procs)


if __name__ == "__main__":
    sys.exit(main())
