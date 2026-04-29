from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_start_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "start_llama_servers.py"
    spec = spec_from_file_location("start_llama_servers", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyPopen:
    def __init__(self, cmd, **kwargs) -> None:
        self.cmd = cmd
        self.kwargs = kwargs


def test_spawn_chat_server_uses_working_qwen_flags(monkeypatch) -> None:
    module = _load_start_script()
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return DummyPopen(cmd, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    module.spawn(
        "/tmp/llama-server",
        Path("/tmp/chat.gguf"),
        mmproj=Path("/tmp/mmproj.gguf"),
        port=1234,
        ngl=99,
        ctx=8192,
        embeddings=False,
    )

    assert captured["cmd"] == [
        "/tmp/llama-server",
        "-m",
        "/tmp/chat.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "1234",
        "-ngl",
        "99",
        "-c",
        "8192",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]


def test_spawn_embedding_server_uses_embeddings_mode(monkeypatch) -> None:
    module = _load_start_script()
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return DummyPopen(cmd, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    module.spawn(
        "/tmp/llama-server",
        Path("/tmp/embed.gguf"),
        mmproj=None,
        port=1235,
        ngl=99,
        ctx=512,
        embeddings=True,
    )

    assert captured["cmd"] == [
        "/tmp/llama-server",
        "-m",
        "/tmp/embed.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "1235",
        "-ngl",
        "99",
        "-c",
        "512",
        "--embeddings",
        "--pooling",
        "cls",
    ]
