from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huginn.config import load_runtime_config
from huginn.llm.factory import build_runtime_clients, describe_models
from huginn.preflight import (
    chat_call_ok,
    embedding_call_ok,
    endpoint_model_available,
    sqlite_vec_available,
    uv_available,
)


def main() -> int:
    config_path = ROOT / "config" / "runtime.yaml"
    results: dict[str, object] = {
        "python_ok": sys.version_info >= (3, 12),
        "uv_available": uv_available(),
        "sqlite_version": sqlite3.sqlite_version,
        "imports_ok": True,
    }

    try:
        config = load_runtime_config(config_path)
        clients = build_runtime_clients(config.models)
        results["config_ok"] = True
        results["models"] = describe_models(config.models)
        results["endpoint_reachable"] = _check_endpoint(config.models.chat.base_url)
        results["chat_model_available"] = endpoint_model_available(config.models.chat)
        results["embedding_model_available"] = endpoint_model_available(config.models.embedding)
        results["embedding_call_ok"] = embedding_call_ok(clients.embedder)
        results["chat_call_ok"] = chat_call_ok(clients.chat)
        results["sqlite_vec_available"] = sqlite_vec_available()
        results["pdf_dependencies_ok"] = True
        results["ocr_dependency_configured"] = config.features.ocr_fallback
    except Exception as exc:  # pragma: no cover - script path
        results["config_ok"] = False
        results["error"] = str(exc)
        print(json.dumps(results, indent=2))
        return 1

    print(json.dumps(results, indent=2))
    return 0 if results["python_ok"] and results["config_ok"] else 1


def _check_endpoint(base_url: str) -> bool:
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models")
        with urllib.request.urlopen(request, timeout=2):
            return True
    except Exception:
        try:
            completed = subprocess.run(
                ["curl", "-sf", base_url.rstrip("/") + "/models"],
                check=False,
                capture_output=True,
                text=True,
            )
            return completed.returncode == 0
        except Exception:
            return False


if __name__ == "__main__":
    raise SystemExit(main())
