import json
import subprocess
import shutil
import sqlite3
import urllib.request

import sqlite_vec

from huginn.schemas import ModelEndpointConfig


def uv_available() -> bool:
    return shutil.which("uv") is not None


def sqlite_vec_available() -> bool:
    try:
        connection = sqlite3.connect(":memory:")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return True
    finally:
        connection.close()


def fetch_model_ids(config: ModelEndpointConfig) -> list[str]:
    url = config.base_url.rstrip("/") + "/models"
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        completed = subprocess.run(
            ["curl", "-sf", url],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise
        payload = json.loads(completed.stdout)
    model_ids: list[str] = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            model_ids.append(model_id)
    for item in payload.get("models", []):
        model_name = item.get("model") or item.get("name")
        if isinstance(model_name, str):
            model_ids.append(model_name)
    return model_ids


def endpoint_model_available(config: ModelEndpointConfig) -> bool:
    try:
        requested = _normalize_model_name(config.model)
        for model_id in fetch_model_ids(config):
            candidate = _normalize_model_name(model_id)
            if requested == candidate:
                return True
            if candidate.startswith(requested) or requested.startswith(candidate):
                return True
        return False
    except Exception:
        return False


def embedding_call_ok(embedder: object) -> bool:
    try:
        result = embedder.embed_text("preflight ping", kind="query")
    except Exception:
        return False
    return bool(result)


def chat_call_ok(chat_model: object) -> bool:
    try:
        response = chat_model.complete(
            system_prompt="Reply with the single word pong.",
            user_prompt="ping",
        )
    except Exception:
        return False
    return isinstance(response, str) and len(response.strip()) > 0


def pdf_dependencies_ok() -> bool:
    try:
        import pypdf  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_support_ok(ocr_fallback_enabled: bool) -> bool:
    if not ocr_fallback_enabled:
        return True
    # Current OCR fallback mode reads checked-in sidecar text files and requires no external binary.
    return True


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.endswith(".gguf"):
        normalized = normalized[:-5]
    return normalized
