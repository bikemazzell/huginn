from huginn.eval.dataset import EvalCase
from huginn.llm.factory import RuntimeClients
from huginn.preflight import (
    chat_call_ok,
    embedding_call_ok,
    endpoint_model_available,
    sqlite_vec_available,
    uv_available,
)
from huginn.schemas import ModelEndpointConfig


class DummyEmbedder:
    def __init__(self, response):
        self.response = response
        self.calls: list[str] = []

    def embed_text(self, text: str, *, kind: str = "document"):
        self.calls.append(text)
        return self.response


class DummyChat:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def test_uv_available_detects_shell_command(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    assert uv_available() is True


def test_sqlite_vec_available_reports_loaded_extension() -> None:
    assert sqlite_vec_available() is True


def test_endpoint_model_available_checks_model_presence(monkeypatch) -> None:
    monkeypatch.setattr(
        "huginn.preflight.fetch_model_ids",
        lambda config: ["Qwen3.5-9B-Q4_K_M.gguf", "nomic-embed-text-v2-moe.Q4_K_M.gguf"],
    )

    assert (
        endpoint_model_available(
            ModelEndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="ollama",
                model="Qwen3.5-9B-Q4_K_M.gguf",
            )
        )
        is True
    )


def test_endpoint_model_available_matches_embedding_model_by_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "huginn.preflight.fetch_model_ids",
        lambda config: ["nomic-embed-text-v2-moe.Q4_K_M.gguf"],
    )

    assert (
        endpoint_model_available(
            ModelEndpointConfig(
                base_url="http://127.0.0.1:1235/v1",
                api_key="ollama",
                model="nomic-embed-text-v2-moe",
            )
        )
        is True
    )


def test_endpoint_model_available_matches_chat_model_without_exact_suffix(monkeypatch) -> None:
    monkeypatch.setattr(
        "huginn.preflight.fetch_model_ids",
        lambda config: ["Qwen3.5-9B-Q4_K_M.gguf"],
    )

    assert (
        endpoint_model_available(
            ModelEndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="ollama",
                model="Qwen3.5-9B",
            )
        )
        is True
    )


def test_embedding_call_ok_exercises_embedder() -> None:
    embedder = DummyEmbedder([0.1, 0.2, 0.3])

    assert embedding_call_ok(embedder) is True
    assert embedder.calls


def test_chat_call_ok_exercises_chat_model() -> None:
    chat = DummyChat("pong")

    assert chat_call_ok(chat) is True
    assert chat.calls
