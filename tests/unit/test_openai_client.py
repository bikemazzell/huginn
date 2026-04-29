import json
from urllib.error import HTTPError

import pytest

from huginn.llm.factory import build_runtime_clients
from huginn.retrieve.basic import lexical_features
from huginn.schemas import ModelEndpointConfig, ModelsConfig


class DummyResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_build_runtime_clients_embedding_calls_openai_compatible_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"data": [{"embedding": [0.25, 0.75]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    embedding = clients.embedder.embed_text("hello world")

    assert embedding == [0.25, 0.75]
    assert captured["url"] == "http://127.0.0.1:8080/v1/embeddings"
    assert captured["auth"] == "Bearer token"
    assert captured["body"] == {"model": "bge-small", "input": "hello world"}


def test_build_runtime_clients_chat_calls_openai_compatible_endpoint(monkeypatch) -> None:
    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "qwen3.6"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert body["reasoning"] == "off"
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Grounded answer from the local model."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    answer = clients.chat.complete(
        system_prompt="Answer from supplied context only.",
        user_prompt="Question: what is the budget?",
    )

    assert answer == "Grounded answer from the local model."


def test_build_runtime_clients_raises_useful_error_on_http_failure(monkeypatch) -> None:
    def fake_urlopen(request, timeout=0):
        raise HTTPError(
            url=request.full_url,
            code=500,
            msg="boom",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    with pytest.raises(RuntimeError, match="embeddings"):
        clients.embedder.embed_text("hello world")


def test_openai_embedder_batch_uses_single_request(monkeypatch) -> None:
    captured: dict[str, object] = {}
    call_count = {"n": 0}

    def fake_urlopen(request, timeout=0):
        call_count["n"] += 1
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse(
            {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                    {"embedding": [0.5, 0.6]},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    vectors = clients.embedder.embed_texts(["alpha", "beta", "gamma"])

    assert call_count["n"] == 1
    assert captured["body"] == {"model": "bge-small", "input": ["alpha", "beta", "gamma"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


def test_openai_embedder_large_batch_splits_requests(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        calls.append(body)
        inputs = body["input"]
        assert isinstance(inputs, list)
        return DummyResponse(
            {
                "data": [{"embedding": [float(index)]} for index, _ in enumerate(inputs)]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    texts = [f"text-{index}" for index in range(300)]
    vectors = clients.embedder.embed_texts(texts)

    assert len(calls) == 2
    assert len(calls[0]["input"]) == 256
    assert len(calls[1]["input"]) == 44
    assert len(vectors) == 300


def test_openai_embedder_retries_failed_batch_by_splitting(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        inputs = body["input"]
        assert isinstance(inputs, list)
        calls.append(list(inputs))
        if len(inputs) > 1 and inputs[0] == "text-256":
            raise HTTPError(
                url=request.full_url,
                code=500,
                msg="boom",
                hdrs=None,
                fp=None,
            )
        return DummyResponse(
            {
                "data": [{"embedding": [float(index)]} for index, _ in enumerate(inputs)]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    texts = [f"text-{index}" for index in range(300)]
    vectors = clients.embedder.embed_texts(texts)

    assert len(vectors) == 300
    assert any(batch[:1] == ["text-256"] and len(batch) == 44 for batch in calls)
    assert any(batch[:1] == ["text-256"] and len(batch) < 44 for batch in calls)


def test_openai_embedder_retries_single_oversized_input_with_shorter_prefix(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        text = body["input"]
        assert isinstance(text, str)
        calls.append(text)
        if len(text.split()) > 4:
            raise HTTPError(
                url=request.full_url,
                code=500,
                msg="boom",
                hdrs=None,
                fp=None,
            )
        return DummyResponse({"data": [{"embedding": [1.0, 2.0]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="bge-small",
            ),
        )
    )

    vector = clients.embedder.embed_text("one two three four five six seven eight")

    assert vector == [1.0, 2.0]
    assert len(calls) >= 2
    assert calls[-1].split() == ["one", "two", "three", "four"]


def test_nomic_embedder_batch_prefixes_each_text(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse(
            {"data": [{"embedding": [0.1]}, {"embedding": [0.2]}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="qwen3.6",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
                model="nomic-embed-text-v2-moe",
            ),
        )
    )

    clients.embedder.embed_texts(["alpha", "beta"], kind="query")

    assert captured["body"]["input"] == ["search_query: alpha", "search_query: beta"]


def test_build_runtime_clients_can_use_local_lexical_embedder() -> None:
    clients = build_runtime_clients(
        ModelsConfig(
            chat=ModelEndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="token",
                model="Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
            ),
            embedding=ModelEndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="token",
                model="local-lexical",
            ),
        )
    )

    assert clients.embedder.embed_text("hello budget world") == lexical_features(
        "hello budget world"
    )
