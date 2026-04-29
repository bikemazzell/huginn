from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.graph.query_graph import run_query
from tests.helpers import write_pdf


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        self.calls.append(text)
        return [float(len(text.split())), 1.0]

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]]:
        return [self.embed_text(text, kind=kind) for text in texts]


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "Model says the launch date is 2026-05-15."


def test_runtime_clients_are_used_by_ingest_and_query(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_pdf(corpus / "minutes.pdf", ["The launch date is 2026-05-15."])
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"root_path: {corpus}",
                "local_only: true",
                "models:",
                "  chat:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: qwen3:8b",
                "  embedding:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: bge-m3",
                "indexing:",
                "  chunk_size: 50",
                "  chunk_overlap: 10",
                "  top_k: 3",
                "features:",
                "  ocr_fallback: true",
                "  query_rewrite: false",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    embedder = FakeEmbedder()
    chat = FakeChatModel()

    run_ingest(
        config,
        db_path=tmp_path / "huginn.db",
        embedder=embedder,
    )
    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the launch date?",
        embedder=embedder,
        chat_model=chat,
    )

    assert embedder.calls
    assert chat.calls
    assert answer.answer_text == "Model says the launch date is 2026-05-15."
