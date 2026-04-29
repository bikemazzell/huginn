from pathlib import Path

from huginn.answer.generate import generate_answer
from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.graph.query_graph import run_query
from huginn.llm.factory import NomicPrefixEmbedder
from huginn.schemas import RetrievedChunk
from tests.helpers import write_pdf


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str, *, kind: str = "document") -> dict[str, float]:
        self.calls.append(text)
        words = text.lower().split()
        return {word: float(words.count(word)) for word in set(words)}

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[dict[str, float]]:
        return [self.embed_text(text, kind=kind) for text in texts]


class PassthroughChatModel:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "The launch date is 2026-05-15."


def test_ingest_prefixes_documents_for_nomic_embedder(tmp_path: Path) -> None:
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
                "    base_url: http://127.0.0.1:1234/v1",
                "    api_key: ollama",
                "    model: Qwen3.5-9B",
                "  embedding:",
                "    base_url: http://127.0.0.1:1235/v1",
                "    api_key: ollama",
                "    model: nomic-embed-text-v2-moe",
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
    inner = RecordingEmbedder()
    embedder = NomicPrefixEmbedder(inner)

    run_ingest(config, db_path=tmp_path / "huginn.db", embedder=embedder)

    assert inner.calls
    assert inner.calls[0].startswith("search_document: ")


def test_query_prefixes_question_for_nomic_embedder(tmp_path: Path) -> None:
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
                "    base_url: http://127.0.0.1:1234/v1",
                "    api_key: ollama",
                "    model: Qwen3.5-9B",
                "  embedding:",
                "    base_url: http://127.0.0.1:1235/v1",
                "    api_key: ollama",
                "    model: nomic-embed-text-v2-moe",
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
    inner = RecordingEmbedder()
    embedder = NomicPrefixEmbedder(inner)

    run_ingest(config, db_path=tmp_path / "huginn.db")
    run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the launch date?",
        embedder=embedder,
        chat_model=PassthroughChatModel(),
    )

    assert inner.calls == ["search_query: What is the launch date?"]


def test_generate_answer_citations_are_unchanged_by_nomic_work() -> None:
    answer = generate_answer(
        "What is the launch date?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/minutes.pdf",
                page_start=2,
                page_end=2,
                text="The launch date is 2026-05-15.",
                score=0.9,
            )
        ],
        chat_model=PassthroughChatModel(),
    )

    assert answer.citations == ["minutes.pdf#page=2"]
