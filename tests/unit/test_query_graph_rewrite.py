from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.query_graph import run_query
from huginn.schemas import RetrievedChunk


def build_runtime_config(tmp_path: Path, *, query_rewrite: bool) -> object:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"root_path: {tmp_path / 'docs'}",
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
                "  chunk_size: 8",
                "  chunk_overlap: 2",
                "  top_k: 2",
                "  min_lexical_score: 0.2",
                "  max_dense_distance: 0.7",
                "features:",
                "  ocr_fallback: true",
                f"  query_rewrite: {'true' if query_rewrite else 'false'}",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )
    return load_runtime_config(config_path)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


class FakeRewriteChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "Rewrite the user question" in system_prompt:
            return "atlas budget"
        return "The budget is 1200 dollars."


def test_run_query_uses_rewritten_question_for_retrieval_only(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, query_rewrite=True)
    chat = FakeRewriteChatModel()
    embedder = FakeEmbedder()
    retrieved_questions: list[str] = []

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, top_k, min_lexical_score, max_dense_distance
        retrieved_questions.append(question)
        embedder.embed_text(question, kind="query")
        return [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Project Atlas budget is 1200 dollars.",
                score=0.9,
            )
        ]

    monkeypatch.setattr("huginn.graph.query_graph.retrieve_top_chunks", fake_retrieve_top_chunks)

    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="what's the spend for atlas?",
        embedder=embedder,
        chat_model=chat,
    )

    assert retrieved_questions == ["atlas budget"]
    assert embedder.calls == ["atlas budget"]
    assert "Question: what's the spend for atlas?" in chat.calls[-1][1]
    assert answer.answer_text == "The budget is 1200 dollars."


def test_run_query_skips_rewrite_when_feature_disabled(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, query_rewrite=False)
    chat = FakeRewriteChatModel()
    embedder = FakeEmbedder()
    retrieved_questions: list[str] = []

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, top_k, min_lexical_score, max_dense_distance
        retrieved_questions.append(question)
        embedder.embed_text(question, kind="query")
        return [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Project Atlas budget is 1200 dollars.",
                score=0.9,
            )
        ]

    monkeypatch.setattr("huginn.graph.query_graph.retrieve_top_chunks", fake_retrieve_top_chunks)

    run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="what's the spend for atlas?",
        embedder=embedder,
        chat_model=chat,
    )

    assert retrieved_questions == ["what's the spend for atlas?"]
    assert embedder.calls == ["what's the spend for atlas?"]
    assert len(chat.calls) == 1


def test_run_query_skips_rewrite_without_chat_model(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, query_rewrite=True)
    embedder = FakeEmbedder()
    retrieved_questions: list[str] = []

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, top_k, min_lexical_score, max_dense_distance
        retrieved_questions.append(question)
        embedder.embed_text(question, kind="query")
        return [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Project Atlas budget is 1200 dollars.",
                score=0.9,
            )
        ]

    monkeypatch.setattr("huginn.graph.query_graph.retrieve_top_chunks", fake_retrieve_top_chunks)

    run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="what's the spend for atlas?",
        embedder=embedder,
        chat_model=None,
    )

    assert retrieved_questions == ["what's the spend for atlas?"]
    assert embedder.calls == ["what's the spend for atlas?"]
