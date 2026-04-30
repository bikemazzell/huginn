from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.query_graph import run_query
from huginn.schemas import RetrievedChunk


def build_runtime_config(tmp_path: Path, *, answer_validation: bool) -> object:
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
                "  query_rewrite: false",
                "  rerank: false",
                f"  answer_validation: {'true' if answer_validation else 'false'}",
            ]
        ),
        encoding="utf-8",
    )
    return load_runtime_config(config_path)


class FakeValidationChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "Validate whether the answer" in system_prompt:
            return "UNSUPPORTED"
        return "Project Atlas budget is 5000 dollars."


class FakeEmbedder:
    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        return [1.0, 0.0]


def test_run_query_applies_answer_validation_when_enabled(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, answer_validation=True)

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, question, top_k, embedder, min_lexical_score, max_dense_distance
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
        question="What is the budget?",
        embedder=FakeEmbedder(),
        chat_model=FakeValidationChatModel(),
    )

    assert answer.answer_text == "I could not find grounded evidence for that question."
    assert answer.citations == []
    assert answer.evidence_note == "Answer validation rejected the generated answer."


def test_run_query_skips_answer_validation_when_disabled(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, answer_validation=False)
    chat = FakeValidationChatModel()

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, question, top_k, embedder, min_lexical_score, max_dense_distance
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
        question="What is the budget?",
        embedder=FakeEmbedder(),
        chat_model=chat,
    )

    assert answer.answer_text == "Project Atlas budget is 5000 dollars."
    assert len(chat.calls) == 1
