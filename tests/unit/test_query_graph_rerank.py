from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.query_graph import run_query
from huginn.schemas import RetrievedChunk


def build_runtime_config(tmp_path: Path, *, rerank: bool) -> object:
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
                f"  rerank: {'true' if rerank else 'false'}",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )
    return load_runtime_config(config_path)


class FakeDenseEmbedder:
    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        return [1.0, 0.0]


def test_run_query_reranks_wider_dense_candidates(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, rerank=True)
    seen_limits: list[int] = []
    rerank_inputs: list[list[int]] = []

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, question, embedder, min_lexical_score, max_dense_distance
        seen_limits.append(top_k)
        return [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Atlas appendix notes.",
                score=-0.1,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/atlas.pdf",
                page_start=2,
                page_end=2,
                text="Atlas budget is 1200 dollars.",
                score=-0.3,
            ),
            RetrievedChunk(
                chunk_id=3,
                source_path="/tmp/atlas.pdf",
                page_start=3,
                page_end=3,
                text="Atlas planning notes.",
                score=-0.2,
            ),
        ]

    def fake_rerank_chunks(question, chunks, *, limit):
        del question
        rerank_inputs.append([chunk.chunk_id for chunk in chunks])
        assert limit == 2
        return [chunks[1], chunks[0]]

    monkeypatch.setattr("huginn.graph.query_graph.retrieve_top_chunks", fake_retrieve_top_chunks)
    monkeypatch.setattr("huginn.graph.query_graph.rerank_chunks", fake_rerank_chunks)

    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the Atlas budget?",
        embedder=FakeDenseEmbedder(),
        chat_model=None,
    )

    assert seen_limits == [6]
    assert rerank_inputs == [[1, 2, 3]]
    assert answer.answer_text == "Atlas budget is 1200 dollars."
    assert answer.citations == ["atlas.pdf#page=2"]


def test_run_query_skips_rerank_when_feature_disabled(monkeypatch, tmp_path: Path) -> None:
    config = build_runtime_config(tmp_path, rerank=False)
    seen_limits: list[int] = []

    def fake_retrieve_top_chunks(*, store, question, top_k, embedder, min_lexical_score, max_dense_distance):
        del store, question, embedder, min_lexical_score, max_dense_distance
        seen_limits.append(top_k)
        return [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Atlas appendix notes.",
                score=-0.1,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/atlas.pdf",
                page_start=2,
                page_end=2,
                text="Atlas budget is 1200 dollars.",
                score=-0.3,
            ),
        ]

    def fail_rerank_chunks(question, chunks, *, limit):
        raise AssertionError("rerank should not be called")

    monkeypatch.setattr("huginn.graph.query_graph.retrieve_top_chunks", fake_retrieve_top_chunks)
    monkeypatch.setattr("huginn.graph.query_graph.rerank_chunks", fail_rerank_chunks)

    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the Atlas budget?",
        embedder=FakeDenseEmbedder(),
        chat_model=None,
    )

    assert seen_limits == [2]
    assert answer.answer_text == "Atlas appendix notes."
    assert answer.citations == ["atlas.pdf#page=1"]
