from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.graph.query_graph import run_query
from tests.helpers import write_pdf


def test_smoke_ingest_then_ask_returns_grounded_answer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_pdf(corpus / "atlas.pdf", ["Project Atlas budget is 1200 dollars."])

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

    ingest_result = run_ingest(config, db_path=tmp_path / "huginn.db")
    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the Project Atlas budget?",
    )

    assert ingest_result.indexed_count == 1
    assert "1200 dollars" in answer.answer_text
    assert answer.citations == ["atlas.pdf#page=1"]
