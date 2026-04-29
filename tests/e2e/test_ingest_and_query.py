from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.graph.query_graph import run_query
from tests.helpers import write_pdf


def build_runtime_config(tmp_path: Path, corpus: Path):
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
                "  chunk_size: 8",
                "  chunk_overlap: 2",
                "  top_k: 4",
                "features:",
                "  ocr_fallback: true",
                "  query_rewrite: false",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )
    return load_runtime_config(config_path)


def test_e2e_ingest_supports_text_ocr_and_multi_page_pdfs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_pdf(corpus / "atlas.pdf", ["Project Atlas budget is 1200 dollars."])
    write_pdf(corpus / "scan.pdf", [""])
    (corpus / "scan.ocr.txt").write_text("Scanned contract mentions Orion vendor.", encoding="utf-8")
    write_pdf(
        corpus / "minutes.pdf",
        ["Meeting intro and background.", "The launch date is 2026-05-15."],
    )
    config = build_runtime_config(tmp_path, corpus)

    ingest_result = run_ingest(config, db_path=tmp_path / "huginn.db")
    atlas_answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="Which vendor is mentioned in the scanned contract?",
    )
    launch_answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the launch date?",
    )

    assert ingest_result.indexed_count == 3
    assert ingest_result.failed_count == 0
    assert "Orion vendor" in atlas_answer.answer_text
    assert atlas_answer.citations == ["scan.pdf#page=1"]
    assert "2026-05-15" in launch_answer.answer_text
    assert launch_answer.citations == ["minutes.pdf#page=2"]


def test_e2e_negative_query_returns_safe_no_answer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_pdf(corpus / "atlas.pdf", ["Project Atlas budget is 1200 dollars."])
    config = build_runtime_config(tmp_path, corpus)

    run_ingest(config, db_path=tmp_path / "huginn.db")
    answer = run_query(
        config,
        db_path=tmp_path / "huginn.db",
        question="What is the employee vacation policy?",
    )

    assert answer.answer_text == "I could not find grounded evidence for that question."
    assert answer.citations == []
    assert answer.evidence_note == "No sufficiently relevant chunks were retrieved."
