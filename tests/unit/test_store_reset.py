from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.store.sqlite import SQLiteStore
from tests.helpers import write_pdf


class FakeEmbedder:
    def embed_text(self, text: str, *, kind: str = "document") -> dict[str, float]:
        words = text.lower().split()
        return {word: float(words.count(word)) for word in set(words)}

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[dict[str, float]]:
        return [self.embed_text(text, kind=kind) for text in texts]


def test_reindex_clears_existing_documents_before_ingest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_pdf(corpus / "one.pdf", ["Alpha text"])
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
    db_path = tmp_path / "huginn.db"

    run_ingest(config, db_path=db_path, embedder=FakeEmbedder())
    write_pdf(corpus / "two.pdf", ["Beta text"])
    run_ingest(config, db_path=db_path, embedder=FakeEmbedder(), reindex=True)

    store = SQLiteStore(db_path)
    try:
        counts = store.status_counts()
    finally:
        store.close()

    assert counts["source_file_count"] == 2
    assert counts["document_count"] == 2
