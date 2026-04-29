from pathlib import Path

from huginn.schemas import ChunkRecord, ExtractedDocument, ExtractedPage
from huginn.store.sqlite import SQLiteStore


def test_sqlite_store_creates_sqlite_vec_virtual_table_after_dense_insert(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "huginn.db")
    try:
        store.replace_document(
            source_path=tmp_path / "alpha.pdf",
            sha256="a" * 64,
            modified_at="1",
            extracted_document=ExtractedDocument(
                source_path=str(tmp_path / "alpha.pdf"),
                title="alpha",
                pages=[ExtractedPage(page_number=1, text="Alpha budget")],
            ),
            chunks=[
                ChunkRecord(
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                    text="Alpha budget",
                    token_count=2,
                )
            ],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        row = store.connection.execute(
            "select sql from sqlite_master where type = 'table' and name = 'vec_chunks'"
        ).fetchone()
    finally:
        store.close()

    assert row is not None
    assert "vec0" in row["sql"]


def test_sqlite_store_can_query_nearest_chunks_from_vec_index(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "huginn.db")
    try:
        store.replace_document(
            source_path=tmp_path / "alpha.pdf",
            sha256="a" * 64,
            modified_at="1",
            extracted_document=ExtractedDocument(
                source_path=str(tmp_path / "alpha.pdf"),
                title="alpha",
                pages=[ExtractedPage(page_number=1, text="Alpha budget")],
            ),
            chunks=[
                ChunkRecord(
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                    text="Alpha budget",
                    token_count=2,
                ),
                ChunkRecord(
                    chunk_index=1,
                    page_start=1,
                    page_end=1,
                    text="Completely different",
                    token_count=2,
                ),
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        rows = store.query_nearest_chunks([1.0, 0.0, 0.0], limit=1)
    finally:
        store.close()

    assert len(rows) == 1
    chunk, distance = rows[0]
    assert chunk.text == "Alpha budget"
    assert distance == 0.0
