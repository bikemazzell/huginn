from pathlib import Path

import pytest

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


def test_replace_document_removes_orphaned_vec_rows_on_reingest(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "huginn.db")
    source_path = tmp_path / "alpha.pdf"
    try:
        for index, text in enumerate(("Alpha budget", "Alpha forecast"), start=1):
            store.replace_document(
                source_path=source_path,
                sha256=str(index) * 64,
                modified_at=str(index),
                extracted_document=ExtractedDocument(
                    source_path=str(source_path),
                    title="alpha",
                    pages=[ExtractedPage(page_number=1, text=text)],
                ),
                chunks=[
                    ChunkRecord(
                        chunk_index=0,
                        page_start=1,
                        page_end=1,
                        text=text,
                        token_count=2,
                    )
                ],
                embeddings=[[1.0, 0.0, 0.0]],
            )

        vec_count = store.connection.execute("select count(*) from vec_chunks").fetchone()[0]
        chunk_count = store.connection.execute("select count(*) from chunks").fetchone()[0]
    finally:
        store.close()

    assert vec_count == 1
    assert chunk_count == 1


def test_replace_document_rejects_embedding_dimension_mismatch_for_existing_vec_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "huginn.db"
    source_path = tmp_path / "alpha.pdf"

    store = SQLiteStore(db_path)
    try:
        store.replace_document(
            source_path=source_path,
            sha256="a" * 64,
            modified_at="1",
            extracted_document=ExtractedDocument(
                source_path=str(source_path),
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
    finally:
        store.close()

    reopened = SQLiteStore(db_path)
    try:
        with pytest.raises(ValueError, match="Embedding dimension changed"):
            reopened.replace_document(
                source_path=source_path,
                sha256="b" * 64,
                modified_at="2",
                extracted_document=ExtractedDocument(
                    source_path=str(source_path),
                    title="alpha",
                    pages=[ExtractedPage(page_number=1, text="Alpha revised")],
                ),
                chunks=[
                    ChunkRecord(
                        chunk_index=0,
                        page_start=1,
                        page_end=1,
                        text="Alpha revised",
                        token_count=2,
                    )
                ],
                embeddings=[[1.0, 0.0]],
            )
    finally:
        reopened.close()


def test_replace_document_derives_file_type_from_source_path(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "huginn.db")
    source_path = tmp_path / "alpha.txt"
    try:
        store.replace_document(
            source_path=source_path,
            sha256="a" * 64,
            modified_at="1",
            extracted_document=ExtractedDocument(
                source_path=str(source_path),
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
            embeddings=[{"alpha": 1.0}],
        )
        file_type = store.connection.execute(
            "select file_type from source_files where path = ?",
            (str(source_path),),
        ).fetchone()[0]
    finally:
        store.close()

    assert file_type == "txt"
