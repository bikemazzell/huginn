import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import sqlite_vec

from huginn.schemas import ChunkRecord, ExtractedDocument, RetrievedChunk


_QUERY_DIR = Path(__file__).parent / "queries"
_MIGRATION_DIR = Path(__file__).parent / "migrations"
_VEC_DIMENSION_RE = re.compile(r"embedding float\[(\d+)\]")


@lru_cache(maxsize=None)
def _query(name: str) -> str:
    return (_QUERY_DIR / f"{name}.sql").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _migration(name: str) -> str:
    return (_MIGRATION_DIR / name).read_text(encoding="utf-8")


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        self.connection.execute(_query("pragma_foreign_keys"))
        self.connection.executescript(_migration("001_init.sql"))
        self._vec_dimensions: int | None = None

    def close(self) -> None:
        self.connection.close()

    def reset(self) -> None:
        self.connection.executescript(_query("reset"))
        self.connection.commit()

    def has_unchanged_source(self, path: str | Path, sha256: str) -> bool:
        row = self.connection.execute(
            _query("select_source_sha"),
            (str(Path(path)),),
        ).fetchone()
        return row is not None and row["sha256"] == sha256

    def replace_document(
        self,
        *,
        source_path: str | Path,
        sha256: str,
        modified_at: str,
        extracted_document: ExtractedDocument,
        chunks: Iterable[ChunkRecord],
        embeddings: list[list[float] | dict[str, float]],
    ) -> None:
        source_path_str = str(Path(source_path))
        file_type = _file_type_for_path(source_path)
        cursor = self.connection.cursor()
        existing = cursor.execute(
            _query("select_source_file_id"),
            (source_path_str,),
        ).fetchone()

        if existing is not None:
            source_file_id = existing["source_file_id"]
            cursor.execute(
                _query("delete_vec_chunks_for_source"),
                (source_file_id,),
            )
            document_rows = cursor.execute(
                _query("select_documents_for_source"),
                (source_file_id,),
            ).fetchall()
            for row in document_rows:
                cursor.execute(_query("delete_document"), (row["document_id"],))
            cursor.execute(
                _query("update_source_file"),
                (sha256, file_type, modified_at, "indexed", source_file_id),
            )
        else:
            cursor.execute(
                _query("insert_source_file"),
                (source_path_str, sha256, file_type, modified_at, "indexed"),
            )
            source_file_id = int(cursor.lastrowid)

        extracted_text_hash = sha256_text("\n".join(page.text for page in extracted_document.pages))
        cursor.execute(
            _query("insert_document"),
            (
                source_file_id,
                extracted_document.title,
                extracted_document.page_count,
                extracted_text_hash,
            ),
        )
        document_id = int(cursor.lastrowid)

        for page in extracted_document.pages:
            cursor.execute(
                _query("insert_page"),
                (document_id, page.page_number, page.text),
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor.execute(
                _query("insert_chunk"),
                (
                    document_id,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.token_count,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            serialized_embedding = self._serialize_embedding(embedding)
            cursor.execute(
                _query("insert_chunk_embedding"),
                (chunk_id, json.dumps(embedding, sort_keys=True)),
            )
            if serialized_embedding is not None:
                self._ensure_vec_table(len(embedding))
                cursor.execute(
                    _query("insert_vec_chunk"),
                    (chunk_id, serialized_embedding),
                )

        self.connection.commit()

    def mark_failed(self, path: str | Path, sha256: str, modified_at: str, error_message: str) -> None:
        self.connection.execute(
            _query("upsert_failed_source"),
            (
                str(Path(path)),
                sha256,
                _file_type_for_path(path),
                modified_at,
                "failed",
                error_message,
            ),
        )
        self.connection.commit()

    def load_chunks(self) -> list[tuple[RetrievedChunk, list[float] | dict[str, float]]]:
        rows = self.connection.execute(_query("load_chunks")).fetchall()
        result: list[tuple[RetrievedChunk, list[float] | dict[str, float]]] = []
        for row in rows:
            result.append(
                (
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        source_path=row["source_path"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        text=row["text"],
                        score=0.0,
                    ),
                    json.loads(row["vector_json"]),
                )
            )
        return result

    def query_nearest_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int,
    ) -> list[tuple[RetrievedChunk, float]]:
        if self._vec_dimensions is None:
            row = self.connection.execute(_query("select_vec_chunks_table")).fetchone()
            if row is None:
                return []
        rows = self.connection.execute(
            _query("query_nearest_chunks"),
            (sqlite_vec.serialize_float32(query_embedding), limit),
        ).fetchall()
        return [
            (
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    source_path=row["source_path"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    text=row["text"],
                    score=float(row["distance"]),
                ),
                float(row["distance"]),
            )
            for row in rows
        ]

    def status_counts(self) -> dict[str, int]:
        row = self.connection.execute(_query("status_counts")).fetchone()
        return {
            "source_file_count": row["source_file_count"],
            "document_count": row["document_count"],
            "chunk_count": row["chunk_count"],
        }

    def _ensure_vec_table(self, dimensions: int) -> None:
        if self._vec_dimensions == dimensions:
            return
        if self._vec_dimensions is None:
            existing_dimensions = self._existing_vec_dimensions()
            if existing_dimensions is None:
                self.connection.execute(
                    _query("create_vec_chunks").format(dimensions=dimensions)
                )
                self._vec_dimensions = dimensions
                return
            self._vec_dimensions = existing_dimensions
            if existing_dimensions != dimensions:
                raise ValueError(
                    f"Embedding dimension changed from {existing_dimensions} to {dimensions}"
                )
            return
        if self._vec_dimensions != dimensions:
            raise ValueError(
                f"Embedding dimension changed from {self._vec_dimensions} to {dimensions}"
            )

    def _serialize_embedding(
        self,
        embedding: list[float] | dict[str, float],
    ) -> bytes | None:
        if isinstance(embedding, list):
            return sqlite_vec.serialize_float32(embedding)
        return None

    def _existing_vec_dimensions(self) -> int | None:
        row = self.connection.execute(_query("select_vec_chunks_schema")).fetchone()
        if row is None:
            return None
        match = _VEC_DIMENSION_RE.search(row["sql"])
        if match is None:
            raise ValueError("Could not determine vec_chunks embedding dimensions")
        return int(match.group(1))


def sha256_text(value: str) -> str:
    from hashlib import sha256

    return sha256(value.encode("utf-8")).hexdigest()


def _file_type_for_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"
