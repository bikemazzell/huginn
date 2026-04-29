from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from huginn.chunking.split import chunk_document
from huginn.config import RuntimeConfig
from huginn.extract.registry import get_extractor_for_path
from huginn.ingest.discover import discover_supported_files
from huginn.ingest.fingerprint import sha256_file
from huginn.retrieve.basic import Embedder, lexical_features
from huginn.schemas import IngestResult
from huginn.store.sqlite import SQLiteStore


class IngestState(TypedDict):
    config: RuntimeConfig
    db_path: str
    embedder: Embedder | None
    reindex: bool
    discovered_files: list[Path]
    result: IngestResult


def run_ingest(
    config: RuntimeConfig,
    *,
    db_path: str | Path,
    embedder: Embedder | None = None,
    reindex: bool = False,
) -> IngestResult:
    graph = StateGraph(IngestState)
    graph.add_node("discover", _discover_files)
    graph.add_node("ingest", _ingest_files)
    graph.set_entry_point("discover")
    graph.add_edge("discover", "ingest")
    graph.add_edge("ingest", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {
            "config": config,
            "db_path": str(db_path),
            "embedder": embedder,
            "reindex": reindex,
            "discovered_files": [],
            "result": IngestResult(),
        }
    )
    return result["result"]


def _discover_files(state: IngestState) -> IngestState:
    discovered_files = discover_supported_files(state["config"].root_path)
    result = state["result"].model_copy(update={"discovered_count": len(discovered_files)})
    return {**state, "discovered_files": discovered_files, "result": result}


def _ingest_files(state: IngestState) -> IngestState:
    config = state["config"]
    embedder = state["embedder"]
    result = state["result"]
    store = SQLiteStore(state["db_path"])
    try:
        if state["reindex"]:
            store.reset()
        for path in state["discovered_files"]:
            file_sha = sha256_file(path)
            modified_at = path.stat().st_mtime_ns
            if store.has_unchanged_source(path, file_sha):
                result = result.model_copy(update={"skipped_count": result.skipped_count + 1})
                continue
            try:
                extractor = get_extractor_for_path(path, ocr_fallback=config.features.ocr_fallback)
                extracted_document = extractor.extract(path)
                chunks = chunk_document(
                    extracted_document,
                    chunk_size=config.indexing.chunk_size,
                    chunk_overlap=config.indexing.chunk_overlap,
                )
                if embedder is None:
                    embeddings = [lexical_features(chunk.text) for chunk in chunks]
                else:
                    embeddings = embedder.embed_texts(
                        [chunk.text for chunk in chunks], kind="document"
                    )
                store.replace_document(
                    source_path=path,
                    sha256=file_sha,
                    modified_at=str(modified_at),
                    extracted_document=extracted_document,
                    chunks=chunks,
                    embeddings=embeddings,
                )
                result = result.model_copy(update={"indexed_count": result.indexed_count + 1})
            except Exception as exc:  # pragma: no cover - exercised via e2e failure paths later
                store.mark_failed(path, file_sha, str(modified_at), str(exc))
                result = result.model_copy(update={"failed_count": result.failed_count + 1})
    finally:
        store.close()

    return {**state, "result": result}
