# Huginn — System Specification

> Living document. Update this file whenever the codebase changes in a way that affects behavior, interfaces, or data models.

## 1. Overview

Huginn is a **local-first, model-agnostic document RAG system**. It indexes a folder of PDFs recursively and answers natural-language questions over the corpus with grounded answers and citations.

The system is designed in two phases:

- **Phase 1 (complete)** — basic retrieve-then-read: discover, extract, chunk, embed, retrieve top-k, answer with citations.
- **Phase 2 (not yet implemented)** — query rewriting, reranking, answer validation, richer eval comparisons. Earlier no-op stubs for these were removed during a cleanup pass; the modules will be added when each feature is built. See §16.

---

## 2. Product Behavior

### 2.1 Ingest

1. User points Huginn at a folder via CLI.
2. Huginn recursively discovers supported files (PDF only in Phase 1).
3. For each file:
   - Fingerprints via SHA-256; skips unchanged files on re-ingest.
   - Extracts text page-by-page using `pypdf`.
   - Falls back to OCR sidecar text (`<filename>.ocr.txt`) when a page has no extractable text and `features.ocr_fallback` is enabled.
   - Chunks the extracted text with configurable size and overlap.
   - Embeds chunks via the configured embedding model (or a lexical fallback).
   - Persists metadata, chunk text, and vectors to SQLite.
4. Reports counts: discovered, indexed, skipped, failed.

### 2.2 Query

1. User asks a natural-language question via CLI.
2. Huginn embeds the question.
3. Retrieves top-k chunks by cosine similarity against stored vectors.
4. Generates an answer from the top retrieved chunk using the chat model.
5. Returns the answer text, citations (file + page range), and an evidence note.

If no sufficiently relevant chunks are found, returns a safe no-answer response instead of fabricating one.

### 2.3 Status

Prints aggregate counts from the database: source files, documents, chunks.

---

## 3. Architecture

### 3.1 Pipeline Orchestration

Three LangGraph `StateGraph` pipelines, each defined in `src/huginn/graph/`:

| Graph | Nodes | Purpose |
|---|---|---|
| `ingest_graph` | `discover → ingest` | Walk folder, extract, chunk, embed, persist |
| `query_graph` | `retrieve → answer` | Embed question, retrieve chunks, generate answer |
| `eval_graph` | `run_cases → report` | Run eval cases through query graph, compute metrics |

Each graph uses a `TypedDict` state and compiles to a linear (non-branching) flow. No agent tools, planners, or autonomous loops.

### 3.2 Data Flow Layers

```
src/huginn/
├── cli.py                  CLI entry point (argparse)
├── config.py               YAML config loader → RuntimeConfig
├── schemas.py              Pydantic models (shared data contracts)
├── preflight.py            Runtime readiness checks
│
├── ingest/
│   ├── discover.py         Recursive file discovery, extension filter, dotdir exclusion
│   └── fingerprint.py      SHA-256 file hashing
│
├── extract/
│   ├── registry.py         Routes file path → extractor by extension
│   └── pdf.py              pypdf extraction + OCR sidecar fallback
│
├── chunking/
│   └── split.py            Word-level windowed chunking with page-aware boundaries
│
├── store/
│   ├── sqlite.py           SQLiteStore: CRUD for sources, documents, pages, chunks, embeddings
│   ├── migrations/          Schema DDL (001_init.sql)
│   └── queries/             Named SQL files loaded on demand
│
├── retrieve/
│   └── basic.py            Embedding-based and lexical retrieval with cosine similarity
│
├── answer/
│   └── generate.py         Chat-model grounded answer generation with citation formatting
│
├── llm/
│   ├── factory.py          Builds embedder + chat clients from config; Nomic prefix handling
│   └── openai_compatible.py HTTP shim for OpenAI-compatible /embeddings and /chat/completions
│
├── eval/
│   ├── dataset.py          Loads eval cases from JSON
│   ├── metrics.py          Metric functions: retrieval hit, citation correctness, groundedness, etc.
│   └── report.py           Aggregates metrics into a report dict
│
└── graph/
    ├── ingest_graph.py     LangGraph ingest pipeline
    ├── query_graph.py      LangGraph query pipeline
    └── eval_graph.py       LangGraph eval pipeline
```

### 3.3 Key Design Decisions

- **Model-agnostic**: all model interaction goes through OpenAI-compatible HTTP endpoints. No provider SDKs; uses `urllib` directly.
- **Local-first**: no cloud dependencies. Two local `llama.cpp` servers (chat on `:1234`, embeddings on `:1235`).
- **Extractor registry**: file-type routing in `extract/registry.py` keeps PDF-specific code isolated. Adding a new file type means adding an extractor class and extending the registry.
- **Dual retrieval**: dense vector search via `sqlite-vec` when embeddings are available; sparse lexical (bag-of-words + cosine) as a zero-dependency fallback.
- **Feature flags**: `features.*` in config gate Phase 2 capabilities. Phase 1 features (`ocr_fallback`) are also togglable.

---

## 4. Data Model

### 4.1 Database Schema (SQLite)

Five core tables plus one virtual table:

```sql
source_files (
  source_file_id  INTEGER PRIMARY KEY,
  path            TEXT NOT NULL UNIQUE,
  sha256          TEXT NOT NULL,
  file_type       TEXT NOT NULL,          -- "pdf" (extensible)
  modified_at     TEXT NOT NULL,
  status          TEXT NOT NULL,          -- "indexed" | "failed"
  error_message   TEXT
)

documents (
  document_id          INTEGER PRIMARY KEY,
  source_file_id       INTEGER NOT NULL REFERENCES source_files ON DELETE CASCADE,
  title                TEXT NOT NULL,
  page_count           INTEGER NOT NULL,
  extracted_text_hash  TEXT NOT NULL
)

pages (
  page_id     INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents ON DELETE CASCADE,
  page_number INTEGER NOT NULL,
  text        TEXT NOT NULL
)

chunks (
  chunk_id     INTEGER PRIMARY KEY,
  document_id  INTEGER NOT NULL REFERENCES documents ON DELETE CASCADE,
  page_start   INTEGER NOT NULL,
  page_end     INTEGER NOT NULL,
  chunk_index  INTEGER NOT NULL,
  text         TEXT NOT NULL,
  token_count  INTEGER NOT NULL
)

chunk_embeddings (
  chunk_id    INTEGER PRIMARY KEY REFERENCES chunks ON DELETE CASCADE,
  vector_json TEXT NOT NULL                -- JSON-serialized embedding (dense list or sparse dict)
)

vec_chunks (                              -- sqlite-vec virtual table (created on first dense embedding)
  chunk_id  INTEGER PRIMARY KEY,
  embedding FLOAT[dimensions]
)
```

### 4.2 Pydantic Models (`schemas.py`)

| Model | Purpose |
|---|---|
| `RuntimeConfig` | Root config: `root_path`, `local_only`, `models`, `indexing`, `features` |
| `ModelsConfig` | Holds `chat` and `embedding` `ModelEndpointConfig` |
| `IndexingConfig` | `chunk_size`, `chunk_overlap`, `top_k` (validated: overlap < size) |
| `FeaturesConfig` | `ocr_fallback`, `query_rewrite`, `rerank`, `answer_validation` |
| `ExtractedDocument` | Output of extraction: `source_path`, `title`, `pages` |
| `ExtractedPage` | Single page: `page_number`, `text` |
| `ChunkRecord` | Chunk output: `chunk_index`, `page_start`, `page_end`, `text`, `token_count` |
| `RetrievedChunk` | Retrieval result: `chunk_id`, `source_path`, `page_start`, `page_end`, `text`, `score` |
| `QueryAnswer` | Final answer: `answer_text`, `citations`, `evidence_note` |
| `IngestResult` | Ingest summary: `discovered_count`, `indexed_count`, `skipped_count`, `failed_count` |

### 4.3 Graph States

| State | TypedDict fields |
|---|---|
| `IngestState` | `config`, `db_path`, `embedder`, `reindex`, `discovered_files`, `result` |
| `QueryState` | `config`, `db_path`, `question`, `embedder`, `chat_model`, `chunks`, `answer` |
| `EvalState` | `config`, `db_path`, `cases`, `embedder`, `chat_model`, `answers`, `report` |

---

## 5. Configuration

Single source of truth: `config/runtime.yaml`.

```yaml
root_path: tests/fixtures/corpus     # corpus root (can be overridden via CLI)
local_only: true

models:
  chat:
    base_url: http://127.0.0.1:1234/v1
    api_key: ollama
    model: Qwen3.5-9B-Q4_K_M.gguf
  embedding:
    base_url: http://127.0.0.1:1235/v1
    api_key: ollama
    model: nomic-embed-text-v2-moe

indexing:
  chunk_size: 128
  chunk_overlap: 24
  top_k: 4

features:
  ocr_fallback: true
  query_rewrite: false      # Phase 2 flag — no implementation yet
  rerank: false             # Phase 2 flag — no implementation yet
  answer_validation: false  # Phase 2 flag — no implementation yet
```

The Phase 2 flags exist in the schema so config files don't need editing later. The flags are currently inert — earlier no-op stub modules were removed during a cleanup pass to avoid the impression of partial behaviour.

All config is validated at load time by Pydantic (`extra="forbid"` on all models). Invalid keys, missing fields, or constraint violations raise immediately.

---

## 6. LLM Integration

### 6.1 Clients

Built in `llm/factory.py` via `build_runtime_clients()`:

- **`OpenAICompatibleEmbedder`** — POSTs to `/embeddings`. Supports single and batch embedding.
- **`OpenAICompatibleChatModel`** — POSTs to `/chat/completions`. Returns assistant message content.
- **`LocalLexicalEmbedder`** — in-process fallback that produces sparse bag-of-words vectors (no model call).
- **`NomicPrefixEmbedder`** — decorator that prepends `search_query:` / `search_document:` for Nomic models.

### 6.2 Embedding Model Handling

- Models whose name starts with `nomic-embed-text-` automatically get task-specific prefixes prepended.
- If the model name is `"local-lexical"`, the system uses a zero-dependency bag-of-words embedder instead of making HTTP calls.
- Dense embeddings (lists of floats) are stored in both `chunk_embeddings.vector_json` (for fallback/sparse retrieval) and `vec_chunks` (for fast ANN via sqlite-vec).
- Sparse embeddings (dicts of token→count) are stored only in `chunk_embeddings.vector_json`.

### 6.3 Reindex Requirement

After changing the embedding model, endpoint, or prefix logic, the on-disk vector index is invalid. The user must run `ingest --reindex` to rebuild from scratch.

---

## 7. CLI

```
huginn --config <path> --db-path <path> <command>
```

| Command | Arguments | Behavior |
|---|---|---|
| `ingest <folder>` | `--reindex` | Ingest folder, rebuild index if `--reindex` |
| `ask <question>` | — | Query the corpus, return JSON answer |
| `status` | — | Print source/document/chunk counts as JSON |

The `--config` flag defaults to `config/runtime.yaml`. The `--db-path` flag defaults to `data/huginn.db`.

Entry point: `huginn.cli:main` (registered as `huginn` console script, also runnable as `python -m huginn.cli`).

---

## 8. Chunking

Implemented in `chunking/split.py`. Algorithm:

1. Flatten all pages into a single word list, tracking which page each word came from.
2. Generate chunk start positions at every `step = chunk_size - chunk_overlap` words.
3. Also ensure a chunk starts at each page boundary (to avoid cross-page chunks silently skipping page starts).
4. For each start position, take a window of `chunk_size` words.
5. Record `page_start` / `page_end` as the min/max page numbers in the window.
6. `token_count` equals the number of words in the chunk.

This produces overlapping, page-aware chunks suitable for citation tracking.

---

## 9. Retrieval

Implemented in `retrieve/basic.py`. Two modes:

### 9.1 Dense Retrieval (default)

When a real `Embedder` is provided:
1. Embed the question via `embedder.embed_text(question, kind="query")`.
2. Query `vec_chunks` via sqlite-vec for the `limit` nearest neighbors.
3. Return chunks sorted by ascending distance (lower = more similar).

### 9.2 Lexical Fallback

When no embedder is provided:
1. Compute sparse bag-of-words vectors for the question and all stored chunks (stopword-filtered).
2. Score via cosine similarity.
3. Filter out zero-similarity matches, sort by score descending, return top-k.

### 9.3 Scoring

- Dense: score = negative distance (higher = better match).
- Lexical: score = cosine similarity (0 to 1).
- Secondary sort: fewer pages, shorter text, lower chunk_id.

---

## 10. Answer Generation

Implemented in `answer/generate.py`.

1. If no chunks were retrieved, return a fixed no-answer response.
2. Otherwise, take the top-ranked chunk.
3. If a `ChatModel` is available, call it with a system prompt ("Answer only from the supplied context...") and the chunk text as context.
4. If no chat model is available, return the top chunk text verbatim as the answer.
5. Format citations as `<filename>#page=<N>` or `<filename>#pages=<N>-<M>`.
6. Return `QueryAnswer` with `answer_text`, `citations`, and `evidence_note`.

### 10.1 No-Answer Policy

When retrieval returns no results, the system returns:

```json
{
  "answer_text": "I could not find grounded evidence for that question.",
  "citations": [],
  "evidence_note": "No sufficiently relevant chunks were retrieved."
}
```

---

## 11. Evaluation

### 11.1 Eval Dataset

Located at `tests/fixtures/eval/dataset.json`. Each case:

```json
{
  "question": "...",
  "expected_citations": ["filename#page=N"],
  "expected_substrings": ["..."],
  "expect_no_answer": false
}
```

### 11.2 Metrics

Computed in `eval/metrics.py`:

| Metric | Logic |
|---|---|
| `retrieval_hit_rate` | At least one returned citation appears in expected citations |
| `citation_correctness` | Returned citations exactly match expected citations |
| `groundedness` | Citation correct AND answer contains all expected substrings (or no-answer correct for negatives) |
| `answer_trait_match` | Answer text contains all expected substrings |
| `no_answer_correctness` | No-answer behavior matches `expect_no_answer` flag |

### 11.3 Eval Runner

`scripts/run_eval.py` loads config, loads dataset, builds runtime clients, runs the eval graph, and prints the report as JSON.

---

## 12. Preflight

`scripts/preflight.py` validates the runtime environment:

| Check | Source |
|---|---|
| Python >= 3.12 | `sys.version_info` |
| `uv` available | `shutil.which("uv")` |
| Config loads and validates | `load_runtime_config()` |
| Endpoints reachable | HTTP GET `/models` |
| Chat model available in endpoint | Model list fetched and matched |
| Embedding model available in endpoint | Model list fetched and matched |
| Embedding call succeeds | `embedder.embed_text("preflight ping", kind="query")` |
| Chat call succeeds | `chat_model.complete()` with ping/pong |
| `sqlite-vec` loads | In-memory extension load test |
| PDF deps (`pypdf`) importable | Hardcoded `True` (no actual import check) |
| OCR configured | Reads `config.features.ocr_fallback` (no real OCR-tooling check) |

Model name matching normalizes by stripping `.gguf` suffix and comparing case-insensitively with prefix matching.

Known issue: `_check_endpoint` in `scripts/preflight.py` references `urllib.request` without importing it. The urllib branch raises `NameError` and silently falls through to a `curl` subprocess. Either fix the import or delete the helper and route reachability through `huginn.preflight.fetch_model_ids`.

---

## 13. Test Strategy

### 13.1 Test Tiers

| Tier | Location | Scope | Model Calls |
|---|---|---|---|
| Validation | `tests/validation/` | Config parsing, schema validation, CLI input validation, unsupported file types | Mocked |
| Unit | `tests/unit/` | File discovery, fingerprinting, chunking, citations, retrieval, PDF extraction, store, preflight, client building, Nomic prefixes | Mocked |
| Smoke | `tests/smoke/` | Minimal ingest + query pipeline wiring | Fake/stub models |
| E2E | `tests/e2e/` | Full ingest → query cycle on fixture corpus | Fake/stub models |
| Regression | `tests/regression/` | Re-ingest skip-unchanged behavior | Fake/stub models |

Run command:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

### 13.2 Test Fixtures

Located in `tests/fixtures/corpus/`:

| File | Purpose |
|---|---|
| `atlas.pdf` | Text-based PDF (synthetic, created by `tests/helpers.py`) |
| `minutes.pdf` | Text-based PDF |
| `negative.pdf` | Unrelated content for no-answer testing |
| `scan.pdf` | Weak/no text PDF for OCR fallback testing |
| `scan.ocr.txt` | Sidecar OCR text for `scan.pdf` (form-feed delimited pages) |

PDFs are generated programmatically via `write_pdf()` in `tests/helpers.py` — no binary fixtures checked in.

### 13.3 Test Conventions

- `conftest.py` adds `src/` to `sys.path`.
- Most tests mock model/embedding calls. Smoke and E2E use stub/fake models.
- Real model-backed tests are not in the default suite (would require running llama.cpp servers).

---

## 14. Dependencies

### 14.1 Runtime

Declared in `pyproject.toml`:

| Package | Version Constraint | Purpose |
|---|---|---|
| `langchain-core` | `>=0.3,<2` | Base abstractions for LangGraph |
| `langgraph` | `>=0.4,<2` | Pipeline graph orchestration |
| `pypdf` | `>=5.4,<7` | PDF text extraction |
| `pydantic` | `>=2.10,<3` | Schema validation and data models |
| `pyyaml` | `>=6.0.2,<7` | YAML config loading |
| `sqlite-vec` | (transitive) | Dense vector index via SQLite virtual table |

### 14.2 Dev

| Package | Version Constraint | Purpose |
|---|---|---|
| `pytest` | `>=8.3,<9` | Test runner |

### 14.3 Implicit Runtime Requirements

- Python 3.12+
- Two OpenAI-compatible endpoints (typically `llama.cpp` servers)
- SQLite with extension loading support

---

## 15. File Layout (Actual)

```
huginn/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .gitignore
├── config/
│   ├── runtime.yaml
│   └── prompts/
│       ├── answer.txt
│       ├── rewrite_query.txt           # Phase 2 placeholder
│       └── validate_answer.txt         # Phase 2 placeholder
├── docs/
│   ├── PLAN.md
│   ├── spec.md
│   └── local-llamacpp-setup.md
├── scripts/
│   ├── preflight.py
│   ├── run_eval.py
│   ├── build_eval_set.py
│   └── generate_fixtures.py
├── src/huginn/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── preflight.py
│   ├── graph/
│   │   ├── ingest_graph.py
│   │   ├── query_graph.py
│   │   └── eval_graph.py
│   ├── llm/
│   │   ├── factory.py
│   │   └── openai_compatible.py
│   ├── ingest/
│   │   ├── discover.py
│   │   └── fingerprint.py
│   ├── extract/
│   │   ├── registry.py
│   │   └── pdf.py
│   ├── chunking/
│   │   └── split.py
│   ├── store/
│   │   ├── sqlite.py
│   │   ├── migrations/
│   │   │   └── 001_init.sql
│   │   └── queries/                    # 19 named SQL files
│   ├── retrieve/
│   │   ├── basic.py
│   │   └── stopwords.txt
│   ├── answer/
│   │   └── generate.py
│   └── eval/
│       ├── dataset.py
│       ├── metrics.py
│       └── report.py
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── validation/
│   │   ├── test_cli.py
│   │   ├── test_config.py
│   │   └── test_extract_validation.py
│   ├── unit/                           # 15 test files
│   ├── smoke/
│   │   └── test_smoke_pipeline.py
│   ├── e2e/
│   │   └── test_ingest_and_query.py
│   ├── regression/
│   │   └── test_reingest_skip_unchanged.py
│   └── fixtures/
│       ├── corpus/                     # atlas, minutes, negative, scan PDFs + OCR sidecar
│       └── eval/
│           └── dataset.json
└── data/                               # Runtime DB (gitignored)
```

---

## 16. Phase 2 Plan

Phase 2 is **not yet implemented**. Config flags, prompt placeholders in `config/prompts/`, and the PLAN sections are in place to capture intent. Earlier no-op stub modules (`retrieve/rewrite.py`, `retrieve/rerank.py`, `answer/validate.py`) were removed during a cleanup pass — they accepted an `enabled` flag, ignored it, and returned input unchanged. The module paths below are targets for future implementations, not files that currently exist.

Recommended implementation order:

1. **Query rewriting** — new file `retrieve/rewrite.py`, gated by `features.query_rewrite`.
   - Rewrite ambiguous questions into better retrieval queries before embedding.
   - Add a `rewrite` node between entry and `retrieve` in the query graph.

2. **Reranking** — new file `retrieve/rerank.py`, gated by `features.rerank`.
   - Re-score initial top-k results using a cross-encoder or LLM-based reranker.
   - Add a `rerank` node between `retrieve` and `answer` in the query graph.

3. **Answer validation** — new file `answer/validate.py`, gated by `features.answer_validation`.
   - Post-check whether the generated answer is grounded in cited chunks.
   - Add a `validate` node after `answer` in the query graph.

4. **Richer eval comparisons**
   - Run eval with each Phase 2 feature toggled on/off independently.
   - Compare baseline vs. Phase 2 metrics.

Each Phase 2 feature must be independently togglable via config. The Phase 1 baseline must remain fully functional when all Phase 2 flags are `false`.

---

## 17. Invariants

These must hold at all times. If one breaks, it's a bug.

1. Re-ingesting an unchanged corpus (same files, same content) produces `skipped_count == total`, `indexed_count == 0`.
2. `ingest --reindex` produces a clean database regardless of prior state.
3. No-answer queries never return fabricated content — they return the fixed no-answer message with empty citations.
4. All citations reference real source files and page numbers present in the indexed corpus.
5. Dense and lexical retrieval paths produce the same shape of output (`list[RetrievedChunk]`).
6. `RuntimeConfig` validation rejects unknown keys, missing fields, and `chunk_overlap >= chunk_size`.
7. After changing the embedding model or endpoint, `ingest --reindex` is required — stale vectors will not match.
8. All tests pass without running llama.cpp servers (stubs and mocks only).
9. The `local_only: true` config flag means no outbound network calls to cloud services.
10. File discovery skips dot-directories (e.g., `.git/`, `.venv/`).
