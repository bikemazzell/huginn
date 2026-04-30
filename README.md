# Huginn

Huginn is a local-first document RAG system for recursively indexing a folder of PDFs and answering grounded questions with citations, including non-English corpora and queries.

The current implementation is a Phase 1 baseline:

- PDF-first ingestion
- sidecar OCR fallback for weak/scanned PDFs
- chunking and local persistence in SQLite
- embedding-backed retrieval
- grounded answer generation with citations
- local eval and preflight scripts
- OpenAI-compatible local model endpoints, tested with `llama.cpp`


## Current Status

Implemented now:

- recursive folder ingest
- PDF extraction with OCR sidecar fallback
- LangGraph-based ingest, query, and eval flows
- `sqlite-vec`-backed dense vector storage plus lexical retrieval fused for stronger exact-match recall
- Unicode-aware lexical retrieval, so Cyrillic and accented Latin queries are not dropped by the fallback path
- weak-evidence refusal via configurable dense/lexical retrieval thresholds
- optional query rewrite stage that rewrites only the retrieval query while preserving the original user question for answer generation
- optional lexical rerank stage that widens retrieval candidates before truncating back to `top_k`
- optional answer validation stage that rejects unsupported generated answers with a safe no-answer response
- eval runner with `precision@k`, `recall@k`, `MRR`, and baseline-vs-variant comparison output
- deterministic validation/unit/smoke/e2e/regression tests
- live local runtime with separate chat and embedding endpoints

Still planned for later:

- additional file types

## Project Layout

```text
config/                  Runtime config and prompt files
docs/                    PLAN.md, spec.md, and local setup notes
scripts/                 Preflight and eval helpers
src/huginn/              Main application code
tests/                   Validation, unit, smoke, e2e, regression, fixtures
models/                  Local model files
data/                    Local SQLite databases
```

## Local Model Setup

The default runtime assumes two local `llama.cpp` servers:

- chat model on `127.0.0.1:1234`
- embedding model on `127.0.0.1:1235`

Current default models in [config/runtime.yaml](config/runtime.yaml):

- chat: `Qwen3.5-9B-Q4_K_M.gguf`
- embeddings: `nomic-embed-text-v2-moe`

To start both servers in one shot (useful for smoke and e2e runs), use the launcher script:

```bash
HUGINN_CHAT_MODEL=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  HUGINN_CHAT_MMPROJ=/path/to/mmproj-F16.gguf \
  python scripts/start_llama_servers.py
```

The script blocks until both endpoints respond on `/v1/models`, then keeps running until interrupted; Ctrl-C terminates both child servers. See `--help` for port, context, and `-ngl` overrides. The chat model path and its matching `mmproj` path must be provided explicitly. The embed model defaults to the path in `models/`; override with `HUGINN_EMBED_MODEL` or `--embed-model`.

There is a more detailed setup note in [docs/local-llamacpp-setup.md](docs/local-llamacpp-setup.md).

## Installation

If your environment already has the dependencies you want, install Huginn editable without forcing dependency resolution:

```bash
pip install -e . --no-deps
```

If you want `pip` to install Huginn's declared dependencies as well:

```bash
pip install -e .
```

## Main Commands

Rebuild the local index from the fixture corpus:

```bash
python -m huginn.cli --config config/runtime.yaml --db-path data/huginn.db ingest --reindex tests/fixtures/corpus
```

Ask a question:

```bash
python -m huginn.cli --config config/runtime.yaml --db-path data/huginn.db ask "What is the launch date?"
```

Inspect the current database counts:

```bash
python -m huginn.cli --config config/runtime.yaml --db-path data/huginn.db status
```

Run preflight checks:

```bash
python scripts/preflight.py
```

Run the local eval set:

```bash
python scripts/run_eval.py
```

Compare multiple runtime configs against a baseline:

```bash
python scripts/run_eval.py --config config/runtime.yaml --config config/variant.yaml
```

Run the offline fixture-corpus eval used by CI:

```bash
python -m huginn.cli --config config/ci_eval.yaml --db-path /tmp/huginn-ci-eval.db ingest --reindex tests/fixtures/corpus
python scripts/run_eval.py --config config/ci_eval.yaml --db-path /tmp/huginn-ci-eval.db
```

## Runtime Notes

- Huginn is model-agnostic at the config level as long as the endpoints are OpenAI-compatible.
- For Nomic embeddings, Huginn automatically adds:
  - `search_query: ` for queries
  - `search_document: ` for indexed chunks
- If you change embedding models or embedding endpoints, run `ingest --reindex` to rebuild the vector index.

## Testing

Run the full local suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

The suite includes:

- validation tests
- unit tests
- smoke tests
- e2e tests
- regression tests

The repository also ships an offline CI eval path via [config/ci_eval.yaml](config/ci_eval.yaml) and [.github/workflows/ci.yml](.github/workflows/ci.yml), so retrieval and answer-quality regressions can be gated without live model endpoints.

## Storage

Huginn stores:

- source file metadata
- extracted documents and pages
- chunks and citations
- dense vectors in `sqlite-vec`

The store lives in [src/huginn/store/sqlite.py](src/huginn/store/sqlite.py). Schema is in [migrations/001_init.sql](src/huginn/store/migrations/001_init.sql) and all queries are kept as `.sql` files under [store/queries/](src/huginn/store/queries/).

## Eval

The local eval runner currently reports:

- retrieval hit rate
- precision@k
- recall@k
- mean reciprocal rank
- citation correctness
- groundedness
- answer trait match
- no-answer correctness

If multiple `--config` paths are provided, the first run is treated as baseline and the output includes metric deltas for the additional runs.
When a comparison run regresses any tracked metric versus baseline, `scripts/run_eval.py` exits non-zero and includes a `regressions` list in the JSON output.
The repository also includes a GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml) that runs the full test suite plus the offline fixture-corpus eval gate.

The default dataset lives in [tests/fixtures/eval/dataset.json](tests/fixtures/eval/dataset.json).

## Language Support

Huginn is not English-only. The current retrieval stack supports non-English text and queries, including Unicode lexical matching for scripts such as Cyrillic and accented Latin text.

Current practical status:
- multilingual exact-term and same-language retrieval work
- limited cross-language retrieval normalization now exists for common contact/info query terms
- broader mixed-language retrieval remains an active tuning area, especially when the user asks in one language about content written in another

## Planning

Status and next priorities are tracked in [docs/PLAN.md](docs/PLAN.md). The living system spec is in [docs/spec.md](docs/spec.md).
