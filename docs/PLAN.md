# Huginn — Generic Document RAG Implementation Plan

## Status Snapshot

As of 2026-04-29, Phase 1 plus the Phase 1.1 follow-ups are working locally.

Implemented:
- recursive PDF ingest
- PDF extraction with sidecar OCR fallback
- chunking, persistence, retrieval, and grounded answers with citations
- CLI commands for `ingest`, `ask`, and `status`
- validation/unit/smoke/e2e/regression test tiers (63 tests, all green)
- live OpenAI-compatible chat + embedding endpoints, tested against local `llama.cpp`
- fixture corpus, eval dataset, and local eval runner
- `sqlite-vec` for dense vector storage and KNN retrieval
- preflight that exercises live endpoint, model, embedding, and chat calls
- batched embedding with retry/split fallback so ingest can recover from embedding-server 500s caused by batch shape or oversized chunk inputs
- weak-evidence refusal via configurable lexical and dense retrieval thresholds
- query rewriting behind `features.query_rewrite`, using the chat model to rewrite only the retrieval query while preserving the original question for answer generation
- reranking behind `features.rerank`, using a widened retrieval pool and lexical overlap to reorder candidates before answer generation
- answer validation behind `features.answer_validation`, using the chat model to reject unsupported generated answers and replace them with the standard safe no-answer response
- eval metrics covering retrieval hit rate, `precision@k`, `recall@k`, MRR, citation correctness, groundedness, answer-trait match, and no-answer correctness
- eval runner support for baseline-vs-variant comparison output across multiple configs, with non-zero exit on regressions
- broader default eval coverage across positive retrieval, OCR fallback, and no-answer cases
- GitHub Actions workflow running the full test suite plus an offline fixture-corpus eval gate via `config/ci_eval.yaml`
- runtime/setup notes and a launcher script for the two-endpoint `llama.cpp` flow

Deviation from the original plan: the original no-op Phase 2 stubs were removed when they provided no behavior. Query rewriting, reranking, and answer validation have since been reintroduced as real implementations.

Checklists below are the source of truth for what is complete versus open.

## Next Priorities

### Phase 1.1

This follow-up block is complete:

1. `[x]` Replace JSON vector storage with real `sqlite-vec` integration.
2. `[x]` Strengthen `scripts/preflight.py` so it validates live embedding and chat calls, not just basic config/runtime checks.
3. `[x]` Expand eval metrics to cover groundedness and clearer retrieval-quality signals.
4. `[x]` Add a small amount of operational documentation for the current two-endpoint local setup (`Qwen` chat + `Nomic` embeddings via `llama.cpp`).

### Phase 2 Ready Queue

Once Phase 1.1 is complete, the recommended next order is:

1. broader real-world eval corpus growth

## Goal

Build a **generic document RAG system** that:

- indexes a folder recursively
- supports **PDF first**
- is shaped to add more file types later
- lets the user ask questions about the indexed corpus
- returns **grounded answers with citations**
- is **local-first**
- is **model-agnostic** through an **OpenAI-compatible endpoint**

The design should balance two goals:

1. **Phase 1** should stay simple enough to build and understand clearly.
2. **Phase 2** should capture more modern RAG patterns so they are not lost.

---

## 0. Design Decision

### Recommendation

Use:

- **Python 3.12**
- **LangChain core pieces**
- **LangGraph** for orchestration
- **OpenAI-compatible model endpoints**
- **SQLite + sqlite-vec**
- **Pydantic**

### Why this is the right compromise

- A fully custom pipeline would be lighter, but it teaches less about current RAG tooling.
- A heavily framework-owned stack would hide too much of the actual system behavior.
- This approach gives exposure to:
  - LangGraph stateful workflow design
  - model-agnostic endpoint usage
  - retrieval pipeline composition
  - explicit testing boundaries
- It still keeps storage and indexing understandable and under local control.

### Scope guard

Do **not** use LangGraph to make everything agentic.

For this project:
- LangGraph should orchestrate ingestion, query, and eval flows.
- It should not turn the app into a tool-calling multi-agent system in Phase 1.

---

## 1. Product Shape

The system should feel like this:

1. User points the app at a folder.
2. The app recursively discovers supported files.
3. It extracts text, chunks it, embeds it, and indexes it.
4. The user asks questions in natural language.
5. The app retrieves relevant chunks and produces an answer grounded in those chunks.
6. The answer cites source file and page or chunk references.

### Phase 1 promise

Phase 1 is **basic retrieve-then-read**:

- discover
- extract
- chunk
- embed
- retrieve top-k
- answer from retrieved context
- cite sources

### Phase 2 promise

Phase 2 adds more advanced RAG patterns:

- query rewriting
- optional reranking
- answer validation / groundedness checks
- retrieval evaluation improvements
- later, possibly hybrid retrieval and decomposition

---

## 2. Supported Inputs

### Phase 1

- **PDF only**

### Phase 1 PDF support policy

- text-based PDFs are required and must be supported well
- scanned or image-heavy PDFs are supported on a best-effort basis through OCR fallback
- encrypted, corrupted, or otherwise unsupported PDFs must fail gracefully with recorded ingest status
- OCR-heavy PDFs must be covered by fixtures and tests, not treated as an informal stretch goal

But the architecture should be generic from day one so more file types can be added later without refactoring the whole pipeline.

### Planned later file types

- `txt`
- `md`
- `html`
- `docx`

### Design rule

Use a file-type extractor interface rather than embedding PDF assumptions throughout the codebase.

---

## 3. Architecture

```text
User Folder
   |
   v
Discover Files
   |
   v
Extract Text (PDF first)
   |
   v
Chunk Text
   |
   v
Embed Chunks
   |
   v
Store Metadata + Vectors
   |
   v
Ask Question
   |
   v
Retrieve Relevant Chunks
   |
   v
Answer with Citations
```

### Core design principle

Keep the system **generic around documents and chunks**, not around receipt-like structured entities.

---

## 4. Directory Layout

```text
/huginn/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .gitignore
├── config/
│   ├── runtime.yaml
│   └── prompts/
├── docs/
│   ├── PLAN.md
│   ├── spec.md
│   └── local-llamacpp-setup.md
├── scripts/
│   ├── preflight.py
│   ├── build_eval_set.py
│   ├── generate_fixtures.py
│   └── run_eval.py
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
│   │   ├── pdf.py
│   │   └── registry.py
│   ├── chunking/
│   │   └── split.py
│   ├── store/
│   │   ├── sqlite.py
│   │   ├── migrations/001_init.sql
│   │   └── queries/*.sql
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
│   ├── validation/
│   ├── unit/
│   ├── smoke/
│   ├── e2e/
│   ├── regression/
│   └── fixtures/
│       ├── corpus/
│       └── eval/
├── data/         # local SQLite databases
└── models/       # local model files
```

Notes on layout choices:
- All SQL is stored as `.sql` files under `store/queries/` (loaded by name via `lru_cache`) and `store/migrations/`. The Python store module holds no SQL strings.
- Stopwords for lexical retrieval live in `retrieve/stopwords.txt` rather than a Python literal, so they can be edited without touching code.
- The `llm/` module ships only the `openai_compatible` HTTP client and a `factory` that wires up embedders (including the Nomic prefix wrapper). Earlier `structured.py` and `prompts.py` shims were removed when they turned out to have no callers.
- The `extract/` module is shaped for future file types via `registry.get_extractor_for_path`, but currently only PDF is wired up. There is no `Extractor` Protocol yet — it would be added the moment a second extractor exists.

---

## 5. Data Model

The storage layer should be generic and traceable.

### Tables

#### `source_files`

- `source_file_id`
- `path`
- `sha256`
- `file_type`
- `modified_at`
- `status`
- `error_message`

#### `documents`

- `document_id`
- `source_file_id`
- `title`
- `page_count`
- `extracted_text_hash`

#### `pages`

- `page_id`
- `document_id`
- `page_number`
- `text`

#### `chunks`

- `chunk_id`
- `document_id`
- `page_start`
- `page_end`
- `chunk_index`
- `text`
- `token_count`

#### `chunk_embeddings`

- vector row keyed by `chunk_id`

### Why this shape

- works well for PDFs
- still generalizes to non-page-based formats later
- keeps citations easy
- avoids overfitting to a domain schema

---

## 6. Runtime Configuration

`config/runtime.yaml` should be the single source of truth:

```yaml
root_path: /abs/path/to/documents
local_only: true

models:
  chat:
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    model: qwen3:8b

  embedding:
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    model: bge-m3

indexing:
  chunk_size: 800
  chunk_overlap: 120
  top_k: 6

features:
  ocr_fallback: true
  query_rewrite: false
  rerank: false
  answer_validation: false
```

### Notes

- Phase 1 defaults should keep advanced features off.
- Phase 2 turns them on selectively.
- The same structure should work against Ollama, `llama-server`, LM Studio, or another OpenAI-compatible provider.

---

## 7. Model Strategy

The system should stay **model-agnostic**.

### Phase 1 model requirements

- a chat model that follows instructions reliably
- an embedding model available through the configured endpoint

### Phase 2 optional model requirements

- a stronger model for reranking or answer validation if desired

### Important constraint

Do not pin the system design to one model family.

Use example defaults, but treat:
- `base_url`
- `api_key`
- `model`

as configuration only.

---

## 8. LangGraph Role

LangGraph is justified here because the workflow benefits from explicit state transitions, resumable execution, and clear graph boundaries for ingest, query, and eval flows.

### Phase 1 graphs

#### Ingest graph

- discover files
- filter supported file types
- extract text
- chunk text
- embed chunks
- persist metadata and vectors

#### Query graph

- accept user question
- retrieve top-k chunks
- generate grounded answer
- return answer with citations

#### Eval graph

- run canned test questions
- collect metrics
- emit report

### Phase 2 graph additions

- query rewrite node
- rerank node
- answer validation node

### Scope guard

Use LangGraph for **clear pipeline nodes and state**.
Do not introduce agent tools, planners, or autonomous looping in Phase 1.

---

## 9. Phase 1 Functional Scope

Phase 1 is deliberately basic.

### Features

- recursive file discovery
- PDF text extraction
- optional OCR fallback for weak PDFs
- chunking with overlap
- embedding and indexing
- semantic retrieval
- grounded answer generation
- source citations
- CLI commands for ingest, ask, and status
- test coverage across all required test categories

### Phase 1 definition of done

Phase 1 is done only when:

- ingest works recursively over a folder
- PDF extraction works on fixture text PDFs
- OCR fallback works on at least one fixture PDF
- unchanged files are skipped on re-ingest
- `ask` returns grounded answers with citations
- negative queries return safe no-answer behavior
- validation, unit, smoke, e2e, and regression tests all pass

### Non-goals

- query rewriting
- reranking
- answer self-critique
- hybrid retrieval
- multi-query decomposition
- MCP server
- non-PDF extraction implementations

Those belong in Phase 2 or later.

---

## 10. Phase 2 Scope

Phase 2 exists to capture additional RAG patterns without bloating Phase 1.

### Planned additions

#### Query rewriting

Rewrite ambiguous or underspecified user questions into better retrieval queries.

#### Reranking

Take initial retrieved chunks and rerank them before answer generation.

#### Answer validation

Check whether the answer is grounded in retrieved chunks and whether citations support claims.

#### Retrieval improvements

- optional hybrid retrieval
- optional metadata-aware filtering
- optional decomposition of harder questions

### Design intent

Phase 2 is where the system expands into a more capable RAG platform.

### Applied retrieval lessons (informing Phase 2 work)

These notes are distilled from production RAG systems on other domains and adapted to Huginn's local-first, generic-PDF context. They are guidance, not commitments.

#### Chunking

- Fixed-size word windows (current Phase 1) are simple and effective, but break context mid-clause. Phase 2 should explore semantic chunking — paragraph- or section-aware splits using PDF layout cues or heading detection. Tax forms, RFCs, and academic papers all reward this; pure prose less so.
- Chunk size ~500-800 tokens is the empirically-settled sweet spot for most prose. Huginn's current 128-word default is conservative: precise retrieval, but a single chunk rarely contains a full reasoning chain. Worth tuning once the eval set is broad enough to measure the trade-off. Larger chunks dilute the embedding signal and burn context budget; smaller chunks fragment reasoning.
- 10-15% token overlap is the usual recommendation for cross-reference continuity. Huginn's 24/128 = ~19% is in the right neighbourhood.
- Per-chunk metadata is the highest-leverage retrieval lever. Beyond `page_start/page_end`, add (when extractable): section heading, document title, file modified-at, file type. This is what makes the metadata-aware filtering item below actually useful.

#### Retrieval

- **Hybrid retrieval (dense + sparse)** catches what either method alone misses. Dense embeddings blur exact-match queries — section numbers, named entities, acronyms, dates. SQLite's built-in FTS5 module gives BM25 keyword scoring without a new dependency. Combine the two rankings via reciprocal rank fusion (`score = Σ 1/(k + rank_i)`).
- **Retrieve wide, rerank narrow**: top-N around 20-30 from retrieval, then rerank down to top-K = 4-6 before generation. Never feed 20 chunks straight into the LLM — it costs context and dilutes attention.
  - Phase 1.1 closed the earlier single-chunk prompt gap: the answer model now receives all retrieved chunks as context. Reranking still matters because retrieved context quality, ordering, and redundancy still directly affect the final answer.
- **Pre-filter by metadata before vector scoring** when corpus segmentation matters (file type, modified-at range, language). Optional for general PDF corpora; essential for time-sensitive ones.

#### Embeddings

- Local `llama.cpp` + Nomic is fine for Phase 1. When/if remote-endpoint support is added (cloud OpenAI, Cohere, Voyage, BGE v2), Matryoshka-truncated dimensions (1024 or 1536 instead of 3072) cut vector storage and KNN cost with minimal quality loss.
- Batched embedding ingestion is in place (`embed_texts`). Async worker pools are overkill for local-first single-user use.

#### Reranking

- First-pass retrieval optimizes for recall; the reranker optimizes for precision. Without reranking, the LLM ends up making the relevance judgement itself, which wastes context and degrades answers.
- Local-first implementation options: a cross-encoder model (e.g., `BAAI/bge-reranker-v2-m3`) loaded via `sentence-transformers` or run through llama.cpp; or LLM-as-reranker, reusing the configured chat model with a "score this chunk's relevance to this query" prompt. Hosted rerankers (Cohere Rerank, Jina Rerank, Azure AI Search semantic) are out of scope for the local-first baseline — leave them as a documented option behind a config flag if remote-endpoint support is ever added.
- The reported signal in production systems is a meaningful reduction in hallucination, not a fixed percentage. The eval set should measure faithfulness before and after reranking is enabled, on the same retrieval depth, to make the trade visible.

#### Generation

- The Phase 1 path uses a single chat model for the final answer. When Phase 2 query rewriting lands, it should be run through a smaller/cheaper model than the answerer (smaller local quant, or just a tighter system prompt with shorter max tokens). Heavy reasoning belongs only on the user-facing answer step.
- Prompt structure already in place: system prompt loaded from `config/prompts/answer.txt`, user question, and retrieved chunk(s) as context. The prompt remains intentionally simple for Phase 1.
- **Threshold-based refusal**: implemented. Phase 1 now refuses when retrieval yields only weak matches by applying configurable lexical and dense retrieval thresholds before answer generation.
- Long-context note: frontier models in 2026 ship with 1M+ token windows, which loosens the retrieval-precision constraint for high-stakes paths. Local llama.cpp models typically run 8k-32k, so this lever applies only if remote endpoints are added later. Even with long context, retrieval still matters for cost, latency, and the audit trail behind every cited claim.

#### Evaluation

- The Phase 1 eval dataset (`tests/fixtures/eval/dataset.json`) is hand-curated, which is the right pattern. Production analogues call this a "gold set" curated by domain experts, refreshed as the corpus changes. The CLI/eval runner is in place, the harness scaffolding is fine.
- Current metrics: retrieval hit rate, citation correctness, groundedness, answer-trait match, no-answer correctness, `precision@k`, `recall@k`, and mean reciprocal rank.
- Still missing for Phase 2 comparison work:
  - Faithfulness as a generation metric distinct from groundedness (does the answer claim only what the retrieved chunks actually say?).
- External harnesses to know about: RAGAS is the most common in 2026, DeepEval and Braintrust are alternatives. Adopting one is optional — the local report dict is enough for Phase 1, but if Phase 2 needs richer per-case breakdowns, RAGAS-style metric definitions are a sensible reference.
- Regression discipline: every retrieval, prompt, or model change should run the eval set and gate merge on no regression. Today `scripts/run_eval.py` can compare multiple configs and emit metric deltas; wiring that into CI is the next small piece of work.

---

## 11. CLI Shape

Start with a minimal CLI:

```bash
huginn ingest /path/to/folder
huginn ask "What does this corpus say about ...?"
huginn status
```

### Notes

- `ingest` should recurse by default
- re-ingesting should skip unchanged files by `sha256`
- `ask` should print answer plus citations
- `status` should show counts and last ingest status

---

## 12. TDD and Test Strategy

This project should be **TDD-driven**.

That is not optional.

### Required testing layers

#### 1. Validation tests

Cover:
- config parsing
- schema validation
- invalid CLI inputs
- unsupported file types
- malformed extraction outputs

#### 2. Unit tests

Cover:
- file discovery
- hashing/fingerprinting
- chunking rules
- citation formatting
- retrieval helpers
- LangGraph node logic
- answer validation helpers

#### 3. Smoke tests

Cover:
- minimal ingest on tiny fixture corpus
- minimal ask flow on a known fixture
- ensures the main pipeline is wired correctly

#### 4. E2E tests

Cover:
- ingesting fixture PDFs
- querying the indexed corpus
- verifying grounded answers and citations
- verifying the full LangGraph flows work together

#### 5. Regression tests

Every bug fix must add a regression test.

Examples:
- broken PDF extraction edge case
- chunk boundary issue
- retrieval miss on previously working content
- hallucinated answer not supported by citation

### TDD workflow rule

For each feature:

1. write a failing test
2. implement the smallest change to pass
3. refactor safely
4. add smoke or e2e coverage if the feature is user-visible
5. preserve bugs as regression tests

### Phase gate between Phase 1 and Phase 2

Do not start Phase 2 work until all of the following are true:

- validation tests pass
- unit tests pass
- smoke tests pass
- e2e tests pass
- regression tests pass
- citation correctness meets the agreed threshold in eval
- at least one OCR fallback fixture passes end to end
- at least one negative-query e2e test passes

---

## 13. Test Fixture Strategy

The project needs a deterministic local fixture corpus.

### `tests/fixtures/corpus/`

Include:
- a text-based PDF
- a scanned or weak-text PDF
- a multi-page PDF
- a no-answer negative case

### Why this matters

- stable TDD loop
- reproducible e2e tests
- regression coverage without relying on user documents

### Mocking rule

For most validation and unit tests:
- mock model calls
- mock embedding calls

For smoke tests:
- prefer a deterministic local stub or fixed fake model layer

For default e2e tests:
- use deterministic local stubs or fixed fake model responses so the suite stays fast and repeatable

For optional real-model e2e tests:
- keep them in a separate suite
- run them only when explicitly requested or in an environment prepared for local model-backed testing

If true model-backed e2e tests are added later, keep them separate from the default fast suite.

---

## 14. Retrieval and Answering Rules

### Phase 1 retrieval

- embed the user question
- retrieve top-k chunks
- pass retrieved chunks to the answer model

### Phase 1 answer generation rules

- answer only from retrieved chunks
- cite source file and page range or chunk reference
- if the evidence is weak, say so
- do not invent facts not supported by retrieved context

### Phase 1 output expectation

Each answer should include:
- answer text
- citations
- optional confidence or evidence note

---

## 15. Preflight

`scripts/preflight.py` should validate:

1. `[x]` Python version (>= 3.12)
2. `[x]` `uv` availability
3. `[x]` configured endpoint reachability
4. `[x]` model availability from the configured endpoint
5. `[x]` embedding call success
6. `[x]` chat call success
7. `[x]` SQLite + `sqlite-vec` availability
8. `[~]` PDF extraction dependencies — currently hardcoded `True`. Real check would import `pypdf` and the OCR sidecar reader.
9. `[~]` OCR dependency if OCR fallback is enabled — currently echoes `config.features.ocr_fallback` rather than checking that `tesseract` (or whatever OCR tooling is in use) is present.

### Non-goals

Preflight should not:
- auto-download large models without approval
- scrape the web for latest model names
- couple the app to a specific provider CLI

---

## 16. Eval Strategy

Evaluation should exist from the beginning, even if it is small.

### Phase 1 eval

Use a small manual dataset of:
- questions
- expected source documents
- expected supporting passages
- expected answer traits

### Metrics

- `[x]` retrieval hit rate
- `[x]` citation correctness
- `[x]` groundedness
- `[x]` no-answer correctness for negative queries

### Phase 1 minimum acceptance thresholds

- citation correctness: 100% on the fixture eval set
- negative-query behavior: 100% safe no-answer on fixture negatives
- retrieval hit rate: sufficient to support the expected answer on the fixture eval set

Exact percentage targets for retrieval hit rate can be tuned after the fixture set is finalized, but citation correctness and negative-query safety should not be relaxed.

### Phase 2 eval

Add comparison runs:
- baseline retrieval vs rewritten query
- baseline retrieval vs reranked retrieval
- baseline answer vs validated answer

This expands the project beyond the Phase 1 baseline while keeping the evaluation loop concrete and measurable.

Current note:
- the local eval dataset and runner exist now
- the current report covers retrieval hit rate, citation correctness, groundedness, answer-trait matching, and no-answer correctness

---

## 17. Verification Checklist

- [x] recursive folder ingest works
- [x] Phase 1 supports PDF ingestion end to end
- [x] unchanged files are skipped on re-ingest
- [x] retrieved answers include citations
- [x] weak-evidence questions do not produce overconfident answers
- [x] validation tests exist and pass
- [x] unit tests exist and pass
- [x] smoke tests exist and pass
- [x] e2e tests exist and pass
- [x] regression tests exist for every fixed bug fixed so far
- [x] LangGraph is used for workflow orchestration without unnecessary agent complexity
- [x] Phase 1 stays basic while Phase 2 advanced patterns are explicitly captured

No major implementation gaps remain in the current Phase 2 feature set.
Future OCR-engine integration would need additional tooling checks beyond the current sidecar OCR mode.

---

## 18. Implementation Phases

### Phase 1

Build:
- generic ingestion state
- PDF extraction
- chunking
- embedding/indexing
- basic retrieval
- grounded answering
- CLI
- full testing skeleton and first test suites

### Phase 1 exit criteria

- [x] all required test categories exist and pass
- [x] fixture corpus covers text PDF, OCR fallback PDF, multi-page PDF, and negative query cases
- [x] eval dataset exists and is runnable locally
- [x] the basic retrieve-then-read path is stable enough to serve as the baseline for Phase 2 comparisons

Status:
- the Phase 1 baseline is stable enough to use as the comparison point for Phase 2 work
- remaining debt: reranking/query-rewrite/validation work, and the hardcoded PDF/OCR dependency checks

### Phase 2

Add:
- query rewriting
- reranking
- answer validation
- eval CI gating and broader dataset coverage

### Phase 2 exit criteria

- each advanced pattern can be toggled on and off independently
- eval can compare baseline Phase 1 behavior against Phase 2 variants
- regression coverage exists for any new failure modes introduced by rewriting, reranking, or validation

### Phase 3

Possible later additions:
- more file types
- MCP server
- hybrid retrieval
- richer metadata filters

---

## 19. Fact-Checked Assumptions

These assumptions informed this plan revision:

- LangGraph is appropriate as an orchestration layer and does not require LangChain to be usable.
- OpenAI-compatible local endpoints remain a good portability layer for local-first model usage.
- `sqlite-vec` is appropriate for a lightweight local vector index, but should be version-pinned because it is still pre-v1.

These facts were checked during plan revision on 2026-04-28.

### Vector store choice: stay on sqlite-vec (2026-04-29)

LanceDB and Chroma were considered as alternatives. Decision: stay on `sqlite-vec` for the local-first baseline.

Rationale:
- One file, one transaction. Vectors live in the same SQLite database as `source_files`, `documents`, `pages`, `chunks`, and citations. Ingest writes are atomic across relational and vector data; a separate vector store would require a second datastore to keep in sync.
- No extra process or service. Chroma is commonly run as a separate server; LanceDB is embedded but adds a second on-disk format and dependency surface.
- Matches the local-first, single-user shape of the project. There is no multi-tenant or distributed-query requirement that would justify the migration cost.

Tradeoff to revisit:
- `sqlite-vec` KNN is brute-force; latency grows linearly with chunk count. If the corpus crosses roughly 500k chunks or `ask` latency becomes the bottleneck, evaluate LanceDB (columnar, ANN indexes, richer filter pushdown) before committing to a rewrite.
- Hybrid retrieval (FTS5 + dense) and metadata pre-filtering, both listed in §10, are achievable inside SQLite today and are not a reason to migrate.

### Embedding model: evaluate BGE-large vs Nomic (2026-04-29)

Current: `nomic-embed-text-v1.5` via `llama.cpp`, wrapped by `NomicPrefixEmbedder` to prepend `search_query:` / `search_document:`.

Alternative to evaluate: `BAAI/bge-large-en-v1.5` (1024-dim, no prefix scheme). BGE-large generally outperforms Nomic on MTEB retrieval benchmarks; the gap on Huginn's own eval set is unknown until measured.

How to evaluate without committing:
- Stand up a second `llama.cpp` embedding server on a different port serving BGE-large, point a copy of `runtime.yaml` at it, run `ingest --reindex` against a separate `data/huginn-bge.db`, then run `scripts/run_eval.py` against both DBs and compare retrieval hit rate, citation correctness, and groundedness.
- The Nomic prefix wrapper is gated by model name in `huginn.llm.factory.build_runtime_clients`, so swapping models flips the wrapper off automatically — no code change needed for the comparison.

Decision deferred until eval numbers exist. Worth noting that Nomic's 768-dim vs BGE-large's 1024-dim affects sqlite-vec storage size and KNN cost; if the quality delta is small, Nomic stays on cost grounds.
