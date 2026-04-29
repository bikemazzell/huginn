# Local `llama.cpp` Setup

This project currently assumes a two-endpoint local setup:

- chat model on `127.0.0.1:1234`
- embedding model on `127.0.0.1:1235`

The default runtime config in [config/runtime.yaml] is aligned to that layout.

## Chat Model

Example `Qwen3.5-9B` server:

```bash
llama-server \
  -m /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --host 127.0.0.1 \
  --port 1234 \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  -ngl 99 \
  -c 8192
```

## Embedding Model

Current default embedding model:

```text
/path/to/nomic-embed-text-v2-moe.Q4_K_M.gguf
```

Example embedding server:

```bash
llama-server \
  -m /path/to/nomic-embed-text-v2-moe.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 1235 \
  --embeddings \
  --pooling cls \
  -ngl 99 \
  -c 512
```

Use whatever local model paths match your own machine and keep [config/runtime.yaml](../config/runtime.yaml) aligned with the model names and ports you actually serve.

## Nomic Prefixing

The runtime automatically prefixes embedding inputs for Nomic:

- queries use `search_query: `
- documents use `search_document: `

Do not add those prefixes manually when using the CLI.

## Rebuild Index

If you change embedding models or embedding endpoints, rebuild the local index:

```bash
python -m huginn.cli --config config/runtime.yaml --db-path data/huginn.db ingest --reindex tests/fixtures/corpus
```

## Smoke Test

```bash
python -m huginn.cli --config config/runtime.yaml --db-path data/huginn.db ask "What is the launch date?"
```
