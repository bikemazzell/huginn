import argparse
import json
from pathlib import Path

from huginn.config import load_runtime_config
from huginn.graph.ingest_graph import run_ingest
from huginn.graph.query_graph import run_query
from huginn.llm.factory import build_runtime_clients
from huginn.store.sqlite import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="huginn")
    parser.add_argument("--config", default="config/runtime.yaml")
    parser.add_argument("--db-path", default="data/huginn.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--reindex", action="store_true")
    ingest_parser.add_argument("folder")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")

    subparsers.add_parser("status")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    db_path = Path(args.db_path)
    runtime_clients = build_runtime_clients(config.models)

    if args.command == "ingest":
        ingest_config = config.model_copy(update={"root_path": Path(args.folder)})
        result = run_ingest(
            ingest_config,
            db_path=db_path,
            embedder=runtime_clients.embedder,
            reindex=args.reindex,
        )
        print(result.model_dump_json(indent=2))
        return 0

    if args.command == "ask":
        answer = run_query(
            config,
            db_path=db_path,
            question=args.question,
            embedder=runtime_clients.embedder,
            chat_model=runtime_clients.chat,
        )
        print(answer.model_dump_json(indent=2))
        return 0

    store = SQLiteStore(db_path)
    try:
        print(json.dumps(store.status_counts(), indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
