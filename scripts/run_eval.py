from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huginn.config import load_runtime_config
from huginn.eval.dataset import load_eval_dataset
from huginn.graph.eval_graph import run_eval
from huginn.llm.factory import build_runtime_clients


def main() -> int:
    config = load_runtime_config(ROOT / "config" / "runtime.yaml")
    cases = load_eval_dataset(ROOT / "tests" / "fixtures" / "eval" / "dataset.json")
    clients = build_runtime_clients(config.models)
    report = run_eval(
        config,
        db_path=ROOT / "data" / "huginn.db",
        cases=cases,
        embedder=clients.embedder,
        chat_model=clients.chat,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
