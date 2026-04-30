from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huginn.config import load_runtime_config
from huginn.eval.dataset import load_eval_dataset
from huginn.eval.report import build_eval_comparison
from huginn.graph.eval_graph import run_eval
from huginn.llm.factory import build_runtime_clients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Huginn evals for one or more configs.")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        help="Path to a runtime config. Repeat to compare multiple runs. First config is baseline.",
    )
    parser.add_argument(
        "--db-path",
        default=str(ROOT / "data" / "huginn.db"),
        help="Path to the SQLite database to query.",
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "tests" / "fixtures" / "eval" / "dataset.json"),
        help="Path to the eval dataset JSON file.",
    )
    return parser.parse_args()


def build_eval_payload(reports: dict[str, dict[str, float | int]]) -> dict[str, object]:
    report_names = list(reports)
    baseline_name = report_names[0]
    comparisons = [
        build_eval_comparison(baseline_name, reports[baseline_name], candidate_name, reports[candidate_name])
        for candidate_name in report_names[1:]
    ]
    if comparisons:
        return {
            "runs": reports,
            "comparisons": comparisons,
        }
    return reports[baseline_name]


def evaluate_regressions(comparisons: list[dict[str, object]]) -> list[str]:
    regressions: list[str] = []
    for comparison in comparisons:
        candidate = str(comparison["candidate"])
        metrics = comparison["metrics"]
        assert isinstance(metrics, dict)
        for metric_name, metric_values in metrics.items():
            assert isinstance(metric_values, dict)
            delta = float(metric_values["delta"])
            if delta < 0:
                regressions.append(f"{candidate} regressed {metric_name} by {delta:.3f}")
    return regressions


def main() -> int:
    args = parse_args()
    config_paths = [Path(path) for path in (args.configs or [str(ROOT / "config" / "runtime.yaml")])]
    cases = load_eval_dataset(args.dataset)
    reports: dict[str, dict[str, float | int]] = {}

    for config_path in config_paths:
        config = load_runtime_config(config_path)
        clients = build_runtime_clients(config.models)
        reports[config_path.stem] = run_eval(
            config,
            db_path=args.db_path,
            cases=cases,
            embedder=clients.embedder,
            chat_model=clients.chat,
        )

    payload = build_eval_payload(reports)
    comparisons = payload.get("comparisons", []) if isinstance(payload, dict) else []
    regressions = evaluate_regressions(comparisons) if isinstance(comparisons, list) else []
    if regressions and isinstance(payload, dict):
        payload["regressions"] = regressions
    print(json.dumps(payload, indent=2))
    if regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
