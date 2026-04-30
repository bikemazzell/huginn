from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_run_eval_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"
    spec = spec_from_file_location("run_eval_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_eval_payload_returns_single_report_when_no_variants() -> None:
    module = _load_run_eval_script()
    payload = module.build_eval_payload(
        {
            "baseline": {
                "total_cases": 2,
                "retrieval_cases": 1,
                "retrieval_hit_rate": 1.0,
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "mean_reciprocal_rank": 1.0,
                "citation_correctness": 1.0,
                "groundedness": 1.0,
                "answer_trait_match": 1.0,
                "no_answer_correctness": 1.0,
            }
        }
    )

    assert payload["total_cases"] == 2
    assert "comparisons" not in payload


def test_build_eval_payload_includes_comparisons_for_variants() -> None:
    module = _load_run_eval_script()
    payload = module.build_eval_payload(
        {
            "baseline": {
                "total_cases": 2,
                "retrieval_cases": 1,
                "retrieval_hit_rate": 0.5,
                "precision_at_k": 0.25,
                "recall_at_k": 0.5,
                "mean_reciprocal_rank": 0.5,
                "citation_correctness": 0.5,
                "groundedness": 0.5,
                "answer_trait_match": 0.5,
                "no_answer_correctness": 1.0,
            },
            "rerank": {
                "total_cases": 2,
                "retrieval_cases": 1,
                "retrieval_hit_rate": 1.0,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "mean_reciprocal_rank": 1.0,
                "citation_correctness": 1.0,
                "groundedness": 1.0,
                "answer_trait_match": 0.5,
                "no_answer_correctness": 1.0,
            },
        }
    )

    assert payload["runs"]["baseline"]["total_cases"] == 2
    assert payload["comparisons"][0]["candidate"] == "rerank"


def test_evaluate_regressions_passes_when_no_metric_drops() -> None:
    module = _load_run_eval_script()

    regressions = module.evaluate_regressions(
        [
            {
                "baseline": "baseline",
                "candidate": "rerank",
                "metrics": {
                    "retrieval_hit_rate": {"baseline": 0.5, "candidate": 0.75, "delta": 0.25},
                    "groundedness": {"baseline": 0.5, "candidate": 0.5, "delta": 0.0},
                },
            }
        ]
    )

    assert regressions == []


def test_evaluate_regressions_reports_metric_drops() -> None:
    module = _load_run_eval_script()

    regressions = module.evaluate_regressions(
        [
            {
                "baseline": "baseline",
                "candidate": "rewrite",
                "metrics": {
                    "retrieval_hit_rate": {"baseline": 1.0, "candidate": 0.5, "delta": -0.5},
                    "groundedness": {"baseline": 1.0, "candidate": 0.75, "delta": -0.25},
                },
            }
        ]
    )

    assert regressions == [
        "rewrite regressed retrieval_hit_rate by -0.500",
        "rewrite regressed groundedness by -0.250",
    ]
