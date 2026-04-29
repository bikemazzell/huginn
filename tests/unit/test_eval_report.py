import pytest

from huginn.eval.dataset import EvalCase
from huginn.eval.report import build_eval_comparison, build_eval_report
from huginn.schemas import QueryAnswer


def test_build_eval_report_computes_expected_phase1_metrics() -> None:
    cases = [
        EvalCase(
            question="What is the budget?",
            expected_citations=["atlas.pdf#page=1"],
            expected_substrings=["1200 dollars"],
        ),
        EvalCase(
            question="What is the vacation policy?",
            expected_citations=[],
            expected_substrings=["I could not find grounded evidence"],
            expect_no_answer=True,
        ),
    ]
    answers = [
        QueryAnswer(answer_text="Project Atlas budget is 1200 dollars.", citations=["atlas.pdf#page=1"]),
        QueryAnswer(
            answer_text="I could not find grounded evidence for that question.",
            citations=[],
            evidence_note="No sufficiently relevant chunks were retrieved.",
        ),
    ]

    report = build_eval_report(cases, answers)

    assert report == {
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


def test_build_eval_report_computes_ranking_metrics_from_citation_order() -> None:
    cases = [
        EvalCase(
            question="What is the budget?",
            expected_citations=["atlas.pdf#page=1", "atlas.pdf#page=2"],
            expected_substrings=["1200 dollars"],
        ),
        EvalCase(
            question="What is the launch date?",
            expected_citations=["minutes.pdf#page=2"],
            expected_substrings=["2026-05-15"],
        ),
    ]
    answers = [
        QueryAnswer(
            answer_text="Budget answer",
            citations=["wrong.pdf#page=1", "atlas.pdf#page=2", "atlas.pdf#page=1"],
        ),
        QueryAnswer(
            answer_text="Launch answer",
            citations=["minutes.pdf#page=2", "appendix.pdf#page=1"],
        ),
    ]

    report = build_eval_report(cases, answers)

    assert report["retrieval_cases"] == 2
    assert report["precision_at_k"] == pytest.approx(7 / 12)
    assert report["recall_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75


def test_build_eval_comparison_reports_metric_deltas() -> None:
    baseline = {
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
    }
    candidate = {
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
    }

    comparison = build_eval_comparison("baseline", baseline, "rerank", candidate)

    assert comparison["baseline"] == "baseline"
    assert comparison["candidate"] == "rerank"
    assert comparison["metrics"]["precision_at_k"] == {
        "baseline": 0.25,
        "candidate": 0.5,
        "delta": 0.25,
    }
    assert comparison["metrics"]["answer_trait_match"]["delta"] == 0.0
