from huginn.eval.dataset import EvalCase
from huginn.eval.metrics import (
    answer_contains_expected_text,
    citation_correct,
    grounded_answer,
    no_answer_correct,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_hit,
)
from huginn.schemas import QueryAnswer


def build_eval_report(cases: list[EvalCase], answers: list[QueryAnswer]) -> dict[str, float | int]:
    total = len(cases)
    retrieval_cases = [
        (case, answer)
        for case, answer in zip(cases, answers, strict=True)
        if case.expected_citations
    ]
    citation_hits = sum(citation_correct(case, answer) for case, answer in zip(cases, answers, strict=True))
    content_hits = sum(
        answer_contains_expected_text(case, answer)
        for case, answer in zip(cases, answers, strict=True)
    )
    no_answer_hits = sum(no_answer_correct(case, answer) for case, answer in zip(cases, answers, strict=True))
    retrieval_hits = sum(retrieval_hit(case, answer) for case, answer in zip(cases, answers, strict=True))
    grounded_hits = sum(grounded_answer(case, answer) for case, answer in zip(cases, answers, strict=True))
    precision_values = [
        value
        for case, answer in retrieval_cases
        if (value := precision_at_k(case, answer)) is not None
    ]
    recall_values = [
        value
        for case, answer in retrieval_cases
        if (value := recall_at_k(case, answer)) is not None
    ]
    reciprocal_rank_values = [
        value
        for case, answer in retrieval_cases
        if (value := reciprocal_rank(case, answer)) is not None
    ]
    return {
        "total_cases": total,
        "retrieval_cases": len(retrieval_cases),
        "retrieval_hit_rate": retrieval_hits / total if total else 0.0,
        "precision_at_k": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "recall_at_k": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_rank_values) / len(reciprocal_rank_values)
            if reciprocal_rank_values
            else 0.0
        ),
        "citation_correctness": citation_hits / total if total else 0.0,
        "groundedness": grounded_hits / total if total else 0.0,
        "answer_trait_match": content_hits / total if total else 0.0,
        "no_answer_correctness": no_answer_hits / total if total else 0.0,
    }


def build_eval_comparison(
    baseline_name: str,
    baseline_report: dict[str, float | int],
    candidate_name: str,
    candidate_report: dict[str, float | int],
) -> dict[str, object]:
    metric_names = [
        "retrieval_hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "citation_correctness",
        "groundedness",
        "answer_trait_match",
        "no_answer_correctness",
    ]
    metrics: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        baseline_value = float(baseline_report[metric_name])
        candidate_value = float(candidate_report[metric_name])
        metrics[metric_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metrics": metrics,
    }
