from huginn.eval.dataset import EvalCase
from huginn.eval.metrics import (
    answer_contains_expected_text,
    citation_correct,
    grounded_answer,
    no_answer_correct,
    retrieval_hit,
)
from huginn.schemas import QueryAnswer


def build_eval_report(cases: list[EvalCase], answers: list[QueryAnswer]) -> dict[str, float | int]:
    total = len(cases)
    citation_hits = sum(citation_correct(case, answer) for case, answer in zip(cases, answers, strict=True))
    content_hits = sum(
        answer_contains_expected_text(case, answer)
        for case, answer in zip(cases, answers, strict=True)
    )
    no_answer_hits = sum(no_answer_correct(case, answer) for case, answer in zip(cases, answers, strict=True))
    retrieval_hits = sum(retrieval_hit(case, answer) for case, answer in zip(cases, answers, strict=True))
    grounded_hits = sum(grounded_answer(case, answer) for case, answer in zip(cases, answers, strict=True))
    return {
        "total_cases": total,
        "retrieval_hit_rate": retrieval_hits / total if total else 0.0,
        "citation_correctness": citation_hits / total if total else 0.0,
        "groundedness": grounded_hits / total if total else 0.0,
        "answer_trait_match": content_hits / total if total else 0.0,
        "no_answer_correctness": no_answer_hits / total if total else 0.0,
    }
