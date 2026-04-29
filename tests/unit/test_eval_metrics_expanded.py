from huginn.eval.dataset import EvalCase
from huginn.eval.report import build_eval_report
from huginn.schemas import QueryAnswer


def test_build_eval_report_includes_retrieval_and_groundedness_metrics() -> None:
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

    assert report["retrieval_hit_rate"] == 1.0
    assert report["groundedness"] == 1.0
    assert report["precision_at_k"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0
