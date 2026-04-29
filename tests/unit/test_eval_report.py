from huginn.eval.dataset import EvalCase
from huginn.eval.report import build_eval_report
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
        "retrieval_hit_rate": 1.0,
        "citation_correctness": 1.0,
        "groundedness": 1.0,
        "answer_trait_match": 1.0,
        "no_answer_correctness": 1.0,
    }
