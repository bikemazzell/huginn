from huginn.eval.dataset import EvalCase
from huginn.schemas import QueryAnswer


def citation_correct(case: EvalCase, answer: QueryAnswer) -> bool:
    return answer.citations == case.expected_citations


def no_answer_correct(case: EvalCase, answer: QueryAnswer) -> bool:
    return (answer.answer_text == "I could not find grounded evidence for that question.") == case.expect_no_answer


def answer_contains_expected_text(case: EvalCase, answer: QueryAnswer) -> bool:
    return all(substring in answer.answer_text for substring in case.expected_substrings)


def retrieval_hit(case: EvalCase, answer: QueryAnswer) -> bool:
    if not case.expected_citations:
        return answer.citations == []
    return any(citation in case.expected_citations for citation in answer.citations)


def grounded_answer(case: EvalCase, answer: QueryAnswer) -> bool:
    if case.expect_no_answer:
        return no_answer_correct(case, answer)
    return citation_correct(case, answer) and answer_contains_expected_text(case, answer)


def precision_at_k(case: EvalCase, answer: QueryAnswer) -> float | None:
    if not case.expected_citations:
        return None
    if not answer.citations:
        return 0.0
    relevant = sum(citation in case.expected_citations for citation in answer.citations)
    return relevant / len(answer.citations)


def recall_at_k(case: EvalCase, answer: QueryAnswer) -> float | None:
    if not case.expected_citations:
        return None
    if not answer.citations:
        return 0.0
    matched = {citation for citation in answer.citations if citation in case.expected_citations}
    return len(matched) / len(case.expected_citations)


def reciprocal_rank(case: EvalCase, answer: QueryAnswer) -> float | None:
    if not case.expected_citations:
        return None
    for index, citation in enumerate(answer.citations, start=1):
        if citation in case.expected_citations:
            return 1.0 / index
    return 0.0
