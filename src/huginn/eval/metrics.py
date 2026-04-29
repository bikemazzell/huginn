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
