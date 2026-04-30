from functools import lru_cache
from pathlib import Path
from typing import Protocol

from huginn.schemas import QueryAnswer, RetrievedChunk


class ValidationModel(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@lru_cache(maxsize=1)
def _validation_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "config" / "prompts" / "validate_answer.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def validate_answer(
    question: str,
    answer: QueryAnswer,
    chunks: list[RetrievedChunk],
    *,
    chat_model: ValidationModel | None = None,
) -> QueryAnswer:
    if chat_model is None or not chunks or _is_safe_no_answer(answer):
        return answer

    context_text = "\n\n".join(chunk.text for chunk in chunks)
    verdict = chat_model.complete(
        system_prompt=_validation_system_prompt(),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Answer: {answer.answer_text}\n\n"
            f"Context:\n{context_text}\n\n"
            "Return SUPPORTED if the answer is grounded in the context. "
            "Return UNSUPPORTED if it is not."
        ),
    ).strip().upper()
    if verdict.startswith("SUPPORTED"):
        return answer
    if verdict.startswith("UNSUPPORTED"):
        return QueryAnswer(
            answer_text="I could not find grounded evidence for that question.",
            citations=[],
            evidence_note="Answer validation rejected the generated answer.",
        )
    return answer


def _is_safe_no_answer(answer: QueryAnswer) -> bool:
    return (
        answer.answer_text == "I could not find grounded evidence for that question."
        and answer.citations == []
    )
