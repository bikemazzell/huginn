from huginn.answer.validate import validate_answer
from huginn.schemas import QueryAnswer, RetrievedChunk


class DummyValidationChatModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self.response


def test_validate_answer_preserves_supported_answer() -> None:
    chat = DummyValidationChatModel("SUPPORTED")
    answer = QueryAnswer(
        answer_text="Project Atlas budget is 1200 dollars.",
        citations=["atlas.pdf#page=1"],
        evidence_note="Answered from 1 retrieved chunk(s) for: What is the budget?",
    )
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            source_path="/tmp/atlas.pdf",
            page_start=1,
            page_end=1,
            text="Project Atlas budget is 1200 dollars.",
            score=0.9,
        )
    ]

    validated = validate_answer(
        "What is the budget?",
        answer,
        chunks,
        chat_model=chat,
    )

    assert validated == answer
    assert "Answer: Project Atlas budget is 1200 dollars." in chat.prompts[0][1]


def test_validate_answer_replaces_unsupported_answer_with_safe_no_answer() -> None:
    chat = DummyValidationChatModel("UNSUPPORTED")
    answer = QueryAnswer(
        answer_text="Project Atlas budget is 5000 dollars.",
        citations=["atlas.pdf#page=1"],
        evidence_note="Answered from 1 retrieved chunk(s) for: What is the budget?",
    )
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            source_path="/tmp/atlas.pdf",
            page_start=1,
            page_end=1,
            text="Project Atlas budget is 1200 dollars.",
            score=0.9,
        )
    ]

    validated = validate_answer(
        "What is the budget?",
        answer,
        chunks,
        chat_model=chat,
    )

    assert validated.answer_text == "I could not find grounded evidence for that question."
    assert validated.citations == []
    assert validated.evidence_note == "Answer validation rejected the generated answer."


def test_validate_answer_skips_validation_for_existing_no_answer() -> None:
    chat = DummyValidationChatModel("UNSUPPORTED")
    answer = QueryAnswer(
        answer_text="I could not find grounded evidence for that question.",
        citations=[],
        evidence_note="No sufficiently relevant chunks were retrieved.",
    )

    validated = validate_answer(
        "What is the budget?",
        answer,
        [],
        chat_model=chat,
    )

    assert validated == answer
    assert chat.prompts == []
