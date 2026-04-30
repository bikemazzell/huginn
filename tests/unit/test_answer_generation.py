from huginn.answer.generate import generate_answer
from huginn.schemas import RetrievedChunk


class DummyChatModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self.response


def test_generate_answer_uses_chat_model_for_grounded_response() -> None:
    chat = DummyChatModel("The budget is 1200 dollars.")

    answer = generate_answer(
        "What is the budget?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Project Atlas budget is 1200 dollars.",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/atlas.pdf",
                page_start=2,
                page_end=2,
                text="The launch date is 2026-05-15.",
                score=0.8,
            )
        ],
        chat_model=chat,
    )

    assert answer.answer_text == "The budget is 1200 dollars."
    assert answer.citations == ["atlas.pdf#page=1", "atlas.pdf#page=2"]
    assert chat.prompts[0][0] == (
        "Answer only from the retrieved document chunks. "
        "Cite the supporting source file and page reference."
    )
    assert "Project Atlas budget is 1200 dollars." in chat.prompts[0][1]
    assert "The launch date is 2026-05-15." in chat.prompts[0][1]


def test_generate_answer_without_chat_model_preserves_top_chunk_fallback() -> None:
    answer = generate_answer(
        "Summarize the findings.",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/atlas.pdf",
                page_start=1,
                page_end=1,
                text="Project Atlas budget is 1200 dollars.",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/atlas.pdf",
                page_start=2,
                page_end=2,
                text="The launch date is 2026-05-15.",
                score=0.8,
            ),
        ],
    )

    assert answer.answer_text == "Project Atlas budget is 1200 dollars."
    assert answer.citations == ["atlas.pdf#page=1"]


def test_generate_answer_deduplicates_identical_citations_for_chat_answers() -> None:
    chat = DummyChatModel("Vincent Valentine appears in a disclosed API user listing.")

    answer = generate_answer(
        "Who is Valentine?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/apis.pdf",
                page_start=82,
                page_end=82,
                text='Example response includes {"name":"Vincent Valentine"}.',
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/apis.pdf",
                page_start=82,
                page_end=82,
                text='Attackers can use the disclosed Vincent Valentine slug.',
                score=0.8,
            ),
        ],
        chat_model=chat,
    )

    assert answer.answer_text == "Vincent Valentine appears in a disclosed API user listing."
    assert answer.citations == ["apis.pdf#page=82"]


def test_generate_answer_filters_chat_citations_to_chunks_supporting_email_answer() -> None:
    chat = DummyChatModel("The contact email is michelthomas-enquiries@hodder.co.uk.")

    answer = generate_answer(
        "What is the email for Michel Thomas?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/german-course.pdf",
                page_start=50,
                page_end=50,
                text=(
                    "For general enquiries and for information on Michel Thomas: "
                    "Email: michelthomas-enquiries@hodder.co.uk"
                ),
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/cyber.pdf",
                page_start=328,
                page_end=329,
                text="Contact the security team for further guidance on incident handling.",
                score=0.8,
            ),
        ],
        chat_model=chat,
    )

    assert answer.answer_text == "The contact email is michelthomas-enquiries@hodder.co.uk."
    assert answer.citations == ["german-course.pdf#page=50"]


def test_generate_answer_prefers_narrower_overlapping_chat_citation() -> None:
    chat = DummyChatModel("The phone number is 020 7873 6400.")

    answer = generate_answer(
        "What is the phone number?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/german-course.pdf",
                page_start=50,
                page_end=50,
                text="Call: 020 7873 6400",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/german-course.pdf",
                page_start=49,
                page_end=50,
                text="Earlier text. Call: 020 7873 6400",
                score=0.8,
            ),
        ],
        chat_model=chat,
    )

    assert answer.citations == ["german-course.pdf#page=50"]


def test_generate_answer_prefers_overlapping_chunk_with_visible_page_label() -> None:
    chat = DummyChatModel("The phone number is 020 7873 6400.")

    answer = generate_answer(
        "What is the phone number?",
        [
            RetrievedChunk(
                chunk_id=1,
                source_path="/tmp/german-course.pdf",
                page_start=49,
                page_end=50,
                text="MT FOUNDATION COURSE GERMAN Page 47 Call: 020 7873 6400",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                source_path="/tmp/german-course.pdf",
                page_start=50,
                page_end=50,
                text="Call: 020 7873 6400",
                score=0.8,
            ),
        ],
        chat_model=chat,
    )

    assert answer.citations == ["german-course.pdf#page=47"]
