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
            )
        ],
        chat_model=chat,
    )

    assert answer.answer_text == "The budget is 1200 dollars."
    assert answer.citations == ["atlas.pdf#page=1"]
    assert "Project Atlas budget is 1200 dollars." in chat.prompts[0][1]
