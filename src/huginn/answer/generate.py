from pathlib import Path
from typing import Protocol

from huginn.schemas import QueryAnswer, RetrievedChunk


class ChatModel(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


def format_citation(chunk: RetrievedChunk) -> str:
    filename = Path(chunk.source_path).name
    if chunk.page_start == chunk.page_end:
        return f"{filename}#page={chunk.page_start}"
    return f"{filename}#pages={chunk.page_start}-{chunk.page_end}"


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    chat_model: ChatModel | None = None,
) -> QueryAnswer:
    if not chunks:
        return QueryAnswer(
            answer_text="I could not find grounded evidence for that question.",
            citations=[],
            evidence_note="No sufficiently relevant chunks were retrieved.",
        )

    top_chunk = chunks[0]
    answer_text = top_chunk.text
    if chat_model is not None:
        answer_text = chat_model.complete(
            system_prompt="Answer only from the supplied context and do not invent facts.",
            user_prompt=(
                f"Question: {question}\n\n"
                f"Context:\n{top_chunk.text}\n\n"
                "Return a concise grounded answer."
            ),
        )
    return QueryAnswer(
        answer_text=answer_text,
        citations=[format_citation(top_chunk)],
        evidence_note=f"Answered from {len(chunks)} retrieved chunk(s) for: {question}",
    )
