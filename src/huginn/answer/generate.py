from functools import lru_cache
from pathlib import Path
from typing import Protocol

from huginn.schemas import QueryAnswer, RetrievedChunk


class ChatModel(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@lru_cache(maxsize=1)
def _answer_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "config" / "prompts" / "answer.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


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
    context_text = "\n\n".join(chunk.text for chunk in chunks)
    citations = [format_citation(chunk) for chunk in chunks]
    answer_text = top_chunk.text
    if chat_model is not None:
        answer_text = chat_model.complete(
            system_prompt=_answer_system_prompt(),
            user_prompt=(
                f"Question: {question}\n\n"
                f"Context:\n{context_text}\n\n"
                "Return a concise grounded answer."
            ),
        )
    return QueryAnswer(
        answer_text=answer_text,
        citations=citations if chat_model is not None else [format_citation(top_chunk)],
        evidence_note=f"Answered from {len(chunks)} retrieved chunk(s) for: {question}",
    )
