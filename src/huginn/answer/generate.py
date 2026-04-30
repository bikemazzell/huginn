from functools import lru_cache
from pathlib import Path
import re
from typing import Protocol

from huginn.schemas import QueryAnswer, RetrievedChunk


class ChatModel(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://\S+\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")


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
    citations = (
        _deduplicated_citations(_supporting_chunks(answer_text, chunks))
        if chat_model is not None
        else [format_citation(top_chunk)]
    )
    return QueryAnswer(
        answer_text=answer_text,
        citations=citations if chat_model is not None else [format_citation(top_chunk)],
        evidence_note=f"Answered from {len(chunks)} retrieved chunk(s) for: {question}",
    )


def _deduplicated_citations(chunks: list[RetrievedChunk]) -> list[str]:
    chunks = _collapse_overlapping_chunks(chunks)
    seen: set[str] = set()
    citations: list[str] = []
    for chunk in chunks:
        citation = format_citation(chunk)
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


def _collapse_overlapping_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        replaced = False
        for index, existing in enumerate(kept):
            if Path(existing.source_path).name != Path(chunk.source_path).name:
                continue
            if not _page_ranges_overlap(existing, chunk):
                continue
            if _prefer_chunk_over_existing(chunk, existing):
                kept[index] = chunk
            replaced = True
            break
        if not replaced:
            kept.append(chunk)
    return kept


def _page_ranges_overlap(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    return not (left.page_end < right.page_start or right.page_end < left.page_start)


def _page_span(chunk: RetrievedChunk) -> int:
    return chunk.page_end - chunk.page_start


def _prefer_chunk_over_existing(chunk: RetrievedChunk, existing: RetrievedChunk) -> bool:
    return _page_span(chunk) < _page_span(existing)


def _supporting_chunks(answer_text: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    anchors = _answer_anchors(answer_text)
    if anchors:
        supported = [chunk for chunk in chunks if _chunk_matches_anchors(chunk.text, anchors)]
        if supported:
            return supported

    answer_terms = set(_normalized_terms(answer_text))
    if answer_terms:
        supported = [chunk for chunk in chunks if answer_terms & set(_normalized_terms(chunk.text))]
        if supported:
            return supported

    return chunks


def _answer_anchors(answer_text: str) -> set[str]:
    anchors = {match.lower() for match in EMAIL_RE.findall(answer_text)}
    anchors.update(match.lower() for match in URL_RE.findall(answer_text))
    anchors.update(_normalize_phone(match) for match in PHONE_RE.findall(answer_text))
    anchors.discard("")
    return anchors


def _chunk_matches_anchors(text: str, anchors: set[str]) -> bool:
    lowered = text.lower()
    chunk_phones = {_normalize_phone(match) for match in PHONE_RE.findall(text)}
    return any(
        anchor in lowered or anchor in chunk_phones
        for anchor in anchors
    )


def _normalize_phone(text: str) -> str:
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if len(digits) >= 7 else ""


def _normalized_terms(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
