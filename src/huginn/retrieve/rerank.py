from huginn.retrieve.basic import score_query_against_text
from huginn.schemas import RetrievedChunk


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    rescored = [
        chunk.model_copy(update={"score": score_query_against_text(question, chunk.text)})
        for chunk in chunks
    ]
    rescored.sort(
        key=lambda chunk: (
            -chunk.score,
            (chunk.page_end - chunk.page_start),
            len(chunk.text.split()),
            chunk.chunk_id,
        )
    )
    return rescored[:limit]
