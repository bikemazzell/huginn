from huginn.retrieve.rerank import rerank_chunks
from huginn.schemas import RetrievedChunk


def test_rerank_chunks_prefers_stronger_lexical_overlap() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            source_path="/tmp/atlas.pdf",
            page_start=1,
            page_end=1,
            text="Atlas appendix and planning backlog.",
            score=-0.1,
        ),
        RetrievedChunk(
            chunk_id=2,
            source_path="/tmp/atlas.pdf",
            page_start=2,
            page_end=2,
            text="Atlas budget is 1200 dollars and owned by project Atlas.",
            score=-0.3,
        ),
    ]

    reranked = rerank_chunks("What is the Atlas budget?", chunks, limit=2)

    assert [chunk.chunk_id for chunk in reranked] == [2, 1]
    assert reranked[0].score > reranked[1].score


def test_rerank_chunks_truncates_to_limit() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=index,
            source_path=f"/tmp/doc-{index}.pdf",
            page_start=1,
            page_end=1,
            text=f"Budget text {index}",
            score=float(-index),
        )
        for index in range(1, 5)
    ]

    reranked = rerank_chunks("budget", chunks, limit=2)

    assert len(reranked) == 2
