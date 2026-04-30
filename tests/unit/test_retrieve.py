from huginn.retrieve.basic import lexical_features, retrieve_top_chunks, score_query_against_text
from huginn.schemas import RetrievedChunk


class FakeSparseStore:
    def __init__(self, rows: list[tuple[RetrievedChunk, dict[str, float]]]) -> None:
        self.rows = rows

    def load_chunks(self) -> list[tuple[RetrievedChunk, dict[str, float]]]:
        return self.rows


class FakeDenseStore:
    def __init__(
        self,
        rows: list[tuple[RetrievedChunk, float]],
        text_rows: list[tuple[RetrievedChunk, list[float] | dict[str, float]]] | None = None,
    ) -> None:
        self.rows = rows
        self.text_rows = text_rows or []

    def query_nearest_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int,
    ) -> list[tuple[RetrievedChunk, float]]:
        return self.rows[:limit]

    def load_chunks(self) -> list[tuple[RetrievedChunk, list[float] | dict[str, float]]]:
        return self.text_rows


class FakeDenseEmbedder:
    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        return [1.0, 0.0]


def test_score_query_against_text_prefers_more_overlapping_terms() -> None:
    strong = score_query_against_text("project atlas budget", "atlas project budget budget")
    weak = score_query_against_text("project atlas budget", "atlas notes appendix")

    assert strong > weak
    assert strong > 0


def test_score_query_against_text_returns_zero_for_no_overlap() -> None:
    assert score_query_against_text("atlas budget", "completely unrelated words") == 0.0


def test_retrieve_top_chunks_filters_sparse_matches_below_min_score() -> None:
    strong_text = "atlas budget budget"
    weak_text = "atlas notes appendix"
    strong_score = score_query_against_text("atlas budget", strong_text)
    weak_score = score_query_against_text("atlas budget", weak_text)
    threshold = (strong_score + weak_score) / 2
    store = FakeSparseStore(
        [
            (
                RetrievedChunk(
                    chunk_id=1,
                    source_path="/tmp/atlas.pdf",
                    page_start=1,
                    page_end=1,
                    text=strong_text,
                    score=0.0,
                ),
                lexical_features(strong_text),
            ),
            (
                RetrievedChunk(
                    chunk_id=2,
                    source_path="/tmp/notes.pdf",
                    page_start=1,
                    page_end=1,
                    text=weak_text,
                    score=0.0,
                ),
                lexical_features(weak_text),
            ),
        ]
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="atlas budget",
        top_k=4,
        min_lexical_score=threshold,
    )

    assert [chunk.chunk_id for chunk in chunks] == [1]
    assert chunks[0].score == 1.0


def test_retrieve_top_chunks_filters_dense_matches_above_max_distance() -> None:
    store = FakeDenseStore(
        [
            (
                RetrievedChunk(
                    chunk_id=1,
                    source_path="/tmp/atlas.pdf",
                    page_start=1,
                    page_end=1,
                    text="Atlas budget",
                    score=0.0,
                ),
                0.2,
            ),
            (
                RetrievedChunk(
                    chunk_id=2,
                    source_path="/tmp/notes.pdf",
                    page_start=1,
                    page_end=1,
                    text="Atlas appendix",
                    score=0.0,
                ),
                0.9,
            ),
        ]
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="atlas budget",
        top_k=4,
        embedder=FakeDenseEmbedder(),
        max_dense_distance=0.5,
    )

    assert [chunk.chunk_id for chunk in chunks] == [1]
    assert chunks[0].score == -0.2


def test_retrieve_top_chunks_uses_text_lexical_matching_without_sparse_vectors() -> None:
    store = FakeSparseStore(
        [
            (
                RetrievedChunk(
                    chunk_id=1,
                    source_path="/tmp/atlas.pdf",
                    page_start=1,
                    page_end=1,
                    text="Atlas budget is 1200 dollars.",
                    score=0.0,
                ),
                [0.1, 0.2],
            ),
            (
                RetrievedChunk(
                    chunk_id=2,
                    source_path="/tmp/notes.pdf",
                    page_start=1,
                    page_end=1,
                    text="Completely unrelated appendix notes.",
                    score=0.0,
                ),
                [0.3, 0.4],
            ),
        ]
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="atlas budget",
        top_k=4,
        min_lexical_score=0.2,
    )

    assert [chunk.chunk_id for chunk in chunks] == [1]
    assert chunks[0].score > 0


def test_retrieve_top_chunks_falls_back_to_lexical_when_dense_returns_nothing() -> None:
    matching_chunk = RetrievedChunk(
        chunk_id=2,
        source_path="/tmp/apis.pdf",
        page_start=82,
        page_end=82,
        text=(
            'It might return this data: [{"id":2,"name":"Vincent Valentine", '
            '"slug":"Vincent"}]'
        ),
        score=0.0,
    )
    store = FakeDenseStore(
        [],
        text_rows=[
            (
                RetrievedChunk(
                    chunk_id=1,
                    source_path="/tmp/notes.pdf",
                    page_start=1,
                    page_end=1,
                    text="General appendix notes about users.",
                    score=0.0,
                ),
                [0.1, 0.2],
            ),
            (matching_chunk, [0.3, 0.4]),
        ],
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="valentine",
        top_k=4,
        embedder=FakeDenseEmbedder(),
        min_lexical_score=0.2,
        max_dense_distance=0.5,
    )

    assert [chunk.chunk_id for chunk in chunks] == [2]
    assert "Vincent Valentine" in chunks[0].text


def test_retrieve_top_chunks_does_not_accept_partial_name_overlap_on_lexical_fallback() -> None:
    matching_chunk = RetrievedChunk(
        chunk_id=2,
        source_path="/tmp/apis.pdf",
        page_start=82,
        page_end=82,
        text=(
            'It might return this data: [{"id":2,"name":"Vincent Valentine", '
            '"slug":"Vincent"}]'
        ),
        score=0.0,
    )
    store = FakeDenseStore(
        [],
        text_rows=[(matching_chunk, [0.3, 0.4])],
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="basil valentine",
        top_k=4,
        embedder=FakeDenseEmbedder(),
        min_lexical_score=0.2,
        max_dense_distance=0.5,
    )

    assert chunks == []


def test_retrieve_top_chunks_fuses_dense_and_lexical_rankings() -> None:
    exact_match_chunk = RetrievedChunk(
        chunk_id=2,
        source_path="/tmp/apis.pdf",
        page_start=82,
        page_end=82,
        text=(
            'It might return this data: [{"id":2,"name":"Vincent Valentine", '
            '"slug":"Vincent"}]'
        ),
        score=0.0,
    )
    loosely_related_chunk = RetrievedChunk(
        chunk_id=1,
        source_path="/tmp/essay.pdf",
        page_start=1,
        page_end=1,
        text="A general discussion of romance, names, and unrelated characters.",
        score=0.0,
    )
    store = FakeDenseStore(
        [
            (loosely_related_chunk, 0.15),
            (exact_match_chunk, 0.45),
        ],
        text_rows=[
            (loosely_related_chunk, [0.1, 0.2]),
            (exact_match_chunk, [0.3, 0.4]),
        ],
    )

    chunks = retrieve_top_chunks(
        store,  # type: ignore[arg-type]
        question="vincent valentine",
        top_k=1,
        embedder=FakeDenseEmbedder(),
        min_lexical_score=0.2,
        max_dense_distance=0.5,
    )

    assert [chunk.chunk_id for chunk in chunks] == [2]
    assert "Vincent Valentine" in chunks[0].text
