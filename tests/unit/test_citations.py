from huginn.answer.generate import format_citation
from huginn.schemas import RetrievedChunk


def test_format_citation_uses_single_page_when_range_collapses() -> None:
    chunk = RetrievedChunk(
        chunk_id=1,
        source_path="/tmp/sample.pdf",
        page_start=2,
        page_end=2,
        text="evidence",
        score=0.9,
    )

    assert format_citation(chunk) == "sample.pdf#page=2"


def test_format_citation_uses_page_range_for_multi_page_chunk() -> None:
    chunk = RetrievedChunk(
        chunk_id=1,
        source_path="/tmp/sample.pdf",
        page_start=2,
        page_end=4,
        text="evidence",
        score=0.9,
    )

    assert format_citation(chunk) == "sample.pdf#pages=2-4"


def test_format_citation_prefers_visible_page_label_when_present() -> None:
    chunk = RetrievedChunk(
        chunk_id=1,
        source_path="/tmp/sample.pdf",
        page_start=49,
        page_end=50,
        text="Appendix header MT FOUNDATION COURSE Page 47 Contact details follow.",
        score=0.9,
    )

    assert format_citation(chunk) == "sample.pdf#page=47"
