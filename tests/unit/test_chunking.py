from huginn.chunking.split import chunk_document
from huginn.schemas import ExtractedDocument, ExtractedPage


def test_chunk_document_creates_overlapping_chunks() -> None:
    document = ExtractedDocument(
        source_path="ignored.pdf",
        title="ignored",
        pages=[
            ExtractedPage(page_number=1, text="one two three four five six"),
            ExtractedPage(page_number=2, text="seven eight nine ten eleven twelve"),
        ],
    )

    chunks = chunk_document(document, chunk_size=5, chunk_overlap=2)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[0].text == "one two three four five"
    assert chunks[1].text == "four five six seven eight"
    assert chunks[1].page_start == 1
    assert chunks[1].page_end == 2


def test_chunk_document_rejects_empty_page_text() -> None:
    document = ExtractedDocument(
        source_path="ignored.pdf",
        title="ignored",
        pages=[ExtractedPage(page_number=1, text="   ")],
    )

    chunks = chunk_document(document, chunk_size=5, chunk_overlap=1)

    assert chunks == []
