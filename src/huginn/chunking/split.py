from huginn.schemas import ChunkRecord, ExtractedDocument


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    words: list[tuple[str, int]] = []
    for page in document.pages:
        for word in page.text.split():
            words.append((word, page.page_number))

    if not words:
        return []

    step = chunk_size - chunk_overlap
    starts = set(range(0, len(words), step))
    word_offset = 0
    for page in document.pages:
        if word_offset > 0:
            starts.add(word_offset)
        word_offset += len(page.text.split())

    chunks: list[ChunkRecord] = []
    for chunk_index, start in enumerate(sorted(starts)):
        window = words[start : start + chunk_size]
        if not window:
            continue
        chunk_words = [word for word, _ in window]
        page_numbers = [page_number for _, page_number in window]
        chunks.append(
            ChunkRecord(
                chunk_index=chunk_index,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                text=" ".join(chunk_words),
                token_count=len(chunk_words),
            )
        )

    return chunks
