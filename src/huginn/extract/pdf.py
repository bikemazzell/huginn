from pathlib import Path

from pypdf import PdfReader

from huginn.schemas import ExtractedDocument, ExtractedPage


class PdfExtractor:
    def __init__(self, ocr_fallback: bool) -> None:
        self.ocr_fallback = ocr_fallback

    def extract(self, path: str | Path) -> ExtractedDocument:
        pdf_path = Path(path)
        reader = PdfReader(str(pdf_path))
        pages: list[ExtractedPage] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text and self.ocr_fallback:
                text = self._read_ocr_sidecar(pdf_path, page_number)
            pages.append(ExtractedPage(page_number=page_number, text=text))

        return ExtractedDocument(
            source_path=str(pdf_path),
            title=pdf_path.stem,
            pages=pages,
        )

    def _read_ocr_sidecar(self, pdf_path: Path, page_number: int) -> str:
        sidecar = pdf_path.with_suffix(".ocr.txt")
        if not sidecar.exists():
            return ""

        contents = sidecar.read_text(encoding="utf-8").strip()
        if "\f" not in contents:
            return contents

        page_texts = [part.strip() for part in contents.split("\f")]
        if 0 < page_number <= len(page_texts):
            return page_texts[page_number - 1]
        return ""
