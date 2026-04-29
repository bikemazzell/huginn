from pathlib import Path

from huginn.extract.pdf import PdfExtractor


def get_extractor_for_path(path: str | Path, *, ocr_fallback: bool = False) -> PdfExtractor:
    file_path = Path(path)
    if file_path.suffix.lower() == ".pdf":
        return PdfExtractor(ocr_fallback=ocr_fallback)
    raise ValueError(f"Unsupported file type: {file_path.suffix or '<none>'}")
