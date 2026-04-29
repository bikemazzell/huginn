from pathlib import Path

from huginn.extract.pdf import PdfExtractor
from tests.helpers import write_pdf


def test_pdf_extractor_reads_text_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    write_pdf(pdf_path, ["Alpha receipt", "Beta appendix"])

    result = PdfExtractor(ocr_fallback=False).extract(pdf_path)

    assert result.title == "sample"
    assert result.page_count == 2
    assert [page.text.strip() for page in result.pages] == [
        "Alpha receipt",
        "Beta appendix",
    ]


def test_pdf_extractor_uses_sidecar_fallback_for_empty_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    write_pdf(pdf_path, [""])
    (tmp_path / "scan.ocr.txt").write_text("Recovered scanned text", encoding="utf-8")

    result = PdfExtractor(ocr_fallback=True).extract(pdf_path)

    assert result.page_count == 1
    assert result.pages[0].text == "Recovered scanned text"
