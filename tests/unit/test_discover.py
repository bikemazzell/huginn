from pathlib import Path

from huginn.ingest.discover import discover_supported_files


def test_discover_supported_files_finds_pdfs_recursively_and_sorts_results(
    tmp_path: Path,
) -> None:
    top_pdf = tmp_path / "a.pdf"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_pdf = nested_dir / "b.PDF"
    ignored_md = nested_dir / "note.md"

    top_pdf.write_text("pdf", encoding="utf-8")
    nested_pdf.write_text("pdf", encoding="utf-8")
    ignored_md.write_text("note", encoding="utf-8")

    results = discover_supported_files(tmp_path)

    assert results == [top_pdf, nested_pdf]


def test_discover_supported_files_skips_hidden_directories(tmp_path: Path) -> None:
    visible_pdf = tmp_path / "visible.pdf"
    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    hidden_pdf = hidden_dir / "hidden.pdf"

    visible_pdf.write_text("pdf", encoding="utf-8")
    hidden_pdf.write_text("pdf", encoding="utf-8")

    results = discover_supported_files(tmp_path)

    assert results == [visible_pdf]
