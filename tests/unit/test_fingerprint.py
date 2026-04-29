from pathlib import Path

from huginn.ingest.fingerprint import sha256_file


def test_sha256_file_is_stable_for_same_contents(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"hello")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64
