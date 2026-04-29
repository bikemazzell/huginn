from pathlib import Path

import pytest

from huginn.extract.registry import get_extractor_for_path


def test_get_extractor_for_path_rejects_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# hi", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        get_extractor_for_path(path)
