from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.helpers import write_pdf


def main() -> int:
    corpus = ROOT / "tests" / "fixtures" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    write_pdf(corpus / "atlas.pdf", ["Project Atlas budget is 1200 dollars."])
    write_pdf(corpus / "scan.pdf", [""])
    (corpus / "scan.ocr.txt").write_text(
        "Scanned contract mentions Orion vendor.",
        encoding="utf-8",
    )
    write_pdf(
        corpus / "minutes.pdf",
        ["Meeting intro and background.", "The launch date is 2026-05-15."],
    )
    write_pdf(corpus / "negative.pdf", ["General notes about office logistics."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
