from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DATASET = [
    {
        "question": "What is the Project Atlas budget?",
        "expected_citations": ["atlas.pdf#page=1"],
        "expected_substrings": ["1200 dollars"],
        "expect_no_answer": False,
    },
    {
        "question": "What is the employee vacation policy?",
        "expected_citations": [],
        "expected_substrings": ["I could not find grounded evidence"],
        "expect_no_answer": True,
    },
]


def main() -> int:
    target = ROOT / "tests" / "fixtures" / "eval" / "dataset.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(DEFAULT_DATASET, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
