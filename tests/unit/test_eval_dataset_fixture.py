from pathlib import Path

from huginn.eval.dataset import load_eval_dataset


def test_default_eval_dataset_covers_positive_negative_and_ocr_cases() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "dataset.json"

    cases = load_eval_dataset(dataset_path)
    questions = {case.question for case in cases}

    assert len(cases) >= 4
    assert "What is the Project Atlas budget?" in questions
    assert "What is the employee vacation policy?" in questions
    assert "Which vendor is mentioned in the scanned contract?" in questions
    assert "What is the launch date?" in questions
