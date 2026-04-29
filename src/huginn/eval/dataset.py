import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    expected_citations: list[str]
    expected_substrings: list[str]
    expect_no_answer: bool = False


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in data]
