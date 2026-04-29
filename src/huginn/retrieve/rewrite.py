from functools import lru_cache
from pathlib import Path
from typing import Protocol


class RewriteModel(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@lru_cache(maxsize=1)
def _rewrite_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "config" / "prompts" / "rewrite_query.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def rewrite_query(question: str, *, chat_model: RewriteModel | None) -> str:
    if chat_model is None:
        return question
    rewritten = chat_model.complete(
        system_prompt=_rewrite_system_prompt(),
        user_prompt=question,
    ).strip()
    if not rewritten:
        return question
    return " ".join(rewritten.splitlines()).strip()
