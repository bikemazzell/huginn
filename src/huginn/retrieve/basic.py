import math
import re
from pathlib import Path
from typing import Protocol

from huginn.schemas import RetrievedChunk
from huginn.store.sqlite import SQLiteStore


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
    (Path(__file__).parent / "stopwords.txt").read_text(encoding="utf-8").split()
)


class Embedder(Protocol):
    def embed_text(
        self, text: str, *, kind: str = "document"
    ) -> list[float] | dict[str, float]: ...

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]] | list[dict[str, float]]: ...


def lexical_features(text: str) -> dict[str, float]:
    return _token_counts(text)


def _token_counts(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def score_query_against_text(query: str, text: str) -> float:
    return cosine_similarity(lexical_features(query), lexical_features(text))


def retrieve_top_chunks(
    store: SQLiteStore,
    *,
    question: str,
    top_k: int,
    embedder: Embedder | None = None,
) -> list[RetrievedChunk]:
    question_vector: list[float] | dict[str, float]
    if embedder is None:
        question_vector = lexical_features(question)
    else:
        question_vector = embedder.embed_text(question, kind="query")

    if isinstance(question_vector, list):
        matches = store.query_nearest_chunks(question_vector, limit=top_k)
        return [
            chunk.model_copy(update={"score": -distance})
            for chunk, distance in matches
        ]

    scored: list[RetrievedChunk] = []

    for chunk, vector in store.load_chunks():
        score = cosine_similarity(question_vector, vector)
        if score <= 0:
            continue
        scored.append(chunk.model_copy(update={"score": score}))

    scored.sort(
        key=lambda chunk: (
            -chunk.score,
            (chunk.page_end - chunk.page_start),
            len(chunk.text.split()),
            chunk.chunk_id,
        )
    )
    return scored[:top_k]


def cosine_similarity(
    left: dict[str, float] | list[float],
    right: dict[str, float] | list[float],
) -> float:
    if isinstance(left, list) and isinstance(right, list):
        return _cosine_similarity_dense(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return _cosine_similarity_sparse(left, right)
    return 0.0


def _cosine_similarity_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right.get(token, 0.0) for token in left)
    if numerator == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_similarity_dense(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    if numerator == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
