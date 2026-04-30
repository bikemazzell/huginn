import math
import re
from pathlib import Path
from typing import Protocol

from huginn.schemas import RetrievedChunk
from huginn.store.sqlite import SQLiteStore


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = frozenset(
    (Path(__file__).parent / "stopwords.txt").read_text(encoding="utf-8").split()
)
QUERY_TERM_ALIASES = {
    "telefonnummer": ("call",),
    "telefon": ("call", "phone"),
    "nummer": (),
    "email": ("email", "mail"),
    "mail": ("email", "mail"),
    "e": ("email",),
    "courriel": ("email", "mail"),
    "téléphone": ("call", "phone"),
    "telefono": ("call", "phone"),
    "телефон": ("call", "phone"),
}


class Embedder(Protocol):
    def embed_text(
        self, text: str, *, kind: str = "document"
    ) -> list[float] | dict[str, float]: ...

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]] | list[dict[str, float]]: ...


def lexical_features(text: str) -> dict[str, float]:
    return _token_counts(text)


def lexical_query_features(text: str) -> dict[str, float]:
    counts = _token_counts(text)
    expanded = dict(counts)
    for token, value in counts.items():
        for alias in QUERY_TERM_ALIASES.get(token, ()):
            expanded[alias] = max(expanded.get(alias, 0.0), value * 2.0)
    return expanded


def _token_counts(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def score_query_against_text(query: str, text: str) -> float:
    return cosine_similarity(lexical_query_features(query), lexical_features(text))


def retrieve_top_chunks(
    store: SQLiteStore,
    *,
    question: str,
    top_k: int,
    embedder: Embedder | None = None,
    min_lexical_score: float = 0.2,
    max_dense_distance: float = 0.7,
) -> list[RetrievedChunk]:
    question_vector: list[float] | dict[str, float]
    if embedder is None:
        return _retrieve_lexical_chunks(
            store,
            question=question,
            top_k=top_k,
            min_lexical_score=min_lexical_score,
        )

    question_vector = embedder.embed_text(question, kind="query")
    if isinstance(question_vector, dict):
        return _retrieve_lexical_chunks(
            store,
            question=question,
            top_k=top_k,
            min_lexical_score=min_lexical_score,
        )

    candidate_limit = max(top_k * 3, top_k)
    matches = store.query_nearest_chunks(question_vector, limit=candidate_limit)
    dense_chunks = [
        chunk.model_copy(update={"score": -distance})
        for chunk, distance in matches
        if distance <= max_dense_distance
    ]
    lexical_chunks = _retrieve_lexical_chunks(
        store,
        question=question,
        top_k=candidate_limit,
        min_lexical_score=min_lexical_score,
    )
    if dense_chunks and lexical_chunks:
        return _fuse_rankings(dense_chunks, lexical_chunks, top_k=top_k)
    if dense_chunks:
        return dense_chunks[:top_k]
    return lexical_chunks[:top_k]


def _retrieve_lexical_chunks(
    store: SQLiteStore,
    *,
    question: str,
    top_k: int,
    min_lexical_score: float,
) -> list[RetrievedChunk]:
    scored: list[RetrievedChunk] = []
    query_features = lexical_query_features(question)
    query_tokens = set(query_features)
    for chunk, _vector in store.load_chunks():
        chunk_features = lexical_features(chunk.text)
        score = cosine_similarity(query_features, chunk_features)
        token_coverage = _token_coverage(query_tokens, chunk_features)
        if score < min_lexical_score and token_coverage < 1.0:
            continue
        scored.append(chunk.model_copy(update={"score": max(score, token_coverage)}))
    scored.sort(
        key=lambda chunk: (
            -chunk.score,
            (chunk.page_end - chunk.page_start),
            len(chunk.text.split()),
            chunk.chunk_id,
        )
    )
    return scored[:top_k]


def _token_coverage(query_tokens: set[str], chunk_features: dict[str, float]) -> float:
    if not query_tokens:
        return 0.0
    matched = sum(1 for token in query_tokens if token in chunk_features)
    return matched / len(query_tokens)


def _fuse_rankings(
    dense_chunks: list[RetrievedChunk],
    lexical_chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    fused_scores: dict[int, float] = {}
    chunks_by_id: dict[int, RetrievedChunk] = {}
    for ranking in (dense_chunks, lexical_chunks):
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + _rrf_score(rank)
            chunks_by_id[chunk.chunk_id] = chunk
    fused = [
        chunks_by_id[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(
        key=lambda chunk: (
            -chunk.score,
            (chunk.page_end - chunk.page_start),
            len(chunk.text.split()),
            chunk.chunk_id,
        )
    )
    return fused[:top_k]


def _rrf_score(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


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
