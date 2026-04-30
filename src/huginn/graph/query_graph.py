from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from huginn.answer.generate import ChatModel, generate_answer
from huginn.answer.validate import validate_answer
from huginn.config import RuntimeConfig
from huginn.retrieve.basic import Embedder, retrieve_top_chunks
from huginn.retrieve.rewrite import rewrite_query
from huginn.retrieve.rerank import rerank_chunks
from huginn.schemas import QueryAnswer, RetrievedChunk
from huginn.store.sqlite import SQLiteStore


class QueryState(TypedDict):
    config: RuntimeConfig
    db_path: str
    question: str
    retrieval_question: str
    embedder: Embedder | None
    chat_model: ChatModel | None
    chunks: list[RetrievedChunk]
    answer: QueryAnswer


def run_query(
    config: RuntimeConfig,
    *,
    db_path: str | Path,
    question: str,
    embedder: Embedder | None = None,
    chat_model: ChatModel | None = None,
) -> QueryAnswer:
    graph = StateGraph(QueryState)
    graph.add_node("rewrite", _rewrite)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("answer", _answer)
    graph.add_node("validate", _validate)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {
            "config": config,
            "db_path": str(db_path),
            "question": question,
            "retrieval_question": question,
            "embedder": embedder,
            "chat_model": chat_model,
            "chunks": [],
            "answer": QueryAnswer(answer_text="", citations=[]),
        }
    )
    return result["answer"]


def _rewrite(state: QueryState) -> QueryState:
    if not state["config"].features.query_rewrite:
        return {**state, "retrieval_question": state["question"]}
    rewritten = rewrite_query(
        state["question"],
        chat_model=state["chat_model"],
    )
    return {**state, "retrieval_question": rewritten}


def _retrieve(state: QueryState) -> QueryState:
    store = SQLiteStore(state["db_path"])
    try:
        chunks = retrieve_top_chunks(
            store=store,
            question=state["retrieval_question"],
            top_k=_retrieve_limit(state["config"]),
            embedder=state["embedder"],
            min_lexical_score=state["config"].indexing.min_lexical_score,
            max_dense_distance=state["config"].indexing.max_dense_distance,
        )
    finally:
        store.close()
    return {**state, "chunks": chunks}


def _rerank(state: QueryState) -> QueryState:
    if not state["config"].features.rerank:
        return {**state, "chunks": state["chunks"][: state["config"].indexing.top_k]}
    chunks = rerank_chunks(
        state["question"],
        state["chunks"],
        limit=state["config"].indexing.top_k,
    )
    return {**state, "chunks": chunks}


def _retrieve_limit(config: RuntimeConfig) -> int:
    if not config.features.rerank:
        return config.indexing.top_k
    return max(config.indexing.top_k * 3, config.indexing.top_k)


def _answer(state: QueryState) -> QueryState:
    answer = generate_answer(
        state["question"],
        state["chunks"],
        chat_model=state["chat_model"],
    )
    return {**state, "answer": answer}


def _validate(state: QueryState) -> QueryState:
    if not state["config"].features.answer_validation:
        return state
    answer = validate_answer(
        state["question"],
        state["answer"],
        state["chunks"],
        chat_model=state["chat_model"],
    )
    return {**state, "answer": answer}
