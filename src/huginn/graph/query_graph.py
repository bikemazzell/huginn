from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from huginn.answer.generate import ChatModel, generate_answer
from huginn.config import RuntimeConfig
from huginn.retrieve.basic import Embedder, retrieve_top_chunks
from huginn.schemas import QueryAnswer, RetrievedChunk
from huginn.store.sqlite import SQLiteStore


class QueryState(TypedDict):
    config: RuntimeConfig
    db_path: str
    question: str
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
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {
            "config": config,
            "db_path": str(db_path),
            "question": question,
            "embedder": embedder,
            "chat_model": chat_model,
            "chunks": [],
            "answer": QueryAnswer(answer_text="", citations=[]),
        }
    )
    return result["answer"]


def _retrieve(state: QueryState) -> QueryState:
    store = SQLiteStore(state["db_path"])
    try:
        chunks = retrieve_top_chunks(
            store,
            question=state["question"],
            top_k=state["config"].indexing.top_k,
            embedder=state["embedder"],
            min_lexical_score=state["config"].indexing.min_lexical_score,
            max_dense_distance=state["config"].indexing.max_dense_distance,
        )
    finally:
        store.close()
    return {**state, "chunks": chunks}


def _answer(state: QueryState) -> QueryState:
    answer = generate_answer(
        state["question"],
        state["chunks"],
        chat_model=state["chat_model"],
    )
    return {**state, "answer": answer}
