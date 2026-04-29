from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from huginn.answer.generate import ChatModel
from huginn.config import RuntimeConfig
from huginn.eval.dataset import EvalCase
from huginn.eval.report import build_eval_report
from huginn.graph.query_graph import run_query
from huginn.retrieve.basic import Embedder
from huginn.schemas import QueryAnswer


class EvalState(TypedDict):
    config: RuntimeConfig
    db_path: str
    cases: list[EvalCase]
    embedder: Embedder | None
    chat_model: ChatModel | None
    answers: list[QueryAnswer]
    report: dict[str, float | int]


def run_eval(
    config: RuntimeConfig,
    *,
    db_path: str | Path,
    cases: list[EvalCase],
    embedder: Embedder | None = None,
    chat_model: ChatModel | None = None,
) -> dict[str, float | int]:
    graph = StateGraph(EvalState)
    graph.add_node("run_cases", _run_cases)
    graph.add_node("report", _report)
    graph.set_entry_point("run_cases")
    graph.add_edge("run_cases", "report")
    graph.add_edge("report", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {
            "config": config,
            "db_path": str(db_path),
            "cases": cases,
            "embedder": embedder,
            "chat_model": chat_model,
            "answers": [],
            "report": {},
        }
    )
    return result["report"]


def _run_cases(state: EvalState) -> EvalState:
    answers = [
        run_query(
            state["config"],
            db_path=state["db_path"],
            question=case.question,
            embedder=state["embedder"],
            chat_model=state["chat_model"],
        )
        for case in state["cases"]
    ]
    return {**state, "answers": answers}


def _report(state: EvalState) -> EvalState:
    return {**state, "report": build_eval_report(state["cases"], state["answers"])}
