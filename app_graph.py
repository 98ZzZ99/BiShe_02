# app_graph.py
from typing import TypedDict, Any, Dict
from langgraph.graph import StateGraph, END
from logging_setup import get_logger
log = get_logger("graph")

from nodes.preprocess import PreprocessNode
from nodes.intent_router import IntentRouterNode
from rag_kb.knowledge_base import KBNode
from RAG_subgraph_qt_react import build_qt_react_subgraph
from RAG_subgraph_anomaly import build_anomaly_subgraph
# ↓ Change this line to import a function and give it an alias to avoid it having the same name as a local variable.
from nodes.summarizer import summarizer as summarizer_node

class AppState(TypedDict, total=False):
    # input / preprocess
    user_input: str
    processed_input: str
    csv_path: str | None
    columns: list[str]       # Full set of column names in the current dataset
    # intent
    need_qt: bool
    need_anomaly: bool
    algorithms: list[str] | None
    # rag
    kb_snippets: list[str]
    # Q&T / anomaly outputs
    execution_output: Any
    bench_summary: Any
    picked_algo: str
    excel_path: str
    # final
    final_answer: str

    qt_last_csv: str
    pr_curve: str
    f1_curve: str
    roc_curve: str

# instantiate nodes
_pre  = PreprocessNode()
_intent = IntentRouterNode()
_kb   = KBNode()

def _pre_node(state: Dict[str, Any]) -> Dict[str, Any]:
    log.info("ENTER node=pre | raw_len=%d", len(state["user_input"]))
    state.update(_pre.run(state["user_input"]))
    log.info("EXIT  node=pre | csv_path=%s | cols=%d",
            state.get("csv_path"), len(state.get("columns", [])))
    return state

def _intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state.update(_intent.run(state["processed_input"], state.get("csv_path")))
    log.info("node=intent | need_qt=%s need_anomaly=%s algos=%s csv=%s",
            state.get("need_qt"), state.get("need_anomaly"),
            state.get("algorithms"), state.get("csv_path"))
    return state

def _kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Build/refresh KB and add snippets related to the current query to state["kb_snippets"].
    state.update(_kb.run(
        query=state["processed_input"],
        csv_path=state.get("csv_path"),
        columns=state.get("columns", []),
    ))
    log.info("node=kb | snippets=%d", len(state.get("kb_snippets", [])))
    return state

def after_kb(state: Dict[str, Any]) -> str:
    if state.get("need_qt", False):
        log.info("route: kb -> qt")
        return "qt"
    if state.get("need_anomaly", False):
        log.info("route: kb -> anomaly")
        return "anomaly"
    log.info("route: kb -> summarizer")
    return "summarizer"

def after_qt(state: Dict[str, Any]) -> str:
    if state.get("need_anomaly", False):
        log.info("route: qt -> anomaly")
        return "anomaly"
    log.info("route: qt -> summarizer")
    return "summarizer"

def _summ_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # `summarizer` is a function, not an object.
    return summarizer_node(state)

def build_app_graph():
    sg = StateGraph(AppState)

    sg.add_node("pre", _pre_node)
    sg.add_node("intent", _intent_node)
    sg.add_node("kb", _kb_node)
    sg.add_node("qt",  build_qt_react_subgraph())     # Q&T subgraph directly used as nodes
    sg.add_node("anomaly", build_anomaly_subgraph())
    sg.add_node("summarizer", _summ_node)

    sg.add_edge("pre", "intent")
    sg.add_edge("intent", "kb")
    sg.add_conditional_edges("kb", after_kb)
    sg.add_conditional_edges("qt", after_qt)
    sg.add_edge("anomaly", "summarizer")

    sg.add_edge("summarizer", END)
    sg.set_entry_point("pre")
    return sg.compile()
