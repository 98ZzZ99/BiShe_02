# RAG_subgraph_qt_react.py
from langgraph.graph import StateGraph, END
from rag_nodes_react.thought    import thought_node
from rag_nodes_react.validator  import validator_node
from rag_nodes_react.execute    import execute_node
import logging
log = logging.getLogger("rag.qt")

def _validate_switch(state: dict) -> str:
    r = state.get("route")
    if r == "execute": return "execute"  # 继续跑队列
    if r == "finish":  return END
    log.info("validator -> thought (replan)")
    return "thought"   # 其他情况回到思考

def _after_execute(state: dict) -> str:
    r = state.get("route")
    if r == "execute": return "validate"  # 队列里还有 action → 直接走 validator 的“有队列就回 execute”逻辑
    if r == "error":   return "thought"   # 工具报错 → 回 thought 让 LLM 观察/重规划
    log.info("execute -> END")
    return END                            # finish

def build_qt_react_subgraph():
    sg = StateGraph(dict)
    sg.add_node("thought", thought_node)
    sg.add_node("validate", validator_node)
    sg.add_node("execute",  execute_node)

    sg.add_edge("thought", "validate")
    sg.add_conditional_edges("validate", _validate_switch)
    sg.add_conditional_edges("execute",  _after_execute)

    sg.set_entry_point("thought")
    return sg.compile()
