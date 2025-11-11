# RAG_subgraph_qt_react.py
from langgraph.graph import StateGraph, END
from rag_nodes_react.thought    import thought_node
from rag_nodes_react.validator  import validator_node
from rag_nodes_react.execute    import execute_node
import logging
log = logging.getLogger("rag.qt")

def _validate_switch(state: dict) -> str:
    r = state.get("route")
    if r == "execute": return "execute"  # Continue running the queue
    if r == "finish":  return END
    log.info("validator -> thought (replan)")
    return "thought"   # Other situations require further consideration.

def _after_execute(state: dict) -> str:
    r = state.get("route")
    if r == "execute": return "validate"  # There are still actions in the queue → directly execute the validator's "execute if there is a queue" logic.
    if r == "error":   return "thought"   # Tool error → Return to thought to let LLM observe/replan
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
