# rag_nodes_react/execute.py
from __future__ import annotations
import textwrap, pandas as pd, os
from pathlib import Path
from typing import Dict, Any
from RAG_tools import TOOL_REGISTRY
from logging_setup import get_logger

log = get_logger("rag.execute")

MAX_PREVIEW = 10
SIDE_EFFECT_FUNCS = {"add_derived_column", "graph_export", "plot_machine_avg_bar"}

def _ensure_outdir(state: Dict[str, Any]) -> Path:
    out = Path(state.get("output_dir") or os.getenv("OUTPUT_DIR", "output"))
    out.mkdir(parents=True, exist_ok=True)
    return out

def _save_df(df: pd.DataFrame, outdir: Path, fname: str) -> str:
    out = outdir / fname
    df.to_csv(out, index=False)
    return str(out)

def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cur = state.get("execution_output")
    queue = state.get("action_queue", [])
    outdir = _ensure_outdir(state)
    step_i = int(state.get("qt_step", 0))

    log.info("ENTER execute | pending=%d", len(queue))
    if not queue:
        state.update(route="finish", final_answer=str(cur))
        log.info("No more actions -> finish")
        # 最终若是 DF，确保持久化一次
        if isinstance(cur, pd.DataFrame):
            path = _save_df(cur, outdir, "qt_result_latest.csv")
            state["qt_last_csv"] = path
            log.info("Saved final Q&T result -> %s", path)
        return state

    action = state["action_queue"].pop(0)
    fname = action["function"]; args = action.get("args", {})
    log.info("RUN tool=%s | args=%s | cur_type=%s", fname, args, type(cur).__name__)
    try:
        result = TOOL_REGISTRY[fname].func(cur, args)
        log.info("OK tool=%s | result_type=%s", fname, type(result).__name__)
    except Exception as e:
        state["observation"] = f"[Tool-Error] {fname} {args} -> {e}"
        state["route"] = "error"
        log.exception("Tool raised exception")
        return state

    state["execution_output"] = result
    if isinstance(result, pd.DataFrame):
        preview = textwrap.dedent(result.head(MAX_PREVIEW).to_string(index=False))
        state["final_answer"] = f"[DataFrame] top-{MAX_PREVIEW} rows\n{preview}"
        # —— 按步落地（可通过环境变量开关）——
        if os.getenv("SAVE_INTERMEDIATE", "1") == "1":
            step_i += 1; state["qt_step"] = step_i
            csv_name = f"qt_step_{step_i:02d}_{fname}.csv"
            path = _save_df(result, outdir, csv_name)
            state["qt_last_csv"] = path
            log.info("Saved intermediate DF -> %s", path)
    else:
        state["final_answer"] = str(result)

    # —— 继续 or 结束 ——（有队列就继续执行；否则 finish）
    state["route"] = "execute" if state.get("action_queue") else "finish"
    if state["route"] == "finish":
        log.info("Q&T finished | last_csv=%s", state.get("qt_last_csv"))
    return state
