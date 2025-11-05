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

def _save_df(df, outdir, csv_name):
    import os, time
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, csv_name)
    try:
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out
    except PermissionError:
        base, ext = os.path.splitext(csv_name)
        alt = f"{base}_{int(time.time())}{ext}"
        out2 = os.path.join(outdir, alt)
        df.to_csv(out2, index=False, encoding="utf-8-sig")
        return out2

def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cur = state.get("execution_output")
    queue = state.get("action_queue", [])
    outdir = _ensure_outdir(state)
    step_i = int(state.get("qt_step", 0))

    # —— 若游标为空且有 csv_path，则自动载入一次 DataFrame 作为起始游标 ——
    if cur is None and state.get("csv_path"):
        try:
            path = str(state["csv_path"])
            if path.lower().endswith(".csv"):
                cur = pd.read_csv(path)
            elif path.lower().endswith((".xlsx", ".xls")):
                cur = pd.read_excel(path)
            else:
                cur = None  # 目录等情况不自动加载
            if isinstance(cur, pd.DataFrame):
                state["execution_output"] = cur
                log.info("Primed cursor from csv_path -> DataFrame shape=%s", cur.shape)
        except Exception as e:
            log.warning("Failed to prime cursor from csv_path: %s", e)

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
    fname = action["function"]
    args = action.get("args", {})
    log.info("RUN tool=%s | args=%s | cur_type=%s", fname, args, type(cur).__name__)
    try:
        result = TOOL_REGISTRY[fname].func(cur, args)
        log.info("OK tool=%s | result_type=%s", fname, type(result).__name__)
    except Exception as e:
        state["observation"] = f"[Tool-Error] {fname} {args} -> {e}"
        state["route"] = "error"
        log.exception("Tool raised exception")
        return state

    # —— 只在结果为 DF/Series 时更新游标；否则保持游标不变 ——
    updated_cursor = False

    # Series 统一转成 DataFrame，便于预览/落盘
    if isinstance(result, pd.Series):
        result = result.to_frame()
        log.info("Coerced Series to DataFrame for preview/persist")

    if isinstance(result, pd.DataFrame):
        # 更新游标
        state["execution_output"] = result
        cur = result
        updated_cursor = True

        # 预览 + 中间结果落盘
        preview = textwrap.dedent(result.head(MAX_PREVIEW).to_string(index=False))
        state["final_answer"] = f"[DataFrame] top-{MAX_PREVIEW} rows\n{preview}"

        if os.getenv("SAVE_INTERMEDIATE", "1") == "1":
            step_i += 1
            state["qt_step"] = step_i
            csv_name = f"qt_step_{step_i:02d}_{fname}.csv"
            path = _save_df(result, outdir, csv_name)
            state["qt_last_csv"] = path
            log.info("Saved intermediate DF -> %s", path)
    else:
        # 非 DF 结果：记录但不改变游标，避免污染后续步骤（例如相关系数为 float）
        side = state.setdefault("non_df_results", [])
        side.append({fname: result})
        log.debug("Non-DF result from %s: %r (cursor unchanged)", fname, result)

        # 若是有副作用的函数返回了文件路径，收集到 artifacts
        if fname in SIDE_EFFECT_FUNCS and isinstance(result, str):
            arts = state.setdefault("qt_artifacts", [])
            arts.append(result)

        # 提供一个简短可读的输出
        state["final_answer"] = f"[{fname}] {result}"

    # —— 继续 or 结束 ——
    state["route"] = "execute" if state.get("action_queue") else "finish"
    if state["route"] == "finish":
        if isinstance(cur, pd.DataFrame):
            log.info("Q&T finished | last_csv=%s", state.get("qt_last_csv"))
        else:
            log.info("Q&T finished | (no DataFrame cursor) artifacts=%s", state.get("qt_artifacts"))
    return state
