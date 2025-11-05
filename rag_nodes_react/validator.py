# rag_nodes_react/validator.py
from __future__ import annotations
import json, logging
from json import JSONDecodeError
from typing import Dict, Any
from pydantic import ValidationError
from json_repair import repair_json      # :contentReference[oaicite:5]{index=5}
from .models import Action, Finish

log = logging.getLogger("rag.validator")

FUNC_ALIAS = {
    # 针对 group_by_aggregate
    "group_by_aggregate": {
        "value_column": "target_column",
        "return_direct": "target_column",
        "agg_column":    "target_column",
    },
    # 针对 group_top_n
    "group_top_n": {
        "sort_column": "column",
    },
    # covariance / correlation 两列写在 columns 数组
    "calculate_covariance": {"columns->": ("x", "y")},
    "calculate_correlation": {"columns->": ("x", "y")},
    # “Job_Type” 或 “job type” 被用户/LLM写出来时会自动替换成真实列 Operation_Type
    "job_type": "Operation_Type",
}

MAX_RETRY = 3

def _safe_json_loads(text: str) -> dict:
    l, r = text.find("{"), text.rfind("}")
    payload = text[l:r + 1] if (l != -1 and r != -1 and r > l) else text
    try:
        return json.loads(payload)
    except Exception:
        fixed = repair_json(payload)
        return json.loads(fixed)

def _alias(args: dict) -> dict:
    return {FUNC_ALIAS.get(k, k): v for k, v in args.items()}

def _normalize(act: dict) -> dict:
    ALIAS = {
        "column": "target_column",
        "value_column": "target_column",
        "return_column": "target_column",
        "return_direct": "target_column",
        "x": "target_column",
        "y": "other_column",
        "columns": "pair",
        "sort_column": "column",
        # 新增：适配 sort_rows 可能的别名
        "sort_by": "column",
        "by": "column",
        "on": "column",
    }
    a = act.get("args", {})

    # pair 展开
    if a.get("pair") and len(a["pair"]) == 2:
        a["x"], a["y"] = a.pop("pair")

    # 统一重命名
    for k in list(a.keys()):
        if k in ALIAS:
            a[ALIAS[k]] = a.pop(k)

    # 统一 order 写法
    if "order" in a:
        v = str(a["order"]).lower()
        a["order"] = "asc" if v in ("asc", "ascending", "升序") else "desc" if v in ("desc", "descending", "降序") else a["order"]

    # select_rows 的最小必需：保证有 condition 键（有些模型可能给 where/expr）
    if act.get("function") == "select_rows":
        if "condition" not in a:
            if "where" in a: a["condition"] = a.pop("where")
            elif "expr" in a: a["condition"] = a.pop("expr")

    act["args"] = a
    return act

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ---------- 队列未清空，直接回执行 ----------
    if state.get("route") == "finish":
        return state

    if state.get("action_queue"):          # 还有待执行步骤
        state["route"] = "execute"
        return state

    raw = state.pop("llm_output", "")        # ➊ 取出后立即 pop，避免下一轮重复解析
    step = state.get("step", 0) + 1
    state["step"] = step

    if step > MAX_RETRY:
        state.update(route="finish", final_answer="[Error] too many retries")
        log.warning("Exceeded max retry, giving up.")
        return state

    log.debug("Validator step %s | raw (200 chars): %s", step, raw[:200])

    # ---------- 解析 JSON ----------

    try:
        data = _safe_json_loads(raw)
    except Exception as e:
        state.update(route="error", observation=f"[JSON-Error] {e}")
        log.error("JSON decode failed (after repair): %s", e)
        return state

    # -------- 生成 action_queue --------
    try:
        if "actions" in data:                                    # 多步
            acts = [_normalize(a) for a in data["actions"]]
        elif "finish" in data:                                   # 一步回答
            state.update(route="finish", final_answer=data["finish"])
            return state
        else:                                                    # 单 action
            acts = [_normalize(data)]

        state["action_queue"] = [Action.model_validate(a).model_dump()
                                  for a in acts]
        state["route"] = "execute"               # ❷ 永远只发往 execute
        return state
    except ValidationError as e:
        state.update(route="error", observation=f"[Action-Validation] {e}")
        return state
