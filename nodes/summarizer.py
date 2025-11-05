# nodes/summarizer.py
from __future__ import annotations
import os, json, textwrap
import pandas as pd
import httpx
from logging_setup import get_logger

log = get_logger("rag.summarizer")

MAX_PREVIEW = 10
MAX_PROMPT_CHARS = 12000

def _df_preview(df: pd.DataFrame, n: int = MAX_PREVIEW) -> str:
    try:
        return textwrap.dedent(df.head(n).to_string(index=False))
    except Exception:
        return "(preview failed)"

def _clip(txt: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if txt is None:
        return ""
    s = str(txt)
    return s if len(s) <= limit else (s[:limit] + f"\n…[truncated to {limit} chars]")

def _build_prompt(state: dict) -> str:
    # 这里增加对 user_input / processed_input 的兼容
    user_req = (
        state.get("user_input")
        or state.get("processed_input")
        or state.get("user_request")
        or state.get("raw_input")
        or ""
    )
    df = state.get("execution_output") if isinstance(state.get("execution_output"), pd.DataFrame) else None
    artifacts = state.get("qt_artifacts", [])
    last_csv = state.get("qt_last_csv")
    non_df = state.get("non_df_results", [])

    parts = [
        "You are a concise data task summarizer.",
        "Summarize the run results in Chinese first, then give an English recap.",
        "If files were produced, list them clearly.",
        "Do not invent results. Use only what is given.",
        "",
        f"User request:\n{_clip(user_req, 2000)}",
    ]

    if df is not None:
        parts += [
            "",
            "Last DataFrame preview (top rows):",
            _df_preview(df, MAX_PREVIEW),
        ]

    if non_df:
        parts += ["", f"Non-DF scalar results: {non_df}"]

    produced = []
    if last_csv:
        produced.append(f"CSV: {last_csv}")
    for a in artifacts or []:
        produced.append(str(a))
    # 同时把流程里常见的键也自动并入
    for k in ("excel_path", "pr_curve", "f1_curve", "roc_curve"):
        if state.get(k):
            produced.append(f"{k}: {state[k]}")

    if produced:
        parts += ["", "Artifacts created:", "\n".join(f"- {p}" for p in produced)]

    return "\n".join(parts)

def _call_llm(prompt: str) -> str:
    """
    调用 DashScope 兼容的 /chat/completions。
    关键点：
    - 不使用 response_format/json schema（此前 400 多出在这里）
    - 只发送纯文本 messages
    """
    model = os.getenv("SUMMARIZER_MODEL", os.getenv("LLM_MODEL", "qwen-plus"))
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/OPENAI_API_KEY")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        # 注意：不要传 response_format、tools、function_call 等，以避免 400
    }

    log.info("summarizer.call | model=%s | url=%s", model, url)
    r = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    # 若失败，抛出以便触发本层的兜底
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Bad response schema: {e}; raw={json.dumps(data)[:500]}")

def summarizer(state: dict) -> dict:
    """
    总结节点：
    - 优先尝试 LLM；
    - 出错则使用可读的本地兜底。
    """
    try:
        prompt = _build_prompt(state)
        log.info("prompt_chars=%d", len(prompt))
        text = _call_llm(_clip(prompt, MAX_PROMPT_CHARS))
        state["final_answer"] = text
        state["route"] = "finish"
        return state
    except Exception as e:
        log.exception("summarizer LLM failed, using fallback")
        df = state.get("execution_output") if isinstance(state.get("execution_output"), pd.DataFrame) else None
        preview = _df_preview(df) if df is not None else "(no DataFrame)"
        arts = []
        if state.get("qt_last_csv"):
            arts.append(state["qt_last_csv"])
        for a in state.get("qt_artifacts", []) or []:
            arts.append(str(a))
        for k in ("excel_path", "pr_curve", "f1_curve", "roc_curve"):
            if state.get(k):
                arts.append(f"{k}: {state[k]}")

        # 统一读取用户请求
        user_req = (
                state.get("user_input")
                or state.get("processed_input")
                or state.get("user_request")
                or state.get("raw_input")
                or ""
        )

        fallback = [
            "【Bilingual Summary Fallback】",
            f"(Reason: {type(e).__name__}: {e})",
            "",
            f"User request:\n{_clip(user_req, 2000)}",
            "",
            "Last Output Preview:",
             preview,
             "",
             f"[Artifacts] {arts if arts else '(none)'}"
        ]
        state["final_answer"] = "\n".join(fallback)
        state["route"] = "finish"
        return state
