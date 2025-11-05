# rag_nodes_react/thought.py
from __future__ import annotations
from typing import Dict, Any
from string import Template
import os, logging
from dotenv import load_dotenv
from openai import OpenAI
from RAG_tools import TOOL_REGISTRY

log = logging.getLogger("rag.thought")
load_dotenv()

# ---------- 可配置项（环境变量） ----------
# 北京区（默认）：https://dashscope.aliyuncs.com/compatible-mode/v1
# 国际区（新加坡）：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY")  # 必填：百炼 API Key
THOUGHT_MODEL      = os.getenv("THOUGHT_MODEL", "qwen-plus")  # 模型名列表见官方文档
LLM_TIMEOUT        = float(os.getenv("LLM_TIMEOUT", "30"))    # 秒
LLM_MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "1024"))
ENABLE_THINKING    = os.getenv("ENABLE_THINKING", "").lower() in ("1","true","yes","y")
# 只有 Qwen3 思考类模型才需要；qwen-plus 通常不必传。见官方说明需通过 extra_body 传递。
# 参考：Enable thinking via extra_body on OpenAI-compatible interface.
# https://www.alibabacloud.com/help/en/model-studio/deep-thinking

client = OpenAI(
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    timeout=LLM_TIMEOUT,
    max_retries=0,  # 失败立即抛出，由下方兜底处理
)

def _tool_spec() -> str:
    return "\n".join(
        f"### {name}\n{tool.description}\nRequired keys: {', '.join(tool.signature)}"
        for name, tool in TOOL_REGISTRY.items()
    )

_PROMPT_T = Template("""
You translate the user's request into RAW JSON tool calls (ReAct style over tabular ops).
If the question restricts rows, ALWAYS start with select_rows.

Use the retrieved KB context to disambiguate column names and required args.

Retrieved KB Context:
$kb

Table Columns:
$cols

Tool Specs:
$tool_spec

JSON schema you MUST follow for each step:
{"function": "<tool_name>", "args": {...}}

If you have the final answer, output {"finish":"<answer>"}.

Scratchpad:
${scratchpad}
User: ${user}

Return either ONE action object, or {"actions":[...]} for multi-step, or {"finish": "..."}.
""".lstrip())

def thought_node(state: Dict[str, Any]) -> Dict[str, Any]:
    kb_txt = "\n".join(state.get("kb_snippets", [])) or "(no kb)"
    cols = ", ".join(state.get("columns", [])) or "(unknown)"
    prompt = _PROMPT_T.substitute(
        kb=kb_txt,
        cols=cols,
        tool_spec=_tool_spec(),
        scratchpad=state.get("scratchpad", ""),
        user=state["processed_input"],
    )

    log.info(
        "thought | prompt_chars=%d | cols=%d | kb_snippets=%d | model=%s | base_url=%s",
        len(prompt), len(state.get("columns", [])),
        len(state.get("kb_snippets", [])), THOUGHT_MODEL, DASHSCOPE_BASE_URL
    )

    # resp = client.chat.completions.create(
    #     # model="meta/llama-3.1-8b-instruct",
    #     model=os.getenv("THOUGHT_MODEL", "qwen/qwen3-next-80b-a3b-thinking"),  # ← 这里切换模型
    #     messages=[{"role":"user","content":prompt}],
    #     temperature=0.0,
    #     max_tokens=1024,
    #     response_format={"type":"json_object"},
    #     stop=["```"],
    # )
    # state["llm_output"] = resp.choices[0].message.content.strip()
    # log.info("thought | llm_json_len=%d", len(state["llm_output"]))
    # return state

    try:
        # 组装 extra_body（仅在需要思考模式时传）
        extra_body = None
        if ENABLE_THINKING:
            # 注意：enable_thinking 不是 OpenAI 标准参数，需走 extra_body
            # 某些模型/版本不支持该参数，请按需开启
            extra_body = {"enable_thinking": True}

        resp = client.chat.completions.create(
            model=THOUGHT_MODEL,
            messages=[
                # 给一条系统提示进一步强调“只返回 JSON”
                {"role": "system", "content": "Return ONLY a single valid JSON object per request."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},  # 百炼 JSON 模式
            stop=["```"],
            **({"extra_body": extra_body} if extra_body else {}),
        )

        content = (resp.choices[0].message.content or "").strip()
        # 守护：确保 validator 接到的是 JSON
        if not content.startswith("{"):
            content = '{"finish":"(LLM returned non-JSON)"}'

        state["llm_output"] = content
        log.info("thought | llm_json_len=%d", len(state["llm_output"]))
    except Exception as e:
        # 兜底：不让子图卡住
        msg = f"[LLM-Error:{type(e).__name__}] {e}"
        log.warning("thought.llm failed -> fallback finish | %s", msg)
        state["llm_output"] = '{"finish":"LLM unavailable; please retry later."}'

    return state
