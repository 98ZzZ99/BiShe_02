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

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NGC_API_KEY"),
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
    log.info("thought | prompt_chars=%d | cols=%d | kb_snippets=%d",
             len(prompt), len(state.get("columns", [])), len(state.get("kb_snippets", [])))
    resp = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type":"json_object"},
        stop=["```"],
    )
    state["llm_output"] = resp.choices[0].message.content.strip()
    log.info("thought | llm_json_len=%d", len(state["llm_output"]))
    return state
