# nodes/summarizer.py
from __future__ import annotations
from typing import Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()
_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NGC_API_KEY"))

_SYS = (
    "You are a data analytics assistant. Produce a concise, structured summary in **both Chinese and English**.\n"
    "Sections:\n"
    "1) 中文总结（不超过150字）\n"
    "2) English Summary (<=150 words)\n"
    "If Q&T produced a DataFrame, briefly describe what was computed/filtered/sorted. "
    "If anomaly detection ran, include the picked model and top suspicious timestamps. "
    "Be faithful to the provided context; do not invent numbers."
    "When you describe sorting, do not write possessive phrases like X's Y. Say sorted by <column> instead. "
)

def _df_preview(obj: Any, n=8) -> str:
    if isinstance(obj, pd.DataFrame):
        return obj.head(n).to_string(index=False)
    return str(obj)

class SummarizerNode:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        parts = []
        parts.append(f"User request: {state.get('processed_input')}")
        if "execution_output" in state and state["execution_output"] is not None:
            parts.append("Q&T / Last Output Preview:\n" + _df_preview(state["execution_output"]))
        if state.get("bench_summary") is not None:
            parts.append(f"Anomaly best: {state.get('picked_algo')} | Excel: {state.get('excel_path')}")
        prompt = "\n\n".join(parts)

        try:
            resp = _client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role":"system","content":_SYS},{"role":"user","content":prompt}],
                temperature=0.2, max_tokens=450,
            )
            state["final_answer"] = resp.choices[0].message.content.strip()
        except Exception:
            state["final_answer"] = "【Bilingual Summary Fallback】\n" + prompt
        return state
