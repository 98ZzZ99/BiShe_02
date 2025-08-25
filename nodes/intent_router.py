# nodes/intent_router.py
from __future__ import annotations
import os, json, re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any
import logging
log = logging.getLogger("node.intent")

load_dotenv()
INTENT_MODE = os.getenv("INTENT_MODE", "llm").lower()  # "llm" | "rule"

_SYS = (
    "You are an intent classifier. Return ONLY one JSON object with keys:\n"
    '{"need_qt": <bool>, "need_anomaly": <bool>, "csv_path": "<opt>", "algorithms": ["EIF","AE",...](optional)}.\n'
    "- If user asks selection/sort/group/aggregate/compute → need_qt=true.\n"
    "- If asks anomaly/outlier/EIF/AE/LOF/INNE/OCSVM → need_anomaly=true.\n"
    "- If both → both true. If neither → both false.\n"
    "Extract csv path if present in the text; else leave empty."
)

class IntentRouterNode:
    def __init__(self) -> None:
        api = os.getenv("NGC_API_KEY")
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api) if INTENT_MODE=="llm" and api else None

    def _rule(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        need_qt = any(k in t for k in ["select", "sort", "group", "avg", "median", "sum", "correlation", "rolling", "top"])
        need_anomaly = any(k in t for k in ["anomaly", "outlier", "eif", "lof", "inne", "ocsvm", "autoencoder", "ae"])
        algos = [a for a in ["EIF","AE","LOF","COPOD","INNE","OCSVM"] if a.lower() in t]
        m = re.search(r"\b([A-Za-z0-9_\./\\-]+\.csv)\b", text)
        return {"need_qt": need_qt, "need_anomaly": need_anomaly, "csv_path": m.group(1) if m else "", "algorithms": algos or None}

    def run(self, processed_input: str, csv_path: str | None) -> Dict[str, Any]:
        log.info("intent.run | mode=%s", INTENT_MODE)
        if not self.client:
            out = self._rule(processed_input)
        else:
            resp = self.client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role":"system","content":_SYS},{"role":"user","content":processed_input}],
                temperature=0.0,
                max_tokens=120,
                response_format={"type":"json_object"},
            )
            out = json.loads(resp.choices[0].message.content)
        # 兜底 csv_path
        if csv_path and not out.get("csv_path"):
            out["csv_path"] = csv_path
        log.info("intent.out | need_qt=%s need_anomaly=%s algos=%s csv=%s",
                 out.get("need_qt"), out.get("need_anomaly"),
                 out.get("algorithms"), out.get("csv_path"))
        return out
