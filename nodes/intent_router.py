# nodes/intent_router.py
from __future__ import annotations
import os, json, re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any
from json_repair import repair_json
from pathlib import Path
import logging

log = logging.getLogger("node.intent")
load_dotenv()

INTENT_MODE = os.getenv("INTENT_MODE", "llm").lower()  # "llm" | "rule"

# ---- DashScope OpenAI 兼容端点 ----
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 国际区：dashscope-intl.aliyuncs.com/compatible-mode/v1
)
DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY")
INTENT_MODEL       = os.getenv("INTENT_MODEL", "qwen-plus")
LLM_TIMEOUT        = float(os.getenv("LLM_TIMEOUT", "30"))  # 秒
LLM_MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "300"))

_SYS = (
    "You are an intent classifier. Return ONLY one JSON object with keys:\n"
    '{"need_qt": <bool>, "need_anomaly": <bool>, "csv_path": "<opt>", "algorithms": ["EIF","AE",...](optional)}.\n'
    "- If user asks selection/sort/group/aggregate/compute → need_qt=true.\n"
    "- If asks anomaly/outlier/EIF/AE/LOF/INNE/OCSVM → need_anomaly=true.\n"
    "- If both → both true. If neither → both false.\n"
    "Extract csv path if present in the text; else leave empty."
)

def _safe_json_loads(text: str) -> dict:
    """
    尝试稳健地从模型输出中提取 JSON：
    1) 截取最外层 {...}
    2) 先 json.loads；失败则用 json_repair.repair_json 修复后再 loads
    """
    l, r = text.find("{"), text.rfind("}")
    payload = text[l:r+1] if (l != -1 and r != -1 and r > l) else text
    try:
        return json.loads(payload)
    except Exception:
        fixed = repair_json(payload)
        return json.loads(fixed)

class IntentRouterNode:
    def __init__(self) -> None:
        if INTENT_MODE == "llm" and DASHSCOPE_API_KEY:
            self.client = OpenAI(
                base_url=DASHSCOPE_BASE_URL,
                api_key=DASHSCOPE_API_KEY,
                timeout=LLM_TIMEOUT,
                max_retries=0,
            )
        else:
            self.client = None

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
            try:
                resp = self.client.chat.completions.create(
                    model=INTENT_MODEL,
                    messages=[{"role": "system", "content": _SYS},
                              {"role": "user", "content": processed_input}],
                    temperature=0.0,
                    max_tokens=LLM_MAX_TOKENS,
                    response_format={"type": "json_object"},  # DashScope 支持 JSON 模式
                )
                raw = resp.choices[0].message.content or "{}"
                out = _safe_json_loads(raw)
            except Exception as e:
                log.warning("intent.llm failed (%s). Fallback to rule mode.", e)
                out = self._rule(processed_input)

        # —— 与规则引擎进行“并集纠偏”，避免 LLM 漏判 ——
        rule_guess = self._rule(processed_input)

        def _b(x): return bool(x)
        out["need_qt"]      = _b(out.get("need_qt"))      or _b(rule_guess.get("need_qt"))
        out["need_anomaly"] = _b(out.get("need_anomaly")) or _b(rule_guess.get("need_anomaly"))

        # 算法集合（去重 + 合法化）
        allow = {"EIF","AE","LOF","COPOD","INNE","OCSVM"}
        algos_llm  = [str(a).upper() for a in (out.get("algorithms") or [])]
        algos_rule = [str(a).upper() for a in (rule_guess.get("algorithms") or [])]
        algos = [a for a in dict.fromkeys(algos_llm + algos_rule) if a in allow]
        out["algorithms"] = algos or None

        # 路径兜底：优先用 preprocess 给到的绝对路径；否则对 LLM 路径做归一化与存在性修复
        if csv_path:
            out["csv_path"] = str(Path(csv_path))
        elif out.get("csv_path"):
            p = Path(out["csv_path"])
            if not p.exists():
                cand1 = Path.cwd() / p
                cand2 = Path.cwd() / "data" / p.name
                for c in (cand1, cand2):
                    if c.exists():
                        p = c; break
            out["csv_path"] = str(p)

        log.info("intent.out | need_qt=%s need_anomaly=%s algos=%s csv=%s",
                 out.get("need_qt"), out.get("need_anomaly"),
                 out.get("algorithms"), out.get("csv_path"))
        return out
