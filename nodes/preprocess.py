# nodes/preprocess.py
from __future__ import annotations
import re, os, glob
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
from RAG_tool_functions import load_data
from logging_setup import get_logger
log = get_logger("node.preprocess")

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def _best_match(token: str, columns: List[str]) -> str | None:
    cands = [c for c,score,_ in process.extract(token, columns, scorer=fuzz.WRatio, limit=3) if score > 80]
    if not cands: return None
    e1 = _MODEL.encode(token, convert_to_tensor=True, show_progress_bar=False)
    e2 = _MODEL.encode(cands, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(e1, e2)[0].tolist()
    return cands[sims.index(max(sims))]

class PreprocessNode:
    """提取 csv 路径/目录；读取列名；列名纠错到 processed_input；返回 columns 与 csv_path"""
    def run(self, user_input: str) -> Dict[str, Any]:
        log.info("start preprocess | raw=%s", user_input)
        lower = user_input.lower().strip()
        # 1) SKAB 目录快速路径
        if "skab" in lower:
            base = Path.cwd() / "data" / "SKAB"
            group = None
            for g in ["valve1", "valve2", "anomaly-free", "other"]:
                if g in lower: group = g; break
            csv_dir = base / group if group else base
            if not csv_dir.exists():
                raise FileNotFoundError(f"目录不存在: {csv_dir}")
            # 任取一个文件采样列名
            sample = next(csv_dir.glob("*.csv"))
            df = load_data(str(sample))
            cols = df.columns.tolist()
            corrected = self._correct_columns_in_text(user_input, cols)
            log.info("preprocess | SKAB dir=%s | cols=%d", csv_dir, len(cols))
            return {"processed_input": corrected, "csv_path": str(csv_dir), "columns": cols}

        # 2) 单文件：匹配 *.csv
        m = re.search(r"\b([A-Za-z0-9_\./\\-]+\.csv)\b", user_input, flags=re.I)
        if m:
            p = Path(m.group(1))
            if not p.exists():
                # 在 data/ 里找
                for d in [Path.cwd()/ "data", Path.cwd()]:
                    cand = d / p.name
                    if cand.exists(): p = cand; break
            df = load_data(str(p))
            cols = df.columns.tolist()
            corrected = self._correct_columns_in_text(user_input, cols)
            log.info("preprocess | file=%s | cols=%d", p, len(cols))
            return {"processed_input": corrected, "csv_path": str(p), "columns": cols}

        # 3) 无路径：用默认 CSV
        # 3) 无路径：用默认 CSV
        from RAG_tool_functions import CSV_FILE
        df = load_data(CSV_FILE)
        cols = df.columns.tolist()
        corrected = self._correct_columns_in_text(user_input, cols)
        log.info("preprocess | default_file=%s | cols=%d", Path(CSV_FILE), len(cols))
        return {"processed_input": corrected, "csv_path": str(Path(CSV_FILE)), "columns": cols}

    def _correct_columns_in_text(self, text: str, columns: List[str]) -> str:
        pat = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        def repl(m):
            w = m.group(0)
            real = _best_match(w, columns)
            return real or w
        out = pat.sub(repl, text)
        return re.sub(r"\s+", " ", out).strip()
