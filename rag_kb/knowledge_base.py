# rag_kb/knowledge_base.py
from __future__ import annotations
from typing import List, Dict, Any
import os, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from RAG_tools import TOOL_REGISTRY
from RAG_tool_functions import load_data
import logging
log = logging.getLogger("rag.kb")

class SimpleKB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs: list[str] = []
        self.meta: list[dict] = []
        self.emb: np.ndarray | None = None

    def add(self, text: str, **meta):
        self.docs.append(text.strip())
        self.meta.append(meta)

    def build(self):
        if not self.docs:
            self.emb = np.zeros((0, 384), dtype="float32")
        else:
            self.emb = self.model.encode(self.docs, convert_to_numpy=True, normalize_embeddings=True)
        log.info("KB built | docs=%d", len(self.docs))

    def search(self, query: str, top_k=8) -> list[str]:
        if self.emb is None or len(self.docs) == 0:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:top_k]
        hits = [self.docs[i] for i in idx]
        log.info("KB search | q_len=%d | top_k=%d", len(query), top_k)
        return [self.docs[i] for i in idx]

def _columns_from_path(csv_path: str) -> list[str]:
    p = Path(csv_path)
    if p.is_dir():
        sample = next(p.glob("*.csv"))
        df = load_data(str(sample))
    else:
        df = load_data(str(p))
    return list(df.columns)

def _build_tool_docs() -> list[str]:
    out = []
    for name, tool in TOOL_REGISTRY.items():
        req_keys = ", ".join(tool.signature)
        out.append(f"### {name}\n{tool.description}\nRequired keys: {req_keys}")
    return out

class KBNode:
    """构建 KB，并基于 query 召回若干片段，放入 state['kb_snippets']"""
    def run(self, query: str, csv_path: str | None, columns: list[str]) -> Dict[str, Any]:
        kb = SimpleKB()
        # 1) 列名文档
        cols = columns or (_columns_from_path(csv_path) if csv_path else [])
        for c in cols:
            kb.add(f"Column: {c} — Observed in the current dataset.")
        # 2) 工具文档
        for d in _build_tool_docs():
            kb.add(d)
        # 3) （可选）算法说明（简要）
        kb.add("EIF: Extended Isolation Forest (isotree). scoring_metric='depth'|'density'.")
        kb.add("AE: Autoencoder-based anomaly scoring (reconstruction error).")
        kb.add("LOF/OCSVM/COPOD/INNE: classic outlier detectors; input should be numeric features.")
        kb.build()
        snippets = kb.search(query, top_k=8)
        return {"kb_snippets": snippets}
