# sitecustomize.py —— 全局兼容层

import importlib
import os, logging

# --- Silence TensorFlow C++/Python logs early (before any tf import) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 2=warning及以上; 3=只保留fatal
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# 1. 如果新 sklearn 已提供 safe_tags，而老包还在找 _safe_tags
utils = importlib.import_module("sklearn.utils")
tags_mod = importlib.import_module("sklearn.utils._tags")

if (not hasattr(utils, "_safe_tags")   # 顶层 utils 缺旧名
        and hasattr(utils, "safe_tags")):             # 但有新名
    def _safe_tags(estimator, key=None):
        tags = utils.safe_tags(estimator)
        return tags if key is None else tags.get(key, None)

    # 挂到两个地方：utils 以及 utils._tags
    utils._safe_tags = _safe_tags
    tags_mod._safe_tags = _safe_tags