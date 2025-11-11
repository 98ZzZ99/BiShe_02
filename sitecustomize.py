# sitecustomize.py —— Global compatibility layer

import importlib
import os, logging

# --- Silence TensorFlow C++/Python logs early (before any tf import) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 2=warning及以上; 3=只保留fatal
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# 1. If the new sklearn package already provides `safe_tags`, while the older package is still looking for `_safe_tags`.
utils = importlib.import_module("sklearn.utils")
tags_mod = importlib.import_module("sklearn.utils._tags")

if (not hasattr(utils, "_safe_tags")   # The top-level utils is missing an old name.
        and hasattr(utils, "safe_tags")):             # But there is a new name
    def _safe_tags(estimator, key=None):
        tags = utils.safe_tags(estimator)
        return tags if key is None else tags.get(key, None)

    # It's attached to two places: utils and utils._tags.
    utils._safe_tags = _safe_tags
    tags_mod._safe_tags = _safe_tags