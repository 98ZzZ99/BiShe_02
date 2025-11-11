# logging_setup.py
import logging, os, sys, pathlib
from logging.handlers import RotatingFileHandler
from functools import wraps

LOG_DIR  = pathlib.Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / os.getenv("LOG_FILE", "app.log")
LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()

def configure_logging():
    """Initialize root logger once (console + rotating file)."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(LEVEL)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LEVEL); ch.setFormatter(fmt)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(LEVEL); fh.setFormatter(fmt)

    root.addHandler(ch); root.addHandler(fh)

    # Common third-party libraries for noise reduction
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    logging.getLogger("tensorflow").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def log_call(logger: logging.Logger):
    """Function-level tagging: Input parameter + return value (DF will include shape/cols)"""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            try:
                logger.debug("CALL %s args=%s kwargs=%s", fn.__name__, a, kw)
            except Exception:
                pass
            res = fn(*a, **kw)
            try:
                import pandas as pd
                if isinstance(res, pd.DataFrame):
                    logger.debug("RET  %s -> DataFrame shape=%s cols=%s",
                                 fn.__name__, getattr(res, "shape", None),
                                 list(res.columns))
                else:
                    logger.debug("RET  %s -> %s", fn.__name__, type(res).__name__)
            except Exception:
                pass
            return res
        return wrapper
    return deco
