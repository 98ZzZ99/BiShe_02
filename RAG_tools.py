# RAG_tools.py

"""Wrap all low-level functions into BaseTool classes + registry."""
import json, inspect, pandas as pd
import RAG_tool_functions as tf
from typing import Any, Dict, Optional, Callable
from pydantic import BaseModel, SkipValidation
from langchain_core.tools import BaseTool

# —— Session-level state —— (DF + scalar)
_STATE: Dict[str, Any] = {"current_df": None, "last_scalar": None}  # 把跨步骤共享的缓存清零，确保每次新请求不会受到上次遗留 DataFrame 或标量的影响。

def reset_state() -> None:
    _STATE["current_df"] = None
    _STATE["last_scalar"] = None

class DataFrameTool(BaseTool):  # BaseTool, from LangChain, is used to wrap any function into a tool object that can be uniformly called within the Agent/Graph.
    """
    The `type-hint` syntax tells the static analyzer the types of these three properties. The last one, `callable` / `Callable`, indicates "an object (function or object with `__call__`) that can be called."
    It wraps any table processing function, making it uniformly callable within LangGraph/Agent.
    """

    name: str
    description: str
    func: SkipValidation[Callable[..., Any]]
    # Suppress the warning "Unable to perform strict validation on callable".

    model_config = {"arbitrary_types_allowed": True}
    # Do not validate any non-standard Python types (such as function, module, Tensor, etc.), just save them directly.

    def _run(self, tool_input: str) -> str:             # sync only
        args = json.loads(tool_input) if tool_input else {} # Parse tool_input → args (if the string is empty, give an empty dictionary).
        cur_df: Optional[pd.DataFrame] = _STATE["current_df"]   # cur_df is taken from the global _STATE, so that all tools can share the latest DataFrame after chained operations.
        result = self.func(cur_df, args)

        if isinstance(result, pd.DataFrame):
            _STATE["current_df"] = result
            preview = result.head(10).to_string(index=False)
            return f"[DataFrame updated]\n{preview}"
        else:
            _STATE["last_scalar"] = result
            return str(result)

    @property
    def signature(self):
        # First try class-level `__signature__`; if it doesn't generate, then fall back to the actual function's `inspect.signature`.
        sig = getattr(type(self), "__signature__", None)
        if sig is None:
            sig = inspect.signature(self.func)
        return sig.parameters    # ← Returns an ordered mapping that can be joined directly from the outside.

    async def _arun(self, tool_input: str) -> str:      # not used
        raise NotImplementedError()

# —— Dynamic registration —— tools are automatically generated simply by using _PREFIX.
_PREFIX = (
    "select_rows", "sort_rows", "group_", "top_n", "filter_date_between_start_end",
    "add_derived_column", "rolling_average", "calculate_", "count_rows",
    "graph_export", "plot_machine_avg_bar", "plot_concurrent_tasks_line", "select_columns"
)
TOOL_REGISTRY: Dict[str, DataFrameTool] = {}    # Create a global dictionary where the keys are function names and the values are pre-packaged DataFrameTool instances for easy dynamic lookup at runtime.

for fname, func in inspect.getmembers(tf, inspect.isfunction):
    if fname.startswith(_PREFIX):
        doc = " ".join((func.__doc__ or "Data processing tool").split())
        tool = DataFrameTool(name=fname, description=doc, func=func)
        TOOL_REGISTRY[fname] = tool

__all__ = ["TOOL_REGISTRY", "reset_state"]