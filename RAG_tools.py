# RAG_tools.py

"""Wrap all low-level functions into BaseTool classes + registry."""
import json, inspect, pandas as pd
import RAG_tool_functions as tf
from typing import Any, Dict, Optional, Callable
from pydantic import BaseModel, SkipValidation
from langchain_core.tools import BaseTool

# —— 会话级状态 —— (DF + scalar)
_STATE: Dict[str, Any] = {"current_df": None, "last_scalar": None}  # 把跨步骤共享的缓存清零，确保每次新请求不会受到上次遗留 DataFrame 或标量的影响。

def reset_state() -> None:
    _STATE["current_df"] = None
    _STATE["last_scalar"] = None

class DataFrameTool(BaseTool):  # BaseTool 来自 LangChain ，用于把任意函数包装成「可在 Agent/Graph 里统一调用」的工具对象。
    """
    Type-hint 写法，告诉静态检查器这三个属性的类型。最后一个 callable / Callable：表示 “可以被调用的对象（函数或带 __call__ 的对象）”。
    包装任意表格处理函数，使其在 LangGraph / Agent 中可统一调用。
    """

    name: str
    description: str
    func: SkipValidation[Callable[..., Any]]
    # 屏蔽 “无法对 callable 做严格校验” 的警告

    model_config = {"arbitrary_types_allowed": True}
    # 遇到任何不是标准 Python 类型（如 function、module、Tensor…）都别校验，直接存。

    def _run(self, tool_input: str) -> str:             # sync only
        args = json.loads(tool_input) if tool_input else {} # 解析 tool_input → args（如果字符串为空就给空字典）。
        cur_df: Optional[pd.DataFrame] = _STATE["current_df"]   # cur_df 取自全局 _STATE，这样所有工具都能共享链式操作后的最新 DataFrame。
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
        # 先尝试类级 __signature__；若未生成，再退回真正函数的 inspect.signature
        sig = getattr(type(self), "__signature__", None)
        if sig is None:
            sig = inspect.signature(self.func)
        return sig.parameters    # ← 返回一个有序映射，可在外部直接 join

    async def _arun(self, tool_input: str) -> str:      # not used
        raise NotImplementedError()

# —— 动态注册 —— 只要 _PREFIX 即会自动生成工具
_PREFIX = (
    "select_rows", "sort_rows", "group_", "top_n", "filter_date_between_start_end",
    "add_derived_column", "rolling_average", "calculate_", "count_rows",
    "graph_export", "plot_machine_avg_bar", "plot_concurrent_tasks_line", "select_columns"
)
TOOL_REGISTRY: Dict[str, DataFrameTool] = {}    # 创建一个全局字典，键是函数名，值是已经包装好的 DataFrameTool 实例，方便运行期动态查找。

for fname, func in inspect.getmembers(tf, inspect.isfunction):
    if fname.startswith(_PREFIX):
        doc = " ".join((func.__doc__ or "Data processing tool").split())
        tool = DataFrameTool(name=fname, description=doc, func=func)
        TOOL_REGISTRY[fname] = tool

__all__ = ["TOOL_REGISTRY", "reset_state"]