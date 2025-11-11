# rag_nodes_react/models.py
# Use Pydantic to turn "Action JSON" and "Finish JSON" into verifiable objects.
# Other nodes should reference them for validation; no business logic should be executed.

from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from RAG_tools import TOOL_REGISTRY

AllowedTool = Literal[tuple(TOOL_REGISTRY.keys())]
# `TOOL_REGISTRY` is a collection of utility functions dynamically gathered in `RAG_tools.py`; it tells Pydantic that "the function field only allows these N strings".
# `.keys()` returns all keys in the dictionary, an iterator of `dict_keys`, such as `["select_rows", "calculate_average", …]`
# `tuple( … )` converts the iterator into a true tuple because `typing.Literal` requires statically iterable values.
# `Literal[ … ]` declares an enumeration type annotation—only the listed literals are valid; otherwise, Pydantic will throw an error.
# Therefore, `AllowedTool` is a custom type that "can only take one of those dozens of function names in `TOOL_REGISTRY`".

class Action(BaseModel):
    function: AllowedTool   # function: 后面是类型注解，告诉 Pydantic 用 AllowedTool 校验
    args: Dict[str, Any] = Field(default_factory=dict)  # args: 是普通字典，default_factory=dict 表示 “省略时给空字典” 而非 None。

class Finish(BaseModel):
    finish: str # 仅有一个键 finish，用于 "{"finish": "All done"}" 这种终止信号