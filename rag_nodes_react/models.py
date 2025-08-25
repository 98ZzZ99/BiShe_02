# rag_nodes_react/models.py
# 用 Pydantic 把 “Action JSON” 与 “Finish JSON” 变成可验证对象。
# 其他节点都要引用它来做校验；不执行任何业务逻辑。

from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from RAG_tools import TOOL_REGISTRY

AllowedTool = Literal[tuple(TOOL_REGISTRY.keys())]
# TOOL_REGISTRY 是在 RAG_tools.py 动态收集的全部工具函数；告诉 Pydantic “function 字段只允许这 N 个字符串”。
# .keys() 返回字典所有键，是一个 dict_keys 迭代器，如 ["select_rows", "calculate_average", …]
# tuple( … ) 把迭代器转换成真正的元组，因为 typing.Literal 需要 静态可迭代 的值。
# Literal[ … ] 声明 枚举型类型注解——只有列出的字面值合法，否则 pydantic 校验会报错。
# 因此 AllowedTool 就是 “只能取 TOOL_REGISTRY 里那几十个函数名之一” 的自定义类型。

class Action(BaseModel):
    function: AllowedTool   # function: 后面是类型注解，告诉 Pydantic 用 AllowedTool 校验
    args: Dict[str, Any] = Field(default_factory=dict)  # args: 是普通字典，default_factory=dict 表示 “省略时给空字典” 而非 None。

class Finish(BaseModel):
    finish: str # 仅有一个键 finish，用于 "{"finish": "All done"}" 这种终止信号