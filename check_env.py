# check_env.py

reqs = [
    # core
    "langgraph", "langchain", "langchain_openai", "openai", "python-dotenv",
    # data / viz
    "pandas", "numpy", "matplotlib", "plotly", "openpyxl",
    # retriever
    "sentence_transformers", "faiss-cpu",  # faiss 可选，当前实现用 SBERT + numpy 即可
    # graph / neo4j（如暂不用可注释）
    "networkx", "pyvis", "neo4j",
    # anomaly
    "isotree", "pyod", "scikit-learn", "tensorflow",
    # utils
    "json-repair",
    # optional
    "sktree",
]
missing = []
for r in reqs:
    try:
        __import__(r.replace("-", "_"))
    except ImportError:
        missing.append(r)
if missing:
    print("缺少包：", missing)
else:
    print("解释器与依赖全部就绪")
