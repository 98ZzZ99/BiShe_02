# RAG_tool_functions.py
# Note: This file contains utility functions such as "select", "sort", "average", "median", and "mode".
#       And the logic for loading and processing CSV data.
"""
Global DataFrame → DataFrame
select_rows（supporting AND/OR）、sort_rows、top_n、group_top_n、filter_date_range、add_derived_column、rolling_average

Statistical Scalar
calculate_average、median、mode、sum、min、max、std、variance、percentile、correlation、covariance

special
calculate_failure_rate、calculate_delay_avg
"""

import os, re, time
import pandas as pd
import numpy as np
import operator as _op
from pathlib import Path
from logging_setup import get_logger, log_call

def _get_one(d, *keys, default=None):
    """Take the first non-None key that exists in args according to priority; and unpack the single-element list into a scalar."""
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, (list, tuple)) and len(v) == 1:
                return v[0]
            return v
    return default

def _ensure_dt(df, col):
    """Force the column to datetime64 (in-place conversion)."""
    if col not in df.columns:
        raise KeyError(f"列不存在: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df[col]

# For {last_scalar} placeholder in add_derived_column
_LAST_SCALAR = None

log = get_logger("tools")

# ---------- Constant ----------
_BOPS = {"==": _op.eq, "!=": _op.ne, "<=": _op.le,
         ">=": _op.ge, "<": _op.lt,  ">": _op.gt}

TIME_COLS = ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]
CSV_FILE = os.path.join("data", "hybrid_manufacturing_categorical.csv")

# ---------- Base ----------
TIME_COLS = ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]
CSV_FILE = os.path.join("data", "hybrid_manufacturing_categorical.csv")

# ==== Public Tools ====
def _col(args: dict, *names, default=None):
    """Try retrieving the column names sequentially (column / target_column / …)."""
    for n in names:
        if n in args:
            return args[n]
    if default is not None:
        return default
    raise KeyError(f"Need one of {names}")

# RAG_tool_functions.py  —— Unified Alias Tool
def _pick_target(d: dict, *keys: str, default=None):
    """Returns the first matching key in d; compatible with the {"derived": {...}} syntax."""
    # A fallback: When LLM mistakenly uses {"derived": {"Energy_Consumption": "Energy_Consumption"}} as a parameter
    if "derived" in d and isinstance(d["derived"], dict) and len(d["derived"]) == 1:
        return next(iter(d["derived"].values()))
    for k in keys:
        if k in d:
            return d[k]
    if default is not None:
        return default
    raise KeyError(f"Expect one of {keys}")

# ----------- General tools -----------
def _df(cur):
    """始终返回 DataFrame；避免布尔歧义"""
    return cur if cur is not None else load_data()

@log_call(log)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    General CSV Reading Functions: Automatically Recognize Delimiters and Parse Time Columns.
    1. If the first row contains more semicolons than commas, use `;` as the delimiter; otherwise, use `,`.
    2. Remove leading and trailing spaces from column names and replace spaces with `_`.
    3. Rename the `datetime` column to `time_stamp` and parse it using `pd.to_datetime`.
    """
    p = Path(path or CSV_FILE)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        sample_line = f.readline()
    sep = ';' if sample_line.count(';') > sample_line.count(',') else ','
    # Read data
    try:
        if sep == ',':
            # When using comma-separated time columns, predefined time columns can be parsed in advance.
            header = sample_line.rstrip('\n')
            cols = [c.strip() for c in header.split(sep)]
            date_cols = [c for c in TIME_COLS if c in cols]
            df = pd.read_csv(p, sep=sep, parse_dates=date_cols or None, dayfirst=False)
        else:
            df = pd.read_csv(p, sep=sep)
    except ValueError:
        df = pd.read_csv(p, sep=sep)
    # Clean up column names and rename datetime
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'time_stamp'})
    # Parse timestamp column
    for c in df.columns:
        if c.lower().endswith('timestamp') or c.lower() == 'time_stamp':
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def _num(s: pd.Series):
    """Safe conversion to numerical value (returns NaN if unsuccessful)"""
    return pd.to_numeric(s, errors="coerce")

# ---------- 1) Row-level function (DF→DF) ----------
@log_call(log)
# def select_rows(cur, args):
#     """
#       Supported:
#           • Single column comparison {column:…, condition:" > 5"} or {condition:"Processing_Time <= 50"}
#           • AND / OR … AND …
#           • Inter-column expression ">= Other_Col + 10"
#           • Time comparison 'HH:MM'
#           • top_n condition:"top_n" + n/order/sort_column
#           • Time difference expression "Actual_End - Actual_Start > 4 hours"
#     """
#     df = _df(cur)
#
#     # ---------- top_n special branches ----------
#     if args.get("condition") == "top_n":
#         col      = args.get("column") or args.get("target_column") or args.get("sort_column")
#         if not col:
#             raise ValueError("top_n 需要 column/sort_column")
#         n        = int(args.get("n", 5))
#         asc      = args.get("order", "desc") == "asc"
#         sort_col = args.get("sort_column", col)
#         return df.sort_values(sort_col, ascending=asc).head(n)
#
#     # Securely obtain condition
#     if "condition" not in args:
#         raise ValueError("select_rows 需要 'condition'")
#     condition = str(args["condition"]).strip()
#
#     # ---------- Default column names: parsed first from the condition header. ----------
#     col = args.get("column") or args.get("target_column")
#     if col is None:
#         m0 = re.match(r"\s*([A-Za-z_]\w*)\s*(==|!=|<=|>=|<|>)", condition)
#         if m0:
#             col = m0.group(1)
#             condition = condition[len(m0.group(1)):].lstrip()
#         else:
#             raise ValueError("select_rows: 无法确定列名（请提供 column 或在 condition 里写 'Col OP ...'）")
#
#     # ---------- 递归处理 AND / OR ----------
#     if m := re.search(r"\s+(AND|OR)\s+", condition, flags=re.I):
#         left, logic, right = re.split(r"\s+(AND|OR)\s+", condition, 1, flags=re.I)
#         df1 = select_rows(df, {"column": col, "condition": left})
#         if re.match(r"\s*[A-Za-z_]\w*\s*(==|!=|<=|>=|<|>)", right):
#             df2 = select_rows(df, {"condition": right})
#         else:
#             df2 = select_rows(df, {"column": col, "condition": right})
#         return (pd.merge(df1, df2) if logic.upper() == "AND"
#                 else pd.concat([df1, df2]).drop_duplicates())
#
#     # ---------- 时间差表达式/表达式/列-列比较/时间点/普通值 ----------
#     # （以下与原实现一致，仅略去注释与重复代码）
#     m_expr = None
#     if col and "-" in col and re.match(r"^\s*[<>]=?", condition):
#         m_expr = re.match(r"^\s*([A-Za-z_]\w*)\s*-\s*([A-Za-z_]\w*)\s*$", col)
#     if m_expr:
#         col_a, col_b = m_expr.groups()
#         m_cond = re.match(r"^\s*([<>]=?)\s*(\d+(?:\.\d+)?)\s*(hours?|minutes?|seconds?)\s*$",
#                           condition, flags=re.I)
#         if not m_cond:
#             raise ValueError("Bad time-diff condition syntax")
#         op_sym, num_str, unit = m_cond.groups()
#         seconds = float(num_str) * {"seconds":1,"second":1,"minutes":60,"minute":60,"hours":3600,"hour":3600}[unit.lower()]
#         delta = (pd.to_datetime(df[col_a]) - pd.to_datetime(df[col_b])).dt.total_seconds()
#         return df[_BOPS[op_sym](delta, seconds)]
#
#     m_td = re.match(r"""^\s*([A-Za-z_]\w*)\s*-\s*([A-Za-z_]\w*)\s*([<>]=?)\s*(\d+(?:\.\d+)?)\s*(?:\s*(hours?|minutes?|seconds?))?\s*$""",
#                     condition, flags=re.I | re.X)
#     if m_td:
#         col_a, col_b, op_sym, num_str, unit = m_td.groups()
#         unit = (unit or "seconds").lower()
#         seconds = float(num_str) * {"seconds":1,"second":1,"minutes":60,"minute":60,"hours":3600,"hour":3600}[unit]
#         delta = (pd.to_datetime(df[col_a]) - pd.to_datetime(df[col_b])).dt.total_seconds()
#         return df[_BOPS[op_sym](delta, seconds)]
#
#     m = re.match(r"^(==|!=|<=|>=|<|>)\s*(.+)$", condition)
#     if not m:
#         raise ValueError("Bad condition syntax")
#     op_sym, rhs_raw = m.groups()
#     rhs_raw = rhs_raw.strip(' "\'')
#
#     if any(sym in rhs_raw for sym in "+-*/") and rhs_raw not in df.columns:
#         return df[df.eval(f"`{col}` {op_sym} {rhs_raw}")]
#     if rhs_raw in df.columns:
#         return df[_BOPS[op_sym](df[col], df[rhs_raw])]
#     if re.fullmatch(r"\d{1,2}:\d{2}", rhs_raw):
#         t_val = pd.to_datetime(rhs_raw, format="%H:%M").time()
#         return df[_BOPS[op_sym](pd.to_datetime(df[col], errors="coerce").dt.time, t_val)]
#     if col in TIME_COLS:
#         if re.fullmatch(r"\d{4}-\d{2}-\d{2}", rhs_raw):
#             rhs_raw += " 00:00"
#         val = pd.to_datetime(rhs_raw, errors="coerce")
#     else:
#         val = pd.to_numeric(rhs_raw, errors="coerce")
#         if pd.isna(val): val = rhs_raw
#     return df[_BOPS[op_sym](df[col], val)]
def select_rows(df, args):
    """
    args:
      - query / condition: pandas query string; also supports 'true' / 'True' / '*' full values.
      - When a natural statement such as "Scheduled_End - Scheduled_Start >= X hour(s)" appears, it is automatically parsed.
    """
    if df is None:
        raise ValueError("select_rows: 当前数据为空")

    query = args.get("query") or args.get("condition")
    if not query:
        raise ValueError("select_rows 需要 'query' 或 'condition'")

    q = str(query).strip()
    if q.lower() in ("true", "all", "*"):
        return df.copy()

    # Parsing the natural language of "time difference >= X hour(s)"
    low = q.lower().replace(" ", "")
    if ("scheduled_end-scheduled_start" in low) and (">=" in low) and ("hour" in low):
        import re
        m = re.search(r">=([0-9]+)", low)
        if not m:
            raise ValueError("select_rows: 无法解析小时数")
        hours = int(m.group(1))
        end = pd.to_datetime(df["Scheduled_End"], errors="coerce")
        start = pd.to_datetime(df["Scheduled_Start"], errors="coerce")
        mask = (end - start) >= pd.Timedelta(hours=hours)
        return df.loc[mask].copy()

    # Otherwise, use pandas.query (the column names must be valid variable names; if not, you can use a boolean expression instead).
    try:
        return df.query(q).copy()
    except Exception as e:
        raise ValueError(f"select_rows: 解析 query 失败 -> {e}")

@log_call(log)
def sort_rows(cur, args):
    df = _df(cur)
    col = args.get("column") or args.get("target_column") or args.get("sort_by") or args.get("by")
    if not col:
        raise ValueError("sort_rows 需要 'column'（支持别名 sort_by/by/target_column）")
    order = str(args.get("order", "asc")).lower()
    asc   = True if order in ("asc", "ascending", "升序") else False
    return df.sort_values(col, ascending=asc)

@log_call(log)
def top_n(df, args):
    """
    args:
      - n: int
      - column/sort_column/target_column (optional): If given, sort by this column and take the first n; otherwise, directly use df.head(n).
      - pair/columns (optional): retrieve columns
    """
    if df is None:
        raise ValueError("top_n: 当前数据为空")
    n = int(args.get("n", 5))
    col = (args.get("column") or args.get("sort_column") or args.get("target_column"))
    out = df
    if col:
        out = out.sort_values(col, ascending=False)
    cols = args.get("pair") or args.get("columns")
    if cols:
        out = out[cols]
    return out.head(n).copy()

@log_call(log)
def group_top_n(cur, args):
    """Take the first N from each group; you can keep_all=True to retain the remaining columns."""
    df = _df(cur)
    g  = args["group_column"]
    s  = _pick_target(args, "sort_column", "column")
    n  = int(args.get("n", 1))
    asc = args.get("order", "desc") == "asc"

    out = (df.sort_values(s, ascending=asc)
             .groupby(g, as_index=False).head(n))
    return out if args.get("keep_all", True) else out[[g, s]]

@log_call(log)
def filter_date_between_start_end(cur, args):
    """
    Quick Time Window
      args = {"column":"Actual_Start",
              "start":"2023-03-18 10:00",
              "end":"2023-03-18 12:00",
              "inclusive":"both"}
    """
    df    = _df(cur)
    col   = _col(args, "column", "target_column")
    start = pd.to_datetime(args["start"])
    end   = pd.to_datetime(args["end"])
    incl  = args.get("inclusive", "both")
    mask  = df[col].between(start, end, inclusive=incl)
    return df[mask]

@log_call(log)
# def add_derived_column(cur, args):
#     """
#     支持：
#       • 一般算式               "Energy_Consumption / Processing_Time"
#       • 单个日期差             "Scheduled_Start - Actual_Start"  → 秒
#       • 占位符 {last_scalar}   参考注释
#       • colA/colB 写法（不传 formula）
#     """
#     df = _df(cur)
#
#     # ------- colA / colB 快捷写法 -------
#     if "formula" not in args:
#         if {"colA", "colB"} <= args.keys():
#             unit = args.get("unit", "seconds")
#             delta = (pd.to_datetime(df[args["colA"]]) -
#                      pd.to_datetime(df[args["colB"]])).dt.total_seconds()
#             if unit == "minutes":
#                 delta /= 60
#             elif unit == "hours":
#                 delta /= 3600
#             df[args["name"]] = delta
#             return df
#         raise KeyError("add_derived_column: need 'formula' or colA/colB")
#
#     # ------- 正常 formula 路径 -------
#     formula = args["formula"]
#
#     # 占位符 {last_scalar}
#     if "{last_scalar}" in formula:
#         last = globals().get("_LAST_SCALAR")
#         formula = formula.replace("{last_scalar}", str(last) if last is not None else "0")
#
#     # 单个日期差（自动转秒）
#     if " - " in formula and any(c in formula for c in TIME_COLS):
#         lhs, rhs = [s.strip() for s in formula.split("-", 1)]
#         delta = (pd.to_datetime(df[lhs]) -
#                  pd.to_datetime(df[rhs])).dt.total_seconds()
#         df[args["name"]] = delta
#     else:
#         df[args["name"]] = df.eval(formula)
#
#     return df

def add_derived_column(df, args):
    """
    Added derived columns.
    - Target column name: 'target_column' / 'name' / 'column'
    - Supports formulas "A - B" representing time difference; unit can be 'minutes' (default) / 'hours' / 'seconds'
    - Other expressions can be calculated using pd.eval (column names must be valid variable names)
    """
    if df is None:
        raise ValueError("add_derived_column: 当前数据为空")

    name = _get_one(args, "target_column", "name", "column")
    formula = args.get("formula")
    unit = str(args.get("unit", "minutes")).lower()
    if not name or not formula:
        raise ValueError("add_derived_column 需要 'formula' 和 目标列名('target_column'/'name')")

    # Time difference：A - B
    m = re.fullmatch(r"\s*(\w+)\s*-\s*(\w+)\s*", formula)
    if m:
        a, b = m.groups()
        A = _ensure_dt(df, a)
        B = _ensure_dt(df, b)
        delta = A - B
        if unit.startswith("hour"):
            val = delta.dt.total_seconds() / 3600.0
        elif unit.startswith("sec"):
            val = delta.dt.total_seconds()
        else:
            val = delta.dt.total_seconds() / 60.0  # 默认分钟
        df[name] = val
        return df

    # Other expressions：pd.eval
    try:
        df[name] = pd.eval(formula, engine="python",
                           local_dict={c: df[c] for c in df.columns})
        return df
    except Exception as e:
        raise ValueError(f"add_derived_column: 解析公式失败: {formula!r}: {e}")

@log_call(log)
# def rolling_average(cur, args):
#     """支持全局与分组滚动平均"""
#     df = _df(cur)
#     w  = int(args.get("window", 3))
#     col = args["column"]
#     g   = args.get("group_by")
#
#     if g:
#         res = (df.sort_values(g)
#                  .groupby(g, group_keys=False)[col]
#                  .rolling(w, min_periods=1).mean()
#                  .reset_index())
#         res.rename(columns={col: f"rolling_avg_{col}"}, inplace=True)
#         return res
#     else:
#         ser = _num(df[col]).rolling(w, min_periods=1).mean()
#         return df.assign(**{f"rolling_avg_{col}": ser})
def rolling_average(df, args):
    """
    args:
      - window: int
      - group_by: str column names
      - target_column / column: target numerical column
    """
    if df is None:
        raise ValueError("rolling_average: 当前数据为空")
    win = int(args.get("window", 3))
    group = args.get("group_by")
    col = args.get("target_column") or args.get("column")
    if not group or not col:
        raise ValueError("rolling_average: 需要 group_by 和 target_column/column")

    df2 = df.copy()
    if not pd.api.types.is_numeric_dtype(df2[col]):
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2.sort_values([group], inplace=True)
    df2[f"{col}_rollavg_{win}"] = df2.groupby(group)[col].transform(lambda s: s.rolling(win, min_periods=1).mean())
    return df2

# ---------- 2) group_by_aggregate ----------
_AGG_MAP = {
    "avg": "mean", "mean": "mean",
    "sum": "sum",  "count":"count",
    "max": "max",  "min": "min",
    "std": "std",  "var": "var",
    "variance":"var",
    "percentile":"quantile",  # ← New
}

@log_call(log)
# def group_by_aggregate(cur, args):
#     """
#     支持 agg：
#       • 平均/总和/极值/计数/标准差/方差
#       • percentile   指定 q/percentile
#       • cov / corr   双列
#     keep_all=True 时将结果 left-join 回原 DF
#     """
#     df   = _df(cur)
#     gcol = _col(args, "group_column")
#     tcol = _col(args, "column", "target_column")
#     agg  = str(args.get("agg", "avg")).lower()
#     keep = bool(args.get("keep_all", True))   # **默认 True**，防止后续步骤缺列
#
#     # ---------- percentile ----------
#     if agg in {"percentile", "quantile"}:
#         q = float(args.get("q") or args.get("percentile")
#                   or args.get("percent") or 50)
#         res = (_num(df[tcol]).groupby(df[gcol])
#                .quantile(q/100)
#                .reset_index(name=f"p{int(q)}_{tcol}"))
#         return df.merge(res, on=gcol, how="left") if keep else res
#
#     # ---------- cov / corr ----------
#     if agg in {"cov", "covariance", "corr"}:
#         other = (_col(args, "other_column", "y", "column2",
#                       default=None))
#         if other is None:
#             raise ValueError("cov/corr 需要指定 other_column / y")
#         func  = pd.Series.cov if agg.startswith("cov") else pd.Series.corr
#         res = (_num(df[tcol]).groupby(df[gcol])
#                .apply(lambda s: func(s, _num(df[other]).loc[s.index]))
#                .reset_index(name=f"{agg[:3]}_{tcol}"))
#         return df.merge(res, on=gcol, how="left") if keep else res
#
#     # ---------- 单列常规聚合 ----------
#     if agg not in _AGG_MAP:
#         raise ValueError(f"Unsupported agg '{agg}'")
#     func = _AGG_MAP[agg]
#     res = getattr(_num(df[tcol]).groupby(df[gcol]), func)() \
#             .reset_index(name=f"{func}_{tcol}")
#     return df.merge(res, on=gcol, how="left") if keep else res

def group_by_aggregate(df, args):
    """
    Grouping aggregation: group_column must be specified; the target column can be 'pair' (or str or single-element list for a single column),
    or 'column', or 'target_column'; agg defaults to 'mean'.
    """
    if df is None:
        raise ValueError("group_by_aggregate: 当前数据为空")
    grp = _get_one(args, "group_column")
    if not grp:
        raise KeyError("group_by_aggregate: 需要 'group_column'")
    agg = args.get("agg", "mean")
    tcol = _get_one(args, "pair", "column", "target_column")
    if isinstance(tcol, (list, tuple)) and len(tcol) == 1:
        tcol = tcol[0]
    if not tcol:
        raise KeyError("group_by_aggregate: 需要 'pair' 或 'column' 或 'target_column'")

    res = df.groupby(grp)[tcol].agg(agg).reset_index()
    return res


# ---------- 3) 标量 ----------
@log_call(log)
def calculate_average(cur, args):
    col = _pick_target(args, "column", "target_column")
    df  = _df(cur)

    if " - " in col:   # Quick: Date Difference (seconds)
        lhs, rhs = [c.strip() for c in col.split("-", 1)]
        delta = (pd.to_datetime(df[lhs]) - pd.to_datetime(df[rhs])).dt.total_seconds()
        unit  = args.get("unit")
        if unit == "minutes": delta /= 60
        elif unit == "hours": delta /= 3600
        return delta.mean()
    return _num(df[col]).mean()

@log_call(log)
def calculate_median(cur, args):
    return _num(_df(cur)[args["column"]]).median()

@log_call(log)
def calculate_mode(cur, args):
    ser = _df(cur)[args["column"]].mode()
    return ser.iat[0] if not ser.empty else None

@log_call(log)
def calculate_sum(cur,args):
    df=_df(cur); return _num(df[args["column"]]).sum()
@log_call(log)
def calculate_min(cur,args):
    df=_df(cur); return _num(df[args["column"]]).min()
@log_call(log)
def calculate_max(cur,args):
    df=_df(cur); return _num(df[args["column"]]).max()
@log_call(log)
def calculate_std(cur,args):
    df=_df(cur); return _num(df[args["column"]]).std()
@log_call(log)
def calculate_variance(cur,args):
    df=_df(cur); return _num(df[args["column"]]).var()
@log_call(log)
def calculate_percentile(cur, args):
    df = _df(cur)
    q = float(args.get("percentile") or args.get("q") or 90)
    g = args.get("group_by") or args.get("group_column")
    if g:
        return (df.groupby(g)[args["column"]]
                  .quantile(q/100)
                  .reset_index(name=f"p{int(q)}_{args['column']}"))
    return _num(df[args["column"]]).quantile(q/100)
@log_call(log)
def calculate_correlation(cur, args):
    """
    Calculate the correlation coefficient between two columns.
    The `args` parameter can be `{ "x":"colA", "y":"colB" }`
        It also supports `{ "column1":…, "column2":… }` for compatibility with older JSON.
    If there are insufficient samples, it returns `np.nan` with a Chinese message (the second return value).
    """
    df = _df(cur)
    col1 = (args.get("x") or args.get("column1")
            or args.get("target_column"))
    col2 = (args.get("y") or args.get("column2")
            or args.get("other_column"))
    if not col1 or not col2:
        raise ValueError("需提供 x/y 或 column1/column2")
    if (len(df) < 2 or df[col1].nunique() < 2 or
            df[col2].nunique() < 2):
        return np.nan, "样本不足计算相关系数"
    return df[col1].corr(df[col2])

@log_call(log)
def count_rows(cur, _=None): return int(len(_df(cur)))

# ---------- 4) Business-specific ----------
@log_call(log)
def calculate_delay_avg(cur,args=None):
    """
    Calculate the average of delay = (col1 - col2).
    Return the original df and write the scalar to _STATE['last_scalar'].
    """
    df = _df(cur)
    dsec = (pd.to_datetime(df[args["column1"]]) -
            pd.to_datetime(df[args["column2"]])).dt.total_seconds()

    avg_minutes = dsec.mean() / 60
    global _LAST_SCALAR
    _LAST_SCALAR = avg_minutes
    df = df.assign(delay=dsec / 60)
    return df

@log_call(log)
def calculate_failure_rate(cur,args):
    """
    Failed / total per group_column
      args={"group_column":"Machine_ID"}
    """
    df = _df(cur)
    g   = args["group_column"]
    failed = df[df["Job_Status"] == "Failed"].groupby(g).size()
    total  = df.groupby(g).size()
    return (failed / total).fillna(0).reset_index(name="failure_rate")

# --------- New addition: Average latency (grouped)-----------
@log_call(log)
def calculate_delay_avg_grouped(cur, args):
    """
    Calculate the average of delay = (col1 - col2).
    Return the original df and write the scalar to _STATE['last_scalar'].
    """
    df = _df(cur)
    dsec = (pd.to_datetime(df[args["column1"]]) -
            pd.to_datetime(df[args["column2"]])).dt.total_seconds()

    avg_minutes = dsec.mean() / 60
    global _LAST_SCALAR
    _LAST_SCALAR = avg_minutes
    df = df.assign(delay=dsec / 60)
    return df

# ---------- 5) Export / Visualization ----------
@log_call(log)
def graph_export(df, args):
    """
Construct a bipartite graph Machine_ID ↔ Job_ID (without time attribute) based on the current depth graph (df), and export it as a GEXF file.
Parameters:
    file: Output path (required)
    machine_col: Defaults to 'Machine_ID'
    job_col: Defaults to 'Job_ID'
    """
    import networkx as nx
    dst = args.get("file")
    if not isinstance(dst, str):
        raise ValueError("graph_export 需要 'file' 输出路径")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    mcol = args.get("machine_col", "Machine_ID")
    jcol = args.get("job_col", "Job_ID")
    if df is None or mcol not in df.columns or jcol not in df.columns:
        raise ValueError("graph_export: 缺少必要列或当前数据为空")

    pairs = df[[mcol, jcol]].dropna().drop_duplicates()

    G = nx.Graph()
    for m in pairs[mcol].unique():
        G.add_node(f"M::{m}", bipartite=0, label=str(m))
    for j in pairs[jcol].unique():
        G.add_node(f"J::{j}", bipartite=1, label=str(j))
    for _, row in pairs.iterrows():
        G.add_edge(f"M::{row[mcol]}", f"J::{row[jcol]}")

    # 确保没有 timeformat 之类的奇怪属性
    if "timeformat" in G.graph:
        del G.graph["timeformat"]

    nx.write_gexf(G, dst)
    return dst

# ========== New 2) plot_machine_avg_bar ==========
@log_call(log)
def plot_machine_avg_bar(df, args):
    """Plot a bar chart of the average values for the specified metrics for each machine and save it."""
    import matplotlib.pyplot as plt
    metric = _get_one(args, "metric", default="Energy_Consumption")
    dst = args.get("file", "output/avg_metric.png")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    grp = df.groupby("Machine_ID")[metric].mean().sort_values(ascending=False)

    plt.figure()
    ax = grp.plot(kind="bar")
    ax.set_xlabel("Machine_ID")
    ax.set_ylabel(f"Avg {metric}")
    ax.set_title(f"Average {metric} by Machine")
    plt.tight_layout()
    plt.savefig(dst, dpi=200)
    plt.close()
    return dst

# ========== New 3) plot_concurrent_tasks_line ==========
@log_call(log)
def plot_concurrent_tasks_line(cur, args: dict | None = None):
    """
    Plot a line graph of the number of concurrent tasks over time.
      args = {"freq": "10T", "file": "output/concurrent.png"}
    """
    import matplotlib.pyplot as plt
    df   = _df(cur)
    freq = (args or {}).get("freq", "10T")
    dst  = (args or {}).get("file", "output/concurrent_tasks.png")

    # Building a timeline (+1/-1)
    events = []
    for _, r in df.iterrows():
        events.append((pd.to_datetime(r["Actual_Start"]), +1))
        events.append((pd.to_datetime(r["Actual_End"]),   -1))
    tl = pd.DataFrame(events, columns=["time", "delta"]).set_index("time")
    series = (tl.sort_index()["delta"].cumsum()
                .resample(freq).mean().ffill())
    plt.figure()
    series.plot()
    plt.ylabel("Concurrent Jobs")
    plt.title(f"Concurrent Jobs over Time (resample={freq})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    plt.savefig(dst)
    plt.close()
    return dst

@log_call(log)
def select_columns(cur, args):
    """
    Preserve/rearrange: args["columns"] or args["pair"]
    Allows to be written as the string "Job_ID, delay"
    """
    df = _df(cur)

    # Compatible with two field names
    cols = args.get("columns") or args.get("pair")
    if cols is None:
        raise ValueError("select_columns: expect 'columns' key")

    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",") if c.strip()]

    if not cols:
        raise ValueError("select_columns: empty column list")

    return df[cols]