# RAG_eval.py

import pandas as pd
import numpy as np
import os
from RAG_tool_functions import load_data
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from logging_setup import get_logger
log = get_logger("rag.eval")

# ---------- 1. 构造“伪标签” ----------
def pseudo_labels(df: pd.DataFrame, top_q: float = 0.02) -> np.ndarray:
    """
    取每个算法分数的 top‑q 作为正样本，再对多算法结果做“少数服从多数”投票，
    生成 ensemble 伪标签。返回 0/1 ndarray。
    """
    algos = [c for c in df.columns if c not in ("time_stamp", "ensemble")]
    votes = np.zeros((len(df), len(algos)), dtype=int)
    for j, col in enumerate(algos):
        thr = np.quantile(df[col], 1 - top_q)
        votes[:, j] = (df[col] >= thr).astype(int)
    # ≥ half 算一致异常
    return (votes.sum(1) >= (len(algos) + 1)//2).astype(int)

# ---------- 2. 单算法评估 ----------
def evaluate(scores: np.ndarray, y_ref: np.ndarray) -> dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_ref, (scores >= np.quantile(scores, 0.98)).astype(int),
        average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_ref, scores)
    return dict(precision=prec, recall=rec, f1=f1, pr_auc=pr_auc)

# ---------- 3. 总入口 ----------
def run_evaluation(excel_path: str, save_fig: bool = True) -> pd.DataFrame:
    log.info("eval.start | excel=%s", excel_path)
    xls = pd.ExcelFile(excel_path)
    dfs  = [xls.parse(s) for s in xls.sheet_names]

    # 只保留含 anomaly_score 的工作表（排除 benchmark）
    algo_sheets = [(s, df) for s, df in zip(xls.sheet_names, dfs) if "anomaly_score" in df.columns]

    # 以第一个算法的 time_stamp 为主键做外连接，避免长度不一致
    merged = algo_sheets[0][1][["time_stamp"]].copy()

    for sheet, df in algo_sheets:
        merged = merged.merge(
            df[["time_stamp", "anomaly_score"]].rename(
                columns={"anomaly_score": sheet}),
            on="time_stamp", how="left"
        )

    y_ens = pseudo_labels(merged)
    merged["ensemble"] = y_ens

    # ---------- 仅保留真正的算法列 ----------
    algo_cols = [c for c in merged.columns if c not in ("time_stamp", "ensemble")]

    # ---------- 生成评估汇总 ----------
    rows = []
    for algo in algo_cols:
        rows.append({"algo": algo,
                     **evaluate(merged[algo].values, y_ens)})

    summary = pd.DataFrame(rows).round(4)

    # ---------- 4. 可视化 ----------
    if save_fig:
        plt.figure()
        for algo in algo_cols:
            p, r, _ = precision_recall_curve(y_ens, merged[algo].values)
            plt.step(r, p, where="post",
                     label=f"{algo}  AP={evaluate(merged[algo].values, y_ens)['pr_auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision‑Recall curves (ensemble pseudo‑labels)")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.splitext(excel_path)[0] + "_pr_curve.png"
        plt.savefig(fig_path, dpi=300)
        log.info("eval.pr_curve saved -> %s", fig_path)
        plt.close()

    # 把 summary & 伪标签追加写回 Excel
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as w:
        summary.to_excel(w, sheet_name="benchmark", index=False)
        merged.to_excel(w, sheet_name="merged_scores", index=False)
    log.info("eval.done | wrote sheets: benchmark, merged_scores")
    return summary

def _post_eval(state: dict) -> dict:
    excel_path = state['excel_path']
    bench_df = state['bench_summary']
    df = load_data(excel_path)  # 读取含有 anomaly_score 的文件
    if 'anomaly' in df.columns:
        plt.figure()
        for algo in bench_df['algo']:
            # 从 Excel 中读取该算法工作表
            sub = pd.read_excel(excel_path, sheet_name=algo)
            y_true = sub['anomaly'].values
            scores = sub['anomaly_score'].values
            p, r, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            plt.step(r, p, where='post', label=f"{algo}  AP={pr_auc:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision‑Recall curves (ground truth)')
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.splitext(excel_path)[0] + '_pr_curve.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        state['eval_summary'] = bench_df  # 这里直接返回评估表
        state['pr_curve'] = fig_path
    else:
        # 沿用原伪标签评估
        state['eval_summary'] = run_evaluation(excel_path)
    return state