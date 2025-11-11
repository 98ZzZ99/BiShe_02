# RAG_eval.py

import pandas as pd
import numpy as np
import os
from RAG_tool_functions import load_data
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from logging_setup import get_logger
log = get_logger("rag.eval")

# ---------- 1. Constructing "Pseudo-Labels" ----------
def pseudo_labels(df: pd.DataFrame, top_q: float = 0.02) -> np.ndarray:
    """
    Take the top-q scores of each algorithm as positive samples, then perform a majority vote on the results of multiple algorithms to generate ensemble pseudo-labels. Return a 0/1 ndarray.
    """
    algos = [c for c in df.columns if c not in ("time_stamp", "ensemble")]
    votes = np.zeros((len(df), len(algos)), dtype=int)
    for j, col in enumerate(algos):
        thr = np.quantile(df[col], 1 - top_q)
        votes[:, j] = (df[col] >= thr).astype(int)
    # ≥ half is considered a consistency anomaly
    return (votes.sum(1) >= (len(algos) + 1)//2).astype(int)

# ---------- 2. Single Algorithm Evaluation ----------
def evaluate(scores: np.ndarray, y_ref: np.ndarray) -> dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_ref, (scores >= np.quantile(scores, 0.98)).astype(int),
        average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_ref, scores)
    return dict(precision=prec, recall=rec, f1=f1, pr_auc=pr_auc)

# ---------- 3. Main Entrance ----------
def run_evaluation(excel_path: str, save_fig: bool = True) -> pd.DataFrame:
    log.info("eval.start | excel=%s", excel_path)
    xls = pd.ExcelFile(excel_path)
    dfs  = [xls.parse(s) for s in xls.sheet_names]

    # Keep only the worksheets containing anomaly_score (excluding benchmark).
    algo_sheets = [(s, df) for s, df in zip(xls.sheet_names, dfs) if "anomaly_score" in df.columns]

    # Use the time_stamp from the first algorithm as the primary key for the outer join to avoid inconsistent lengths.
    merged = algo_sheets[0][1][["time_stamp"]].copy()

    for sheet, df in algo_sheets:
        merged = merged.merge(
            df[["time_stamp", "anomaly_score"]].rename(
                columns={"anomaly_score": sheet}),
            on="time_stamp", how="left"
        )

    y_ens = pseudo_labels(merged)
    merged["ensemble"] = y_ens

    # ---------- Only keep the actual algorithm columns ----------
    algo_cols = [c for c in merged.columns if c not in ("time_stamp", "ensemble")]

    # ---------- Generate evaluation summary ----------
    rows = []
    for algo in algo_cols:
        rows.append({"algo": algo,
                     **evaluate(merged[algo].values, y_ens)})

    summary = pd.DataFrame(rows).round(4)

    # ---------- 4. Visualization ----------
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

    # Write the summary and pseudo-tags back to Excel.
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as w:
        summary.to_excel(w, sheet_name="benchmark", index=False)
        merged.to_excel(w, sheet_name="merged_scores", index=False)
    log.info("eval.done | wrote sheets: benchmark, merged_scores")
    return summary

def _post_eval(state: dict) -> dict:
    excel_path = state['excel_path']
    bench_df = state['bench_summary']
    df = load_data(excel_path)  # Reading files containing anomaly_score
    if 'anomaly' in df.columns:
        plt.figure()
        for algo in bench_df['algo']:
            # Read the algorithm worksheet from Excel.
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
        # Continue using the original pseudo-label evaluation
        state['eval_summary'] = run_evaluation(excel_path)
    return state