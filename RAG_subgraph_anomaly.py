# RAG_subgraph_anomaly.py
from typing import Dict, Any, TypedDict
import os, glob, time
import pandas as pd, numpy as np
from pathlib import Path
from langgraph.graph import StateGraph
from RAG_tool_functions import load_data
from AD_algorithms import ALGOS, run_algo, build_autoencoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support, average_precision_score, f1_score, roc_auc_score
from RAG_eval import run_evaluation
from logging_setup import get_logger
import re
log = get_logger("rag.anomaly")

INVALID_SHEET_CHARS = r'[:\\/?*\[\]]'

def _ensure_unique_path(path: str) -> str:
    """若文件已存在，自动在末尾添加 _1, _2, ..."""
    base, ext = os.path.splitext(path)
    i, cand = 1, path
    while os.path.exists(cand):
        cand = f"{base}_{i}{ext}"
        i += 1
    return cand

def _basename_for_csv_or_dir(csv_path: str) -> str:
    """目录名或文件无扩展名作为基名"""
    p = Path(csv_path)
    if p.is_dir():
        return p.name
    return p.stem

def _parse_eif_params(text: str) -> dict:
    """
    从自然语言中解析 EIF 参数（可选）：
      例： "EIF(n=3, ss=256, t=800, metric=density)"
    支持键别名：
      n/ndim, ss/sample_size, t/trees/ntrees, metric/scoring_metric
    """
    text = text.lower()
    m = re.search(r"eif\s*\((.*?)\)", text)
    if not m:
        return {}
    body = m.group(1)
    params = {}
    for kv in re.split(r"[,\s]+", body):
        if "=" not in kv:
            continue
        k, v = [x.strip() for x in kv.split("=", 1)]
        if k in ("n", "ndim"):
            params["ndim"] = int(v)
        elif k in ("ss", "sample_size"):
            params["sample_size"] = int(v) if v.isdigit() else v
        elif k in ("t", "trees", "ntrees"):
            params["ntrees"] = int(v)
        elif k in ("metric", "scoring_metric"):
            params["scoring_metric"] = v
    return params

def _to_sheet_name(name: str, used: set | None = None) -> str:
    """把任意算法名变成合法的 Excel sheet 名（去非法字符、<=31 字符、避免重名）。"""
    import re
    s = re.sub(INVALID_SHEET_CHARS, '-', str(name)).strip()
    if not s:
        s = "sheet"
    s = s[:31]  # Excel 限制
    if used is not None:
        base, i = s, 1
        while s in used:
            suf = f"_{i}"
            s = (base[:31 - len(suf)] + suf)
            i += 1
        used.add(s)
    return s

ALIAS = {
    'autoencoder': 'AE', 'ae': 'AE',
    'eif': 'EIF', 'isolationforest': 'EIF',
    'lof': 'LOF', 'knnlof': 'LOF',
    'copod': 'COPOD',
    'inne': 'INNE',
    'ocsvm': 'OCSVM'
}

def parse_algos(processed_input: str):
    """
    从自然语言指令里解析出要运行的算法名称列表。
    支持 EIF, LOF, COPOD, INNE, OCSVM，大小写不敏感。
    返回值为空列表时表示使用全部算法。
    """
    text = processed_input.lower()
    if any(x in text for x in ["全部", "所有", "all"]):
        return list(ALGOS.keys())
    selected = []
    for alias, name in ALIAS.items():
        if alias in text:
            selected.append(name)
    # 去重
    return list(dict.fromkeys(selected))

def get_loss(cfg):
    if cfg["loss"] == "mse":
        return "mse"
    elif cfg["loss"] == "mae":
        return "mae"
    elif cfg["loss"] == "huber":
        return tf.keras.losses.Huber(delta=cfg["huber_delta"])
    else:
        raise ValueError("Unknown loss")

def pick_threshold(scores_normal, method="quantile", q=0.995, k=3.0):
    scores_normal = np.asarray(scores_normal)
    if method == "quantile":
        return np.quantile(scores_normal, q)
    elif method == "std":
        return scores_normal.mean() + k * scores_normal.std()
    elif method == "mad":
        med = np.median(scores_normal)
        mad = np.median(np.abs(scores_normal - med)) + 1e-12
        return med + k * 1.4826 * mad
    else:
        raise ValueError("Unknown method")

def smooth(x, k=5):
    if k <= 1: return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k) / k
    return np.convolve(xp, w, mode="valid")

def _benchmark(state: dict, top_q: float = 0.02) -> dict:
    csv_path = state['csv_path']
    log.info("anomaly.benchmark | path=%s | top_q=%.3f", csv_path, top_q)
    # 支持目录：将目录下所有 .csv 拼接
    if os.path.isdir(csv_path):
        dfs = [load_data(f) for f in sorted(glob.glob(os.path.join(csv_path, '*.csv')))]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = load_data(csv_path)
    # 保留真实标签
    y_true = df['anomaly'].values if 'anomaly' in df.columns else None
    # 选出数值特征
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # 删除标签列和 changepoint
    for dropcol in ['anomaly', 'changepoint']:
        if dropcol in feature_cols:
            feature_cols.remove(dropcol)
    X = df[feature_cols].values
    log.info("dataset | rows=%d cols(total)=%d cols(numeric)=%d has_y=%s",
             len(df), len(df.columns), len(feature_cols), 'anomaly' in df.columns)

    processed_input = state.get("processed_input", "")
    selected_algos = state.get("algorithms") or parse_algos(processed_input)
    log.info("algorithms | selected=%s (empty means all=%s)", selected_algos, list(ALGOS.keys()))
    if not selected_algos:  # 如果未解析到算法，则默认全选
        selected_algos = list(ALGOS.keys())

    rows, scores_map, tune_tables = [], {}, {}
    # 仅遍历用户选择的算法
    for name in selected_algos:
        if name not in ALGOS:
            # 提前终止：包含未支持算法
            state.update({"route": "finish", "final_answer": f"算法 '{name}' 未支持"})
            return state
        elif name == "EIF":
            # ===== EIF：单次运行（默认）；若需网格调参，文本含 'tune'/'grid' 或 EIF_TUNING=1 =====
            t0 = time.perf_counter()

            # --- 预处理配置（与原逻辑一致，可按需调）---
            eif_pre = {
                "scaler": "standard",  # "standard" | "robust" | "none"
                "use_pca": False,
                "pca_dim": 0.95,
                "smooth_k": 11,
            }

            # === 预处理 ===
            X0 = X.copy()
            if eif_pre["scaler"] == "standard":
                _scaler = StandardScaler();
                Xp = _scaler.fit_transform(X0)
            elif eif_pre["scaler"] == "robust":
                _scaler = RobustScaler();
                Xp = _scaler.fit_transform(X0)
            else:
                Xp = X0

            pca = None
            if eif_pre["use_pca"]:
                n_samples, in_dim0 = Xp.shape[:2]
                n_comp = eif_pre["pca_dim"]
                if isinstance(n_comp, float) and 0.0 < n_comp < 1.0:
                    pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
                else:
                    max_comp = max(1, min(n_samples, in_dim0) - 1)
                    n_comp_use = max(1, min(int(n_comp), max_comp))
                    pca = PCA(n_components=n_comp_use, svd_solver="auto", random_state=42)
                Xp = pca.fit_transform(Xp)

            # --- 是否走网格？默认否 ---
            want_tune = (
                    ("tune" in processed_input.lower()) or
                    ("grid" in processed_input.lower()) or
                    (os.getenv("EIF_TUNING", "0") == "1")
            )

            if want_tune:
                # === 原有『网格调参』路径（保留，必要时可用） ===
                grid_ntrees = [200, 400, 800]
                grid_sample_size = [256, 512, 'auto']
                grid_ndim = [1, 3]
                grid_metric = ["depth", "density"]
                variants = []

                for ntrees, sample_size, ndim, metric in product(grid_ntrees, grid_sample_size, grid_ndim, grid_metric):
                    if isinstance(sample_size, int) and sample_size > len(Xp):
                        continue
                    cfg_eif = dict(
                        ntrees=ntrees, sample_size=sample_size, ndim=ndim, nthreads=-1,
                        scoring_metric=metric, penalize_range=True, weigh_by_kurtosis=True,
                    )
                    if metric == "density":
                        cfg_eif["penalize_range"] = False
                        cfg_eif["weigh_by_kurtosis"] = False

                    sc = run_algo("EIF", Xp, cfg=cfg_eif)
                    sc = smooth(sc, eif_pre["smooth_k"])

                    if y_true is not None:
                        try:
                            auc_pos = roc_auc_score(y_true, sc)
                            auc_neg = roc_auc_score(y_true, -sc)
                            if np.isfinite(auc_neg) and auc_neg > auc_pos:
                                sc = -sc
                        except Exception:
                            pass

                    if y_true is not None:
                        ap = average_precision_score(y_true, sc)
                        thr = np.quantile(sc, 1 - top_q)
                        y_pred = (sc >= thr).astype(int)
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average="binary", zero_division=0)
                    else:
                        thr = np.quantile(sc, 1 - top_q)
                        y_ref = (sc >= thr).astype(int)
                        ap = average_precision_score(y_ref, sc)
                        prec = rec = f1 = np.nan

                    key = f"EIF[{metric}|n={ndim}|ss={sample_size}|t={ntrees}]"
                    variants.append((ap, key, sc, prec, rec, f1))

                variants.sort(key=lambda z: z[0], reverse=True)
                dt = time.perf_counter() - t0
                log.info("EIF grid done | variants=%d | best=%.4f | time=%.2fs",
                         len(variants), variants[0][0], dt)

                # 只保留**最佳**变体进入 scores_map/benchmark（避免曲线爆炸）
                ap, key, sc, prec, rec, f1 = variants[0]
                scores_map["EIF"] = sc
                rows.append({
                    "algo": "EIF", "seconds": round(dt, 2),
                    "precision": prec, "recall": rec, "f1": f1,
                    "pr_auc": ap, "roc_auc": (roc_auc_score(y_true, sc) if y_true is not None else np.nan)
                })
                continue

            # === 单次运行路径（默认） ===
            # 1) 从文本解析/环境变量拿参数；若都没给，给一个合理默认
            p = _parse_eif_params(processed_input)
            ntrees = int(os.getenv("EIF_NTREES", p.get("ntrees", 900)))
            ndim = int(os.getenv("EIF_NDIM", p.get("ndim", 1)))
            metric = os.getenv("EIF_METRIC", p.get("scoring_metric", p.get("metric", "density")))
            ss_raw = os.getenv("EIF_SAMPLE_SIZE", str(p.get("sample_size", "128")))
            sample_size = int(ss_raw) if str(ss_raw).isdigit() else ss_raw

            cfg_eif = dict(
                ntrees=ntrees, sample_size=sample_size, ndim=ndim,
                nthreads=-1, scoring_metric=metric,
                penalize_range=(metric != "density"),
                weigh_by_kurtosis=(metric != "density"),
            )

            sc = run_algo("EIF", Xp, cfg=cfg_eif)
            sc = smooth(sc, eif_pre["smooth_k"])

            # 分数方向自检
            if y_true is not None:
                try:
                    auc_pos = roc_auc_score(y_true, sc)
                    auc_neg = roc_auc_score(y_true, -sc)
                    if np.isfinite(auc_neg) and auc_neg > auc_pos:
                        sc = -sc
                except Exception:
                    pass

            # 计算指标（与其它算法保持一致）
            if y_true is not None:
                thr = np.quantile(sc, 1 - top_q)
                y_pred = (sc >= thr).astype(int)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0)
                ap = average_precision_score(y_true, sc)
                try:
                    roc_auc = roc_auc_score(y_true, sc)
                except ValueError:
                    roc_auc = np.nan
            else:
                thr = np.quantile(sc, 1 - top_q)
                y_ref = (sc >= thr).astype(int)
                ap = average_precision_score(y_ref, sc)
                prec = rec = f1 = roc_auc = np.nan

            dt = time.perf_counter() - t0

            # 只写入一个“EIF”结果到 map/benchmark（不再铺满变体）
            scores_map["EIF"] = sc
            rows.append({
                "algo": "EIF", "seconds": round(dt, 2),
                "precision": prec, "recall": rec,
                "f1": f1, "pr_auc": ap, "roc_auc": roc_auc
            })
            continue

        if name == "AE":
            # ===== AE 分支 =====
            t0 = time.perf_counter()

            # 1) 载入 anomaly-free 训练自编码器
            path_csv = Path(csv_path).resolve()
            if path_csv.is_file():
                base_skab_dir = path_csv.parent.parent  # …/SKAB/valveX/0.csv → 上两级是 SKAB
            else:
                base_skab_dir = path_csv.parent  # …/SKAB/valveX     → 上一级是 SKAB
            anomaly_free_file = base_skab_dir / 'anomaly-free' / 'anomaly-free.csv'
            train_df = load_data(str(anomaly_free_file))

            # 与测试列对齐，避免 anomaly-free 少/多列导致错位
            feat_set = [c for c in feature_cols if c in train_df.columns]
            X_train = train_df[feat_set].values
            X_test = df[feat_set].values

            # 2) 预处理（Scaler + 可选 PCA + 可选 滑窗）
            cfg = {
                # --- 预处理 ---
                "scaler": "standard",  # "standard" | "robust" | "quantile"
                "use_pca": True,  # 先不开；若想试：True + pca_dim=64/128 或 95%方差
                "pca_dim": 32,  # 0.95, 32, 64, 128

                # --- 窗口 ---
                "use_windows": False,  # 默认关闭；需要时再扫。(50,1), (100,1), (200,1), (100,5)
                "win_len": 64,
                "win_stride": 1,

                # --- 结构 ---
                "hidden_units": [256, 128, 8],  # 32, 24, 16， 12， 8
                "activation": "relu",  # "relu" | "elu" | "leaky_relu"
                "use_batchnorm": True,
                "dropout_rate": 0.0,
                "output_activation": "linear",
                "sparse_l1": 0.0,
                "denoise_sigma": 0.0,

                # --- 训练 ---
                "loss": "mae",  # 你实验证明 MAE 最好
                "huber_delta": 1.0,
                "lr": 1e-3,
                "epochs": 200,
                "batch_size": 256,
                "earlystop_patience": 10,
                "reduce_on_plateau": True,
                "rop_factor": 0.2,
                "rop_patience": 5,

                # --- 后处理/阈值（用于参考，可与下方 top_q 分开） ---
                "smooth_k": 11,  # {1, 5, 11}
                "thr_method": "quantile",  # "quantile" | "std" | "mad" | "pot"
                "thr_q": 0.995,  # 分位法的 q
                "pot_q0": 0.98,  # POT：基线阈 u 的分位（先取 0.98）
                "pot_alpha": 1e-3  # POT：尾部风险水平
            }

            # 2.1 Scaler
            if cfg["scaler"] == "standard":
                scaler = StandardScaler()
            elif cfg["scaler"] == "robust":
                scaler = RobustScaler()
            elif cfg["scaler"] == "quantile":
                scaler = QuantileTransformer(output_distribution="normal",
                                             subsample=10_000, random_state=42)
            else:
                raise ValueError("Unknown scaler")

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 2.2 PCA（可选）
            pca = None
            if cfg["use_pca"]:
                n_samples, in_dim0 = X_train_scaled.shape[:2]

                # 允许两种写法：
                # ① 传整数（目标维度）→ 自动夹到合法范围 [1, min(n_samples, in_dim0)-1]
                # ② 传 0~1 的浮点数（解释为保留方差比例，如 0.95）
                n_comp = cfg["pca_dim"]

                if isinstance(n_comp, float) and 0.0 < n_comp < 1.0:
                    # 方差占比写法，交给 sklearn 自动估计
                    pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
                else:
                    # 整数维度写法：自动夹逼，避免 "must be between 0 and min(n_samples, n_features)" 的报错
                    max_comp = max(1, min(n_samples, in_dim0) - 1)  # -1 避免满秩导致求解失败
                    if n_comp is None:
                        n_comp_use = min(in_dim0, max_comp)
                    else:
                        n_comp_use = int(n_comp)
                        n_comp_use = max(1, min(n_comp_use, max_comp))
                    pca = PCA(n_components=n_comp_use, svd_solver="auto", random_state=42)

                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)

            # 2.3 滑窗（可选）：把 [T, D] → [N, L*D]
            def make_windows(X2d, L=100, stride=1):
                T, D = X2d.shape
                if T < L:  # 太短，返回 None 触发回退
                    return None, None
                idx = np.arange(0, T - L + 1, stride)
                Xw = np.stack([X2d[i:i + L].reshape(-1) for i in idx], axis=0)
                return Xw, idx

            if cfg["use_windows"]:
                X_train_in, _ = make_windows(X_train_scaled, L=cfg["win_len"], stride=cfg["win_stride"])
                X_test_in, idx_te = make_windows(X_test_scaled, L=cfg["win_len"], stride=cfg["win_stride"])
                # 若序列过短或没形成窗口，自动回退到逐点
                if X_train_in is None or X_test_in is None:
                    cfg["use_windows"] = False
                    X_train_in, X_test_in = X_train_scaled, X_test_scaled
                    idx_te = None
            else:
                X_train_in, X_test_in = X_train_scaled, X_test_scaled
                idx_te = None

            # 3) 构建 AE（注意：输入维取窗口化后的）
            input_dim = X_train_in.shape[1]
            autoencoder = build_autoencoder(input_dim, cfg)

            # 4) 编译
            opt = Adam(learning_rate=cfg["lr"])  # Keras 3: 用 learning_rate（不是 lr）
            if not hasattr(opt, "lr"):  # 兼容少数回调/日志读取 .lr
                opt.lr = opt.learning_rate
            autoencoder.compile(optimizer=opt, loss=get_loss(cfg))

            # 5) 训练（早停 + 降学习率）
            callbacks = [
                EarlyStopping(monitor="val_loss",
                              patience=cfg["earlystop_patience"],
                              restore_best_weights=True)
            ]
            if cfg["reduce_on_plateau"]:
                callbacks.append(
                    ReduceLROnPlateau(monitor="val_loss",
                                      factor=cfg["rop_factor"],
                                      patience=cfg["rop_patience"],
                                      min_lr=1e-6,
                                      verbose=1)
                )
            autoencoder.fit(
                X_train_in, X_train_in,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                shuffle=True,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0
            )

            # 6) 打分（重构误差）
            recon = autoencoder.predict(X_test_in, verbose=0)
            err = np.mean((X_test_in - recon) ** 2, axis=1)

            # 窗口回投到逐点
            if cfg["use_windows"] and idx_te is not None and len(err) > 0:
                T = X_test_scaled.shape[0]
                scores = np.zeros(T, dtype=float)
                counts = np.zeros(T, dtype=float)
                L = cfg["win_len"]
                for j, start in enumerate(idx_te):
                    end = start + L
                    if start >= T: break
                    end = min(end, T)
                    scores[start:end] += err[j]
                    counts[start:end] += 1.0
                scores = scores / np.maximum(counts, 1.0)
            else:
                scores = err

            # 7) 可选：平滑
            scores = smooth(scores, cfg["smooth_k"])

            # 8) 分数方向自检（避免 AUROC≈0.5 的“反号”问题）
            if y_true is not None:
                try:
                    auc_pos = roc_auc_score(y_true, scores)
                    auc_neg = roc_auc_score(y_true, -scores)
                    if np.isfinite(auc_neg) and auc_neg > auc_pos:
                        scores = -scores
                except Exception:
                    pass

            # 9) 参考阈值（不影响下面统一的 top_q 评估）
            if cfg["thr_method"] == "pot":
                try:
                    # 注意：不要在这里 'import numpy as np'，否则会把 np 变成本函数的局部变量
                    from scipy.stats import genpareto as _gpd
                    u = np.quantile(scores, cfg["pot_q0"])
                    excess = scores[scores > u] - u
                    # 仅拟合尾部（loc 固定为 0）
                    c, loc, scale = _gpd.fit(excess, floc=0)
                    pot_thr = u + _gpd.ppf(1 - cfg["pot_alpha"], c, loc=0, scale=scale)
                except Exception:
                    # scipy 不可用或尾部太短时，回退到分位数阈值
                    pot_thr = np.quantile(scores, cfg["thr_q"])
                # y_pred_ref = (scores >= pot_thr).astype(int)  # 如需参考标签可解开
            else:
                _ = pick_threshold(scores, method=cfg["thr_method"], q=cfg["thr_q"])

            dt = time.perf_counter() - t0

        elif name == "LOF":
            # ==== LOF：PCA 参数更易调试 ====
            t0 = time.perf_counter()

            # 方式 A：用预设（推荐）：通过环境变量 LOF_PCA_PRESET 选择
                #  - "off"    : 关闭 PCA
                #  - "var95"  : n_components=0.95（默认）
                #  - "dim64"  : 固定 64 维（会自动夹到合法维度）
                #  - "rand64" : 随机 SVD + 64 维（大样本更快）
            preset = os.getenv("LOF_PCA_PRESET", "rand64").lower()

            if preset == "off":
                lof_cfg = {"pca": {"on": False}}
            elif preset == "dim64":
                lof_cfg = {"pca": {"on": True, "n_components": 64, "svd_solver": "auto", "whiten": False}}
            elif preset == "rand64":
                lof_cfg = {"pca": {"on": True, "n_components": 64, "svd_solver": "randomized",
                                   "random_state": 42, "whiten": False}}
            else:  # 默认 "var95"
                lof_cfg = {"pca": {"on": True, "n_components": 0.95, "svd_solver": "auto", "whiten": False}}

            # 方式 B：想完全手动可直接改这块：
            # lof_cfg = {"pca": {"on": True, "n_components": 32, "svd_solver": "auto", "whiten": False}}


            scores = run_algo("LOF", X, cfg=lof_cfg)
            dt = time.perf_counter() - t0

        else:
            # ==== 其它算法 ====
            t0 = time.perf_counter()
            scores = run_algo(name, X)  # 非 LOF/EIF/AE 保持原样
            dt = time.perf_counter() - t0

        scores_map[name] = scores

        if y_true is not None:
            # 根据分位数 threshold 计算预测标签
            thr = np.quantile(scores, 1 - top_q)
            y_pred = (scores >= thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            pr_auc = average_precision_score(y_true, scores)
            try:
                roc_auc = roc_auc_score(y_true, scores)
            except ValueError:
                roc_auc = np.nan
            rows.append({
                "algo": name, "seconds": round(dt, 2),
                "precision": prec, "recall": rec,
                "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc
            })
        else:
            # 无真实标签时仅输出 pr_auc
            thr = np.quantile(scores, 1 - top_q)
            y_ref = (scores >= thr).astype(int)
            pr_auc = average_precision_score(y_ref, scores)
            rows.append({
                "algo": name,
                "seconds": round(dt, 2),
                "pr_auc": pr_auc
            })

    bench_df = pd.DataFrame(rows)
    # 在选定算法内挑选 pr_auc 或 f1 最高的模型
    best = bench_df.sort_values('pr_auc', ascending=False)['algo'].iloc[0]
    log.info("best algo=%s", best)
    df['anomaly_score'] = scores_map[best]

    # 导出结果到 Excel
    outdir = Path(state.get("output_dir", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    base_name = _basename_for_csv_or_dir(csv_path)
    excel_path = _ensure_unique_path(str(outdir / f"{base_name}_anomaly_results.xlsx"))
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as w:
        sheet_map = {}
        used = set()

        # 每个算法/变体一张 sheet（名字先规整）
        for algo, sc in scores_map.items():
            sheet = _to_sheet_name(algo, used)
            sheet_map[algo] = sheet
            tmp = pd.DataFrame({
                'time_stamp': df['time_stamp'].astype(str) if 'time_stamp' in df else df.index.astype(str),
                'anomaly_score': sc,
                'anomaly': df['anomaly'] if 'anomaly' in df.columns else None
            })
            tmp.to_excel(w, sheet_name=sheet, index=False)

        # 在 benchmark 里附带 sheet 名，便于 post_eval 读取
        bench_out = bench_df.copy()
        bench_out['sheet'] = bench_out['algo'].map(sheet_map)
        bench_out.to_excel(w, sheet_name='benchmark', index=False)

        # 若跑了 EIF 的网格调参，把结果也写入
        if "EIF" in tune_tables:
            tune_tables["EIF"].to_excel(w, sheet_name="EIF_tuning", index=False)

    # 末尾写回：
    state['sheet_map'] = sheet_map
    state['execution_output'] = df.sort_values('anomaly_score', ascending=False).head(5)[['time_stamp', 'anomaly_score']]
    state['bench_summary'] = bench_df
    state['picked_algo'] = best
    state['excel_path'] = excel_path
    log.info("excel saved -> %s | sheets=%d", excel_path, len(scores_map) + 1)
    return state

def _post_eval(state: dict) -> dict:
    """
    读取 benchmark 写出的 Excel，再根据是否有真实标签绘制 PR 曲线。
    如果存在 anomaly 列，则用真实标签计算 PR‑AUC；
    否则调用现有 run_evaluation() 生成伪标签评估。
    """
    excel_path = state["excel_path"]
    bench_df = state.get("bench_summary")
    # 打开 Excel 文件，检查是否有 'anomaly' 列
    xls = pd.ExcelFile(excel_path)
    # 从 state 取回写 Excel 时保存的映射（若没有就用规整函数重建）
    sheet_map = state.get("sheet_map", {})

    def _sheet_for(algo: str) -> str:
        return sheet_map.get(algo, _to_sheet_name(algo))

    # 用映射后的第一个 sheet 试探是否含有真实标签列
    first_sheet = _sheet_for(bench_df["algo"].iloc[0])
    sheet0 = pd.read_excel(xls, sheet_name=first_sheet)
    if "anomaly" in sheet0.columns:
        plt.figure()
        topk = 6
        plot_list = bench_df.sort_values("pr_auc", ascending=False)["algo"].head(topk).tolist()
        for algo in bench_df["algo"]:
            df_algo = pd.read_excel(xls, sheet_name=_sheet_for(algo))
            y_true = df_algo["anomaly"].values
            scores = df_algo["anomaly_score"].values
            p, r, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            plt.step(r, p, where="post", label=f"{algo}  AP={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curves (ground truth)")
        plt.legend()
        plt.tight_layout()
        pr_path = os.path.splitext(excel_path)[0] + "_pr_curve.png"
        plt.savefig(pr_path, dpi=300)
        plt.close()

        # (1) F1 曲线
        plt.figure()
        for algo in bench_df["algo"]:
            df_algo = pd.read_excel(xls, sheet_name=_sheet_for(algo))
            y_true = df_algo["anomaly"].values
            scores = df_algo["anomaly_score"].values
            p, r, _ = precision_recall_curve(y_true, scores)
            f1 = 2 * p * r / (p + r + 1e-9)
            plt.step(r, f1, where="post", label=f"{algo}  maxF1={np.nanmax(f1):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("F1")
        plt.title("F1-Recall curves (ground truth)")
        plt.legend()
        f1_path = os.path.splitext(excel_path)[0] + "_f1_curve.png"
        plt.tight_layout()
        plt.savefig(f1_path, dpi=300)
        plt.close()

        # (2) ROC 曲线
        plt.figure()
        for algo in bench_df["algo"]:
            df_algo = pd.read_excel(xls, sheet_name=_sheet_for(algo))
            y_true = df_algo["anomaly"].values
            scores = df_algo["anomaly_score"].values
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{algo}  AUC={roc_auc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves (ground truth)")
        plt.legend()
        roc_path = os.path.splitext(excel_path)[0] + "_roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()

        state["pr_curve"] = pr_path
        state["f1_curve"] = f1_path
        state["roc_curve"] = roc_path
        log.info("plots saved | pr=%s | f1=%s | roc=%s", pr_path, f1_path, roc_path)
        return state

    # 没有真实标签时，走原来的伪标签评估逻辑
    summary = run_evaluation(excel_path)
    state["eval_summary"] = summary
    return state

def build_anomaly_subgraph():
    sg = StateGraph(dict)
    sg.add_node("benchmark", _benchmark)
    sg.add_node("post_eval", _post_eval)
    sg.add_edge("benchmark", "post_eval")
    sg.set_entry_point("benchmark")
    sg.set_finish_point("post_eval")
    return sg.compile()
