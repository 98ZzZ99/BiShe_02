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
    """If the file already exists, automatically append _1, _2, ... to the end."""
    base, ext = os.path.splitext(path)
    i, cand = 1, path
    while os.path.exists(cand):
        cand = f"{base}_{i}{ext}"
        i += 1
    return cand

def _basename_for_csv_or_dir(csv_path: str) -> str:
    """The directory name or file name without an extension is used as the base name."""
    p = Path(csv_path)
    if p.is_dir():
        return p.name
    return p.stem

def _parse_eif_params(text: str) -> dict:
    """
    Parsing EIF parameters from natural language (optional):
      e.g.： "EIF(n=3, ss=256, t=800, metric=density)"
    Support key aliases：
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
    """Convert any algorithm name into a valid Excel sheet name (remove illegal characters, characters <= 31, and avoid duplicate names)."""
    import re
    s = re.sub(INVALID_SHEET_CHARS, '-', str(name)).strip()
    if not s:
        s = "sheet"
    s = s[:31]  # Excel Limitations
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
    Parse a list of algorithm names to be run from natural language instructions.
    Supports EIF, LOF, COPOD, INNE, and OCSVM; case-insensitive.
    An empty list indicates that all algorithms are used.
    """
    text = processed_input.lower()
    if any(x in text for x in ["全部", "所有", "all"]):
        return list(ALGOS.keys())
    selected = []
    for alias, name in ALIAS.items():
        if alias in text:
            selected.append(name)
    # Deduplication
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
    # Supports directories: Concatenates all .csv files in a directory.
    if os.path.isdir(csv_path):
        dfs = [load_data(f) for f in sorted(glob.glob(os.path.join(csv_path, '*.csv')))]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = load_data(csv_path)
    # Preserve authentic labels
    y_true = df['anomaly'].values if 'anomaly' in df.columns else None
    # Select numerical features
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Delete label column and changepoint
    for dropcol in ['anomaly', 'changepoint']:
        if dropcol in feature_cols:
            feature_cols.remove(dropcol)
    X = df[feature_cols].values
    log.info("dataset | rows=%d cols(total)=%d cols(numeric)=%d has_y=%s",
             len(df), len(df.columns), len(feature_cols), 'anomaly' in df.columns)

    processed_input = state.get("processed_input", "")
    selected_algos = state.get("algorithms") or parse_algos(processed_input)
    log.info("algorithms | selected=%s (empty means all=%s)", selected_algos, list(ALGOS.keys()))
    if not selected_algos:  # If no algorithm is found, all will be selected by default.
        selected_algos = list(ALGOS.keys())

    rows, scores_map, tune_tables = [], {}, {}
    # If no algorithm is found, all will be selected by default.
    for name in selected_algos:
        if name not in ALGOS:
            # Early termination: Includes unsupported algorithms
            state.update({"route": "finish", "final_answer": f"算法 '{name}' 未支持"})
            return state
        elif name == "EIF":
            # ===== EIF: Single run (default); if grid tuning is required, include 'tune'/'grid' in the text or EIF_TUNING=1 =====
            t0 = time.perf_counter()

            # --- Preprocessing configuration (consistent with the original logic, can be adjusted as needed)---
            eif_pre = {
                "scaler": "standard",  # "standard" | "robust" | "none"
                "use_pca": False,
                "pca_dim": 0.95,
                "smooth_k": 11,
            }

            # === Preprocessing ===
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

            # --- Should we use the grid? Default: No. ---
            want_tune = (
                    ("tune" in processed_input.lower()) or
                    ("grid" in processed_input.lower()) or
                    (os.getenv("EIF_TUNING", "0") == "1")
            )

            if want_tune:
                # === The existing 'mesh parameter tuning' path is retained and can be used when necessary. ===
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

                # Only the **best** variants are included in scores_map/benchmark (to avoid curve explosion).
                ap, key, sc, prec, rec, f1 = variants[0]
                scores_map["EIF"] = sc
                rows.append({
                    "algo": "EIF", "seconds": round(dt, 2),
                    "precision": prec, "recall": rec, "f1": f1,
                    "pr_auc": ap, "roc_auc": (roc_auc_score(y_true, sc) if y_true is not None else np.nan)
                })
                continue

            # === Single run path (default) ===
            # 1) Retrieve parameters from text parsing/environment variables; if neither is provided, assign a reasonable default.
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

            # Fractional direction self-test
            if y_true is not None:
                try:
                    auc_pos = roc_auc_score(y_true, sc)
                    auc_neg = roc_auc_score(y_true, -sc)
                    if np.isfinite(auc_neg) and auc_neg > auc_pos:
                        sc = -sc
                except Exception:
                    pass

            # Calculation metrics (consistent with other algorithms)
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

            # Write only one "EIF" result to map/benchmark (no longer fill the entire variant).
            scores_map["EIF"] = sc
            rows.append({
                "algo": "EIF", "seconds": round(dt, 2),
                "precision": prec, "recall": rec,
                "f1": f1, "pr_auc": ap, "roc_auc": roc_auc
            })
            continue

        if name == "AE":
            # ===== AE branch =====
            t0 = time.perf_counter()

            # 1) Load anomaly-free trained autoencoder
            path_csv = Path(csv_path).resolve()
            if path_csv.is_file():
                base_skab_dir = path_csv.parent.parent  # …/SKAB/valveX/0.csv → The two levels above are SKAB
            else:
                base_skab_dir = path_csv.parent  # …/SKAB/valveX     → The one level above is SKAB
            anomaly_free_file = base_skab_dir / 'anomaly-free' / 'anomaly-free.csv'
            train_df = load_data(str(anomaly_free_file))

            # Align with the test column to avoid misalignment caused by missing/excessive columns in anomaly-free systems.
            feat_set = [c for c in feature_cols if c in train_df.columns]
            X_train = train_df[feat_set].values
            X_test = df[feat_set].values

            # 2) Preprocessing (Scaler + optional PCA + optional sliding window)
            cfg = {
                # --- Preprocessing ---
                "scaler": "standard",  # "standard" | "robust" | "quantile"
                "use_pca": True,  # 先不开；若想试：True + pca_dim=64/128 或 95%方差
                "pca_dim": 32,  # 0.95, 32, 64, 128

                # --- Window ---
                "use_windows": False,  # 默认关闭；需要时再扫。(50,1), (100,1), (200,1), (100,5)
                "win_len": 64,
                "win_stride": 1,

                # --- Structure ---
                "hidden_units": [256, 128, 8],  # 32, 24, 16， 12， 8
                "activation": "relu",  # "relu" | "elu" | "leaky_relu"
                "use_batchnorm": True,
                "dropout_rate": 0.0,
                "output_activation": "linear",
                "sparse_l1": 0.0,
                "denoise_sigma": 0.0,

                # --- Training ---
                "loss": "mae",  # Your experiments have shown that MAE is the best.
                "huber_delta": 1.0,
                "lr": 1e-3,
                "epochs": 200,
                "batch_size": 256,
                "earlystop_patience": 10,
                "reduce_on_plateau": True,
                "rop_factor": 0.2,
                "rop_patience": 5,

                # --- Post-processing/threshold (for reference only, may be separated from top_q below) ---
                "smooth_k": 11,  # {1, 5, 11}
                "thr_method": "quantile",  # "quantile" | "std" | "mad" | "pot"
                "thr_q": 0.995,  # q of the quantile method
                "pot_q0": 0.98,  # POT: Quantile of baseline threshold u (starting with 0.98)
                "pot_alpha": 1e-3  # POT: Tail Risk Level
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

            # 2.2 PCA（optinal）
            pca = None
            if cfg["use_pca"]:
                n_samples, in_dim0 = X_train_scaled.shape[:2]

                # Two ways to write this are allowed:
                # ① Pass an integer (target dimension) → Automatically squeezed into the valid range [1, min(n_samples, in_dim0)-1]
                # ② Pass a floating-point number between 0 and 1 (interpreted as preserving the variance proportion, such as 0.95)
                n_comp = cfg["pca_dim"]

                if isinstance(n_comp, float) and 0.0 < n_comp < 1.0:
                    # The variance proportion is written and automatically estimated by sklearn.
                    pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
                else:
                    # Integer dimension syntax: Automatically squeezes to avoid "must be between 0 and min(n_samples, n_features)" error.
                    max_comp = max(1, min(n_samples, in_dim0) - 1)  # -1 to avoid the solution failing due to full rank.
                    if n_comp is None:
                        n_comp_use = min(in_dim0, max_comp)
                    else:
                        n_comp_use = int(n_comp)
                        n_comp_use = max(1, min(n_comp_use, max_comp))
                    pca = PCA(n_components=n_comp_use, svd_solver="auto", random_state=42)

                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)

            # 2.3 Sliding window (optional): Change [T, D] to [N, L*D]
            def make_windows(X2d, L=100, stride=1):
                T, D = X2d.shape
                if T < L:  # Too short, return None to trigger a fallback.
                    return None, None
                idx = np.arange(0, T - L + 1, stride)
                Xw = np.stack([X2d[i:i + L].reshape(-1) for i in idx], axis=0)
                return Xw, idx

            if cfg["use_windows"]:
                X_train_in, _ = make_windows(X_train_scaled, L=cfg["win_len"], stride=cfg["win_stride"])
                X_test_in, idx_te = make_windows(X_test_scaled, L=cfg["win_len"], stride=cfg["win_stride"])
                # If the sequence is too short or does not form a window, it will automatically revert to point-by-point.
                if X_train_in is None or X_test_in is None:
                    cfg["use_windows"] = False
                    X_train_in, X_test_in = X_train_scaled, X_test_scaled
                    idx_te = None
            else:
                X_train_in, X_test_in = X_train_scaled, X_test_scaled
                idx_te = None

            # 3) Construct the AE (Note: the input dimensions are the windowed result).
            input_dim = X_train_in.shape[1]
            autoencoder = build_autoencoder(input_dim, cfg)

            # 4) Compilation
            opt = Adam(learning_rate=cfg["lr"])  # Keras 3: use learning_rate（not lr）
            if not hasattr(opt, "lr"):  # 兼Allows for a few callbacks/log readings. .lr
                opt.lr = opt.learning_rate
            autoencoder.compile(optimizer=opt, loss=get_loss(cfg))

            # 5) Training (early stop + reduce learning rate)
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

            # 6) Scoring (Reconstruction Error)
            recon = autoencoder.predict(X_test_in, verbose=0)
            err = np.mean((X_test_in - recon) ** 2, axis=1)

            # Window recast to point
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

            # 7) Optional: Smooth
            scores = smooth(scores, cfg["smooth_k"])

            # 8) Self-check the fraction direction (to avoid the "reverse sign" problem when AUROC≈0.5)
            if y_true is not None:
                try:
                    auc_pos = roc_auc_score(y_true, scores)
                    auc_neg = roc_auc_score(y_true, -scores)
                    if np.isfinite(auc_neg) and auc_neg > auc_pos:
                        scores = -scores
                except Exception:
                    pass

            # 9) Reference threshold (does not affect the unified top_q evaluation below)
            if cfg["thr_method"] == "pot":
                try:
                    # Note: Do not 'import numpy as np' here, otherwise np will become a local variable of this function.
                    from scipy.stats import genpareto as _gpd
                    u = np.quantile(scores, cfg["pot_q0"])
                    excess = scores[scores > u] - u
                    # Fit only the tail (loc is fixed to 0)
                    c, loc, scale = _gpd.fit(excess, floc=0)
                    pot_thr = u + _gpd.ppf(1 - cfg["pot_alpha"], c, loc=0, scale=scale)
                except Exception:
                    # If scipy is unavailable or the tail is too short, fall back to the quantile threshold.
                    pot_thr = np.quantile(scores, cfg["thr_q"])
                # y_pred_ref = (scores >= pot_thr).astype(int)  # 如需参考标签可解开
            else:
                _ = pick_threshold(scores, method=cfg["thr_method"], q=cfg["thr_q"])

            dt = time.perf_counter() - t0

        elif name == "LOF":
            # ==== LOF: PCA parameters are easier to adjust ====
            t0 = time.perf_counter()

            # Method A: Use the default (recommended): Select via environment variable LOF_PCA_PRESET
                #  - "off"    : Close PCA
                #  - "var95"  : n_components=0.95 (default)
                #  - "dim64"  : Fixed 64 dimensions (will automatically be clipped to valid dimensions)
                #  - "rand64" : Random SVD + 64 dimensions (faster for large samples)
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

            # Method B: If you want to do it completely manually, you can directly modify this part:
            # lof_cfg = {"pca": {"on": True, "n_components": 32, "svd_solver": "auto", "whiten": False}}


            scores = run_algo("LOF", X, cfg=lof_cfg)
            dt = time.perf_counter() - t0

        else:
            # ==== Other algorithms ====
            t0 = time.perf_counter()
            scores = run_algo(name, X)  # Non-LOF/EIF/AE formats should remain unchanged.
            dt = time.perf_counter() - t0

        scores_map[name] = scores

        if y_true is not None:
            # Calculate predicted labels based on quantile thresholds
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
            # When there are no real labels, only pr_auc is output.
            thr = np.quantile(scores, 1 - top_q)
            y_ref = (scores >= thr).astype(int)
            pr_auc = average_precision_score(y_ref, scores)
            rows.append({
                "algo": name,
                "seconds": round(dt, 2),
                "pr_auc": pr_auc
            })

    bench_df = pd.DataFrame(rows)
    # Select the model with the highest pr_auc or f1 score from the chosen algorithms.
    best = bench_df.sort_values('pr_auc', ascending=False)['algo'].iloc[0]
    log.info("best algo=%s", best)
    df['anomaly_score'] = scores_map[best]

    # Export results to Excel
    outdir = Path(state.get("output_dir", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    base_name = _basename_for_csv_or_dir(csv_path)
    excel_path = _ensure_unique_path(str(outdir / f"{base_name}_anomaly_results.xlsx"))
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as w:
        sheet_map = {}
        used = set()

        # One sheet per algorithm/variant (names should be organized first).
        for algo, sc in scores_map.items():
            sheet = _to_sheet_name(algo, used)
            sheet_map[algo] = sheet
            tmp = pd.DataFrame({
                'time_stamp': df['time_stamp'].astype(str) if 'time_stamp' in df else df.index.astype(str),
                'anomaly_score': sc,
                'anomaly': df['anomaly'] if 'anomaly' in df.columns else None
            })
            tmp.to_excel(w, sheet_name=sheet, index=False)

        # Include sheet names in the benchmark to facilitate post_eval reading.
        bench_out = bench_df.copy()
        bench_out['sheet'] = bench_out['algo'].map(sheet_map)
        bench_out.to_excel(w, sheet_name='benchmark', index=False)

        # If you ran the EIF mesh parameter tuning, also write the results to...
        if "EIF" in tune_tables:
            tune_tables["EIF"].to_excel(w, sheet_name="EIF_tuning", index=False)

    # Write back at the end:
    state['sheet_map'] = sheet_map
    state['execution_output'] = df.sort_values('anomaly_score', ascending=False).head(5)[['time_stamp', 'anomaly_score']]
    state['bench_summary'] = bench_df
    state['picked_algo'] = best
    state['excel_path'] = excel_path
    log.info("excel saved -> %s | sheets=%d", excel_path, len(scores_map) + 1)
    return state

def _post_eval(state: dict) -> dict:
    """
    Read the Excel file generated by the benchmark and then plot the PR curve based on whether there are real labels.
    If an anomaly column exists, calculate the PR-AUC using the real labels;
    Otherwise, call the existing run_evaluation() function to generate a pseudo-label evaluation.
    """
    excel_path = state["excel_path"]
    bench_df = state.get("bench_summary")
    # Open the Excel file and check if there is a 'anomaly' column.
    xls = pd.ExcelFile(excel_path)
    # Retrieve the mapping saved when writing Excel from the state (rebuild it using a regularization function if it doesn't exist).
    sheet_map = state.get("sheet_map", {})

    def _sheet_for(algo: str) -> str:
        return sheet_map.get(algo, _to_sheet_name(algo))

    # Use the first sheet after mapping to test whether it contains a real label column.
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

        # (1) F1 Cureve
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

        # (2) ROC Cureve
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

    # When there are no real labels, the original pseudo-label evaluation logic applies.
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
