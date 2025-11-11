# AD_algorithms.py
"""
Unified management of the anomaly detection models to be run
EIF      – isotree.IsolationForest
LOF      – PyOD kNN‑LOF
COPOD    – PyOD COPOD
INNE     – PyOD INNE
OCSVM    – PyOD One‑ClassSVM
"""

# --- ① 依赖导入
from typing import Dict, Callable
import numpy as np
import pandas as pd
from isotree import IsolationForest          # Extended IF (EIF)
from pyod.models.lof import LOF              # kNN/LOF
from pyod.models.copod import COPOD          # COPOD
from pyod.models.ocsvm import OCSVM          # One‑Class SVM
from pyod.models.inne import INNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import logging
log = logging.getLogger("ad")
from RAG_tool_functions import load_data

def _apply_pca_with_guard(X: np.ndarray, pca_cfg: dict | None) -> np.ndarray:
    """
    Apply PCA safely (supports n_components as floating-point or integer from 0 to 1).
    - Floating-point (0,1): Variance percentage
    - Integer: Automatically clipped to [1, min(n_samples, n_features)-1] to avoid out-of-bounds errors.
    """
    if not pca_cfg or not pca_cfg.get("on", False):
        return X

    n_samples, n_features = X.shape
    req = pca_cfg.get("n_components", 0.95)

    if isinstance(req, float) and 0.0 < req < 1.0:
        n_comp = req
    else:
        try:
            n_int = int(req)
        except (TypeError, ValueError):
            n_int = n_features
        max_comp = max(1, min(n_samples, n_features) - 1)
        n_comp = max(1, min(n_int, max_comp))

    svd_solver   = pca_cfg.get("svd_solver", "auto")
    whiten       = bool(pca_cfg.get("whiten", False))
    random_state = pca_cfg.get("random_state", None)

    log.debug("PCA | on=%s n_components=%s->%s solver=%s whiten=%s",
              True, req, n_comp, svd_solver, whiten)

    pca = PCA(n_components=n_comp, svd_solver=svd_solver,
              whiten=whiten, random_state=random_state)
    return pca.fit_transform(X)

# --- ② Factory function — Returns an object that has already been fitted using .fit() and has the .decision_scores_ property.
def _wrap_pyod(cls, **kw):
    """Encapsulating PyOD models into "parameterless constructors" facilitates lazy instantiation."""
    return lambda: cls(**kw)

ALGOS: Dict[str, Callable[[], object]] = {
    "EIF":   lambda : IsolationForest(ntrees=300, sample_size='auto', ndim=1, nthreads=-1),
    "LOF":   lambda: LocalOutlierFactor(
        n_neighbors     =   75,
        novelty         =   False,              # Scoring of the same batch of data
        contamination   =   'auto',             # If quantile thresholds are used for evaluation, the impact is minimal.
        metric          =   'minkowski', p=1,   # First, try Euclidean 'minkowski', p=2; then try p=1 (Manhattan); finally, try 'chebyshev'.
        n_jobs          =   -1
    ),
    "COPOD": _wrap_pyod(COPOD),
    "INNE":  _wrap_pyod(INNE, n_estimators=200, max_samples=256),
    "OCSVM": _wrap_pyod(OCSVM, kernel="rbf", nu=0.05, gamma="scale"),
    "AE": lambda: None
}

def build_autoencoder(input_dim: int, cfg: dict | None = None) -> Model:
    log.info("AE.build | input_dim=%d | cfg=%s", input_dim, (cfg and "custom") or "simple")
    """
    Build an autoencoder (without compiling).
    - If cfg is None: Fall back to a simple symmetric AE, compatible with legacy logic.
    - If cfg provides: Use configurable deep symmetric bottlenecks, optional BN/Dropout/sparse/DAE/output activation.
    """
    if cfg is None:
        # ---- Compatible with older versions of Simple After Effects ----
        inp = Input(shape=(input_dim,))
        x = Dense(16, activation='relu')(inp)
        x = Dense(8, activation='relu')(x)
        encoded = Dense(4, activation='relu')(x)
        x = Dense(8, activation='relu')(encoded)
        x = Dense(16, activation='relu')(x)
        out = Dense(input_dim, activation='linear')(x)
        return Model(inputs=inp, outputs=out)

    # ---- Configurable deep AE ----
    hidden_units = cfg.get("hidden_units", [128, 64, 16])  # The last layer is the bottleneck.
    activation   = cfg.get("activation", "relu")           # "relu" | "elu" | "leaky_relu"
    use_bn       = bool(cfg.get("use_batchnorm", True))
    dropout_rate = float(cfg.get("dropout_rate", 0.0))
    sparse_l1    = float(cfg.get("sparse_l1", 0.0))
    denoise_sig  = float(cfg.get("denoise_sigma", 0.0))
    out_act      = cfg.get("output_activation", "linear")  # 标准化后一般用 "linear"

    def act_layer(x):
        if activation == "relu":
            return tf.keras.layers.ReLU()(x)
        elif activation == "elu":
            return tf.keras.layers.ELU()(x)
        elif activation == "leaky_relu":
            return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    inputs = Input(shape=(input_dim,))
    x = inputs

    # Denoising autoencoder (effective only during training)
    if denoise_sig and denoise_sig > 0:
        x = GaussianNoise(denoise_sig)(x)

    # Encoder (excluding bottleneck)
    for units in hidden_units[:-1]:
        x = Dense(units)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = act_layer(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Bottleneck layer (active regular expressions can be added)
    bottleneck = hidden_units[-1]
    if sparse_l1 and sparse_l1 > 0:
        x = Dense(bottleneck, activation=None,
                  activity_regularizer=regularizers.l1(sparse_l1))(x)
    else:
        x = Dense(bottleneck, activation=None)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = act_layer(x)

    # Decoder (symmetric)
    for units in reversed(hidden_units[:-1]):
        x = Dense(units)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = act_layer(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(input_dim, activation=out_act)(x)
    return Model(inputs, outputs)

def train_autoencoder(normal_data):
    log.info("AE.train | samples=%d features=%d", *normal_data.shape)
    """
    normal_data: ndarray of shape (n_samples, n_features) includes only normal sample features
    Returns the trained autoencoder model and scaler.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_data)
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    # The number of training epochs can be adjusted as needed; here it is set to 20, and verbose=0 for silent training.
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, verbose=0)
    log.info("AE.train | done")
    return autoencoder, scaler

def _try_get_score(model, X, prefer_neg=True) -> np.ndarray:
    """
    Try to get "continuous" scores and maintain a consistent direction: the larger the score, the more abnormal it is.
    """
    # --- LOF Special Criterion: "The larger the value, the more abnormal" ---
    if isinstance(model, LocalOutlierFactor):
        if getattr(model, "novelty", False):
            # When novelty=True, use score_samples/decision_function; the lower the value, the more abnormal it is → take the negative value.
            return (-model.score_samples(X).ravel()).astype(float)
        else:
            # When novelty=False, use negative_outlier_factor_; the smaller the value, the more outlier it is → take the negative value.
            return (-model.negative_outlier_factor_.ravel()).astype(float)

    if hasattr(model, "decision_scores_"):            # PyOD / LOF(novelty=True)
        scores = model.decision_scores_
        prefer_neg = False  # ← ★ Add this line: EIF/PyOD algorithms indicate that "the larger the value, the more abnormal it is."
    elif hasattr(model, "anomaly_score_"):            # isotree EIF
        scores = model.anomaly_score_
        prefer_neg = False
    elif hasattr(model, "decision_function"):         # OCSVM(novelty=True)
        scores = model.decision_function(X).ravel()
        prefer_neg = False
    elif hasattr(model, "negative_outlier_factor_"):  # ←★ sklearn LOF(novelty=False)
        scores = model.negative_outlier_factor_
    elif hasattr(model, "score_samples"):
        scores = model.score_samples(X).ravel()
    else:                                             # Worst case: ±1 tag
        scores = model.predict(X).astype(float)
        prefer_neg = True

    # Unified direction
    return (-scores if prefer_neg else scores).astype(float)

def run_algo(name: str, X: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """
    All algorithms are scheduled uniformly, returning continuous anomaly scores.
    Higher scores indicate more anomalies.
    """
    cfg = {} if cfg is None else cfg
    if name == "EIF":
        scoring_metric   = cfg.get("scoring_metric", "depth")  # "depth" 或 "density"
        penalize_range   = cfg.get("penalize_range", True)
        weigh_by_kurtosis = cfg.get("weigh_by_kurtosis", True)

        # ★ Key point: density is incompatible with penalize_range/weight and must be turned off.
        if scoring_metric == "density":
            penalize_range = False
            weigh_by_kurtosis = False

        base_params = dict(
            ntrees=cfg.get("ntrees", 500),
            sample_size=cfg.get("sample_size", "auto"),
            ndim=cfg.get("ndim", 2),
            nthreads=cfg.get("nthreads", -1),
            penalize_range=penalize_range,
            weigh_by_kurtosis=weigh_by_kurtosis,
        )

        # Compatible with different versions of isotree: Send the scoring_metric if it exists, ignore it if it doesn't.
        try:
            model = IsolationForest(scoring_metric=scoring_metric, **base_params)
        except TypeError:
            model = IsolationForest(**base_params)

        model.fit(X)

        # After fitting the training set, prioritize using anomaly_score (the larger the score, the more outliers).
        if hasattr(model, "anomaly_score_"):
            return model.anomaly_score_.astype(float)

        return _try_get_score(model, X, prefer_neg=False)
    # --- EIF: Supports configurable parameters ---

    # 1) Feature scaling: Distance/density model sensitive
    # if name in {"LOF", "OCSVM", "INNE", "COPOD"}: —— Try modifying it to only standardize OCSVM
    # 1) First, scale: LOF/OCSVM are both scale-sensitive.
    # 1) Scaling
    if name in {"OCSVM", "LOF"}:
        X_ = StandardScaler().fit_transform(X)
    else:
        X_ = X

    # 1.5) PCA (Uniform Guardrail + Default: LOF Only 0.95)
    use_pca_default = (name == "LOF")
    pca_cfg = {}
    if isinstance(cfg.get("pca"), dict):
        pca_cfg = cfg["pca"]
    elif use_pca_default:
        pca_cfg = {"on": True, "n_components": 0.95, "svd_solver": "auto", "whiten": False}

    X_ = _apply_pca_with_guard(X_, pca_cfg if pca_cfg.get("on", False) else None)

    # 2) Train
    model = ALGOS[name]()
    model.fit(X_)

    if name == "LOF":
        log.debug("LOF details | novelty=%s has_decision_fn=%s has_NOF=%s",
                  getattr(model, "novelty", None),
                  hasattr(model, "decision_function"),
                  hasattr(model, "negative_outlier_factor_"))

    # 3) Scoring (with the standard rule of "higher score indicates greater abnormality")
    scores = _try_get_score(model, X_, prefer_neg=True)
    return scores