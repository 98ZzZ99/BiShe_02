# AD_algorithms.py
"""
统一管理要跑的异常检测模型
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


# --- ② 工厂函数 —— 返回已经 .fit() 好且带 .decision_scores_ 属性的对象
def _wrap_pyod(cls, **kw):
    """把 PyOD 模型封装成 “无参构造器” 便于延迟实例化"""
    return lambda: cls(**kw)

ALGOS: Dict[str, Callable[[], object]] = {
    "EIF":   lambda : IsolationForest(ntrees=300, sample_size='auto', ndim=1, nthreads=-1),
    "LOF":   lambda: LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02),
    "COPOD": _wrap_pyod(COPOD),
    "INNE":  _wrap_pyod(INNE, n_estimators=200, max_samples=256),
    "OCSVM": _wrap_pyod(OCSVM, kernel="rbf", nu=0.05, gamma="scale"),
    "AE": lambda: None
}

def build_autoencoder(input_dim: int, cfg: dict | None = None) -> Model:
    log.info("AE.build | input_dim=%d | cfg=%s", input_dim, (cfg and "custom") or "simple")
    """
    构建自编码器（不编译）。
    - 若 cfg is None：回退到一个简单的对称 AE，兼容旧逻辑。
    - 若 cfg 提供：使用可配置的深层对称瓶颈、可选 BN/Dropout/稀疏/DAE/输出激活。
    """
    if cfg is None:
        # ---- 兼容旧版的简单 AE ----
        inp = Input(shape=(input_dim,))
        x = Dense(16, activation='relu')(inp)
        x = Dense(8, activation='relu')(x)
        encoded = Dense(4, activation='relu')(x)
        x = Dense(8, activation='relu')(encoded)
        x = Dense(16, activation='relu')(x)
        out = Dense(input_dim, activation='linear')(x)
        return Model(inputs=inp, outputs=out)

    # ---- 可配置的深层 AE ----
    hidden_units = cfg.get("hidden_units", [128, 64, 16])  # 最后一层为瓶颈
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

    # 去噪自编码器（仅训练期生效）
    if denoise_sig and denoise_sig > 0:
        x = GaussianNoise(denoise_sig)(x)

    # 编码器（除瓶颈外）
    for units in hidden_units[:-1]:
        x = Dense(units)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = act_layer(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # 瓶颈层（可加活动正则）
    bottleneck = hidden_units[-1]
    if sparse_l1 and sparse_l1 > 0:
        x = Dense(bottleneck, activation=None,
                  activity_regularizer=regularizers.l1(sparse_l1))(x)
    else:
        x = Dense(bottleneck, activation=None)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = act_layer(x)

    # 解码器（对称）
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
    normal_data: ndarray of shape (n_samples, n_features) 只包含正常样本特征
    返回训练好的 autoencoder 模型和 scaler
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_data)
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    # 训练 epochs 可根据需要调整，这里设为20，verbose=0 静默训练
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, verbose=0)
    log.info("AE.train | done")
    return autoencoder, scaler

def _try_get_score(model, X, prefer_neg=True) -> np.ndarray:
    """
    尽量拿到“连续”得分并统一方向：越大越异常
    """
    if hasattr(model, "decision_scores_"):            # PyOD / LOF(novelty=True)
        scores = model.decision_scores_
        prefer_neg = False  # ← ★加这一行 EIF / PyOD 系算法“越大越异常”
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
    else:                                             # 最坏：±1 标签
        scores = model.predict(X).astype(float)
        prefer_neg = True

    # 统一方向
    return (-scores if prefer_neg else scores).astype(float)

def run_algo(name: str, X: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """
    统一调度各算法，返回连续 anomaly score。
    分数越大 → 越异常。
    """
    cfg = {} if cfg is None else cfg
    if name == "EIF":
        scoring_metric   = cfg.get("scoring_metric", "depth")  # "depth" 或 "density"
        penalize_range   = cfg.get("penalize_range", True)
        weigh_by_kurtosis = cfg.get("weigh_by_kurtosis", True)

        # ★ 关键：density 与 penalize_range/权重 不兼容，必须关闭
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

        # 兼容不同版本的 isotree：有 scoring_metric 就传，没有就忽略
        try:
            model = IsolationForest(scoring_metric=scoring_metric, **base_params)
        except TypeError:
            model = IsolationForest(**base_params)

        model.fit(X)

        # 拟合训练集后优先用 anomaly_score_（越大越异常）
        if hasattr(model, "anomaly_score_"):
            return model.anomaly_score_.astype(float)

        return _try_get_score(model, X, prefer_neg=False)
    # --- EIF：支持可配参数 ---
    if name == "EIF":
        scoring_metric    = cfg.get("scoring_metric", "depth")
        penalize_range    = cfg.get("penalize_range", True)
        weigh_by_kurtosis = cfg.get("weigh_by_kurtosis", True)
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
        try:
            model = IsolationForest(scoring_metric=scoring_metric, **base_params)
        except TypeError:
            model = IsolationForest(**base_params)

        model.fit(X)
        if hasattr(model, "anomaly_score_"):
            return model.anomaly_score_.astype(float)
        return _try_get_score(model, X, prefer_neg=False)

    # 1) 特征缩放：距离 / 密度模型敏感
    # if name in {"LOF", "OCSVM", "INNE", "COPOD"}: —— 尝试修改为只对 OCSVM 进行标准化
    if name in {"OCSVM"}:
        X_ = StandardScaler().fit_transform(X)
    else:                       # EIF 或其它
        X_ = X

    # 1.5) 降维仅用于 LOF
    if name == "LOF":
        pca = PCA(n_components=0.95)  # 保留 95% 方差
        X_ = pca.fit_transform(X_)

    # 2) 训练
    model = ALGOS[name]()
    model.fit(X_)
    # === DEBUG ===
    if name == "LOF":
        log.debug("LOF details | novelty=%s has_decision_fn=%s has_NOF=%s",
                           getattr(model, "novelty", None),
                           hasattr(model, "decision_function"),
                           hasattr(model, "negative_outlier_factor_"))


    # 3) 取分（关键）
    scores = _try_get_score(model, X_, prefer_neg=True)

    return scores