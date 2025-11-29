import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, json, math, time, random, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, average_precision_score, label_ranking_average_precision_score,
    hamming_loss, jaccard_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.utils import check_random_state

from xgboost import XGBClassifier
from joblib import Parallel, delayed
from tqdm import tqdm

import pickle
from datetime import datetime

# ------ Fusion (PyTorch) ------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import MultiheadAttention
import torch.nn.functional as F

# === NEW: CV & Optuna ===
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna





# 导入iterstrat用于多标签分层划分
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    print("需要安装 iterstrat：pip install iterstrat")
    exit(1)

# 导入SHAP用于模型解释
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP不可用，将跳过可解释性分析。安装: pip install shap")
    SHAP_AVAILABLE = False

# ====== Repro ======
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import os as _os

_os.environ["PYTHONHASHSEED"] = str(SEED)

# ====== Paths ======
ODOR_MM_PATH = "Odor-MM-clean.csv"
PADEL_PATH = "PaDEL-clean.csv"
CHEMBERTA_FEATURES_PATH = "odor_zinc.npy"

# ====== Switches ======
EARLY_STOP_ROUNDS = 50  # XGBoost early stopping
USE_GPU = True  # 若支持 CUDA，建议开启以显著加速

# ====== SHAP Parameters ======
SHAP_SAMPLE_SIZE = 100  # Number of samples to use for SHAP analysis
TOP_FEATURES = 20  # Number of top features to show in plots
TARGET_LABELS = ["odorless", "green", "sweet", "floral", "fruity"]  # Labels for specific analysis

# ====== Argument Parser ======
def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost for Multilabel Odor Prediction')
    parser.add_argument('--features', type=str, default='both',
                        choices=['chemberta', 'padel', 'both'],
                        help='Feature types to use: chemberta, padel, or both')
    parser.add_argument('--shap', action='store_true',
                        help='Perform SHAP analysis for model interpretability')
    parser.add_argument('--fusion', type=str, default='none',
                        choices=['none', 'attn'],  # 'none'=原拼接; 'attn'=注意力融合
                        help='Feature fusion mode: none (concat) or attn (attention-based)')
    parser.add_argument('--fusion-dim', type=int, default=256,
                        help='Projection dimension for each modality before fusion')
    parser.add_argument('--fusion-epochs', type=int, default=100)
    parser.add_argument('--fusion-batch', type=int, default=256)
    parser.add_argument('--fusion-lr', type=float, default=1e-3)
    parser.add_argument('--fusion-dropout', type=float, default=0.2)
    parser.add_argument('--fusion-patience', type=int, default=10)
    parser.add_argument('--fusion-useZonly', action='store_true',
                        help='If set, use only fused Z for XGB (else use [orig concat || Z])')
    parser.add_argument('--lam-align', type=float, default=0.05)
    parser.add_argument('--lam-orth', type=float, default=0.01)
    parser.add_argument('--lam-supcon', type=float, default=0.05)
    parser.add_argument('--lam-ent', type=float, default=0.01)
    parser.add_argument('--cv-folds', type=int, default=5, help='Stratified K-Fold (default: 5)')
    parser.add_argument('--optuna-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--optuna-timeout', type=int, default=0, help='Optuna timeout (sec), 0=disable')
    parser.add_argument('--optuna-sample-labels', type=int, default=0,
                        help='Use top-K frequent labels during Optuna (0=use all labels)')
    parser.add_argument('--shap-fold', type=int, default=0, help='Run SHAP only on this fold (default: 0)')


    return parser.parse_args()


# ====== Utils ======
def detect_columns(df: pd.DataFrame):
    """Identify SMILES column and label columns."""
    cols = list(df.columns)
    smiles_col = None
    for cand in ("SMILES", "smiles"):
        if cand in cols:
            smiles_col = cand
            break
    if smiles_col is None:
        raise ValueError("未找到 SMILES 列（应为 'SMILES' 或 'smiles'）。")

    label_candidates = [c for c in cols if c != smiles_col]

    def is_binary_col(s: pd.Series) -> bool:
        if not np.issubdtype(s.dtype, np.number):
            return False
        uniq = pd.unique(s.dropna().astype(int))
        return set(uniq).issubset({0, 1})

    if all(is_binary_col(df[c]) for c in label_candidates):
        label_cols = label_candidates
    else:
        label_cols = [c for c in label_candidates if is_binary_col(df[c])]
        if not label_cols:
            raise ValueError("未识别到任何二元标签列，请检查数据。")
    return smiles_col, label_cols

def compute_comprehensive_metrics(y_true, probs, thr_vec, label_names=None):
    """计算全面的评估指标，包括逐标签指标和整体指标"""
    y_pred = (probs >= thr_vec).astype(int)
    eps = 1e-12

    # 整体指标
    overall_metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "lrap": label_ranking_average_precision_score(y_true, probs),
        "map_macro": average_precision_score(y_true, probs, average="macro"),
    }

    # 逐标签指标
    labelwise_metrics = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        pj = probs[:, j]
        yj = y_pred[:, j]
        support_pos = int(yt.sum())
        support_neg = int((1 - yt).sum())

        # 排序指标
        try:
            ap = average_precision_score(yt, pj) if support_pos > 0 else np.nan
        except:
            ap = np.nan

        try:
            auc = roc_auc_score(yt, pj) if (support_pos > 0 and support_neg > 0) else np.nan
        except:
            auc = np.nan

        # 分类指标
        f1 = f1_score(yt, yj, zero_division=0)
        pr = precision_score(yt, yj, zero_division=0)
        rc = recall_score(yt, yj, zero_division=0)
        acc = accuracy_score(yt, yj)

        # 混淆矩阵元素
        tn = int(((yt == 0) & (yj == 0)).sum())
        tp = int(((yt == 1) & (yj == 1)).sum())
        fn = int(((yt == 1) & (yj == 0)).sum())
        fp = int(((yt == 0) & (yj == 1)).sum())

        # 派生指标
        tpr = tp / (tp + fn + eps)  # sensitivity = recall
        tnr = tn / (tn + fp + eps)  # specificity
        ba = 0.5 * (tpr + tnr)  # Balanced Accuracy

        # MCC
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
        mcc = ((tp * tn) - (fp * fn)) / denom

        # LR+ = sensitivity / (1 - specificity)
        lr_pos = tpr / (1.0 - tnr + eps)
        if fp == 0 and tp > 0:
            lr_pos = np.inf

        label_metrics = {
            "label": label_names[j] if label_names is not None else f"Label_{j}",
            "support_pos": support_pos,
            "support_neg": support_neg,
            "ap": ap,
            "auc": auc,
            "f1": f1,
            "precision": pr,
            "recall": rc,
            "accuracy": acc,
            "balanced_accuracy": ba,
            "mcc": mcc,
            "specificity": tnr,
            "lr_plus": lr_pos,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        labelwise_metrics.append(label_metrics)

    # 追加整体 PR-AUC(micro) 与 ROC-AUC(micro)
    try:
        overall_metrics["map_micro"] = average_precision_score(y_true, probs, average="micro")
    except Exception:
        overall_metrics["map_micro"] = np.nan

    # micro-ROC-AUC：需二分类且每标签既有正也有负
    try:
        overall_metrics["auc_micro"] = roc_auc_score(y_true, probs, average="micro")
    except Exception:
        overall_metrics["auc_micro"] = np.nan

    # 追加基于逐标签的宏统计：Balanced Accuracy 与 MCC 的 nanmean
    ba_vals = np.array([m["balanced_accuracy"] for m in labelwise_metrics], dtype=float)
    mcc_vals = np.array([m["mcc"] for m in labelwise_metrics], dtype=float)
    overall_metrics["balanced_accuracy_macro"] = float(np.nanmean(ba_vals))
    overall_metrics["mcc_macro"] = float(np.nanmean(mcc_vals))

    return overall_metrics, labelwise_metrics


def tune_thresholds(y_true, probs, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    L = y_true.shape[1]
    thr = np.zeros(L, dtype=np.float32)
    for j in range(L):
        yt = y_true[:, j]
        pj = probs[:, j]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            yj = (pj >= t).astype(int)
            f1 = f1_score(yt, yj, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[j] = best_t
    return thr

def build_shared_xgb_params(
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    reg_lambda, reg_alpha, min_child_weight, gamma, seed=SEED
):
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
    )
    # GPU 优先
    if USE_GPU:
        try:
            params["device"] = "cuda"   # xgboost>=2.0
        except Exception:
            params.pop("device", None)
            params["tree_method"] = "gpu_hist"
    return params

def optuna_objective(trial, X_tr, Y_tr, X_va, Y_va, label_order=None):
    # 搜索空间（可按需调整上下界）
    params = build_shared_xgb_params(
        n_estimators = trial.suggest_int("n_estimators", 300, 1200),
        max_depth    = trial.suggest_int("max_depth", 3, 10),
        learning_rate= trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        subsample    = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha    = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        min_child_weight = trial.suggest_float("min_child_weight", 1e-1, 10.0, log=True),
        gamma        = trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        seed=SEED
    )

    L = Y_tr.shape[1]
    label_idx_list = list(range(L)) if label_order is None else label_order

    # 逐标签用同一组参数训练 -> 验证集概率 -> 宏平均AP
    ap_list = []
    for j in label_idx_list:
        yt_tr, yt_va = Y_tr[:, j], Y_va[:, j]
        pos_tr = int(yt_tr.sum()); neg_tr = yt_tr.shape[0] - pos_tr
        spw = float(neg_tr) / float(pos_tr + 1e-6)

        clf = XGBClassifier(**params, n_jobs=1, scale_pos_weight=spw)
        try:
            clf.fit(X_tr, yt_tr, eval_set=[(X_va, yt_va)],
                    early_stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)
        except TypeError:
            clf.fit(X_tr, yt_tr)

        p_va = clf.predict_proba(X_va)[:, 1]

        # 只有当该标签在验证集中有正负样本时才纳入AP
        has_pos = yt_va.sum() > 0
        has_neg = (1 - yt_va).sum() > 0
        if has_pos and has_neg:
            try:
                ap = average_precision_score(yt_va, p_va)
                ap_list.append(ap)
            except:
                ap_list.append(0.0)
        else:
            # 使用训练集正样本比例作为简单替代
            train_pos_ratio = np.mean(yt_tr)
            ap_list.append(train_pos_ratio)  # 这只是一个简单替代

    # 末端保护：若极端情况下无可算AP的标签，返回极小值避免报错
    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))

def train_ovr_with_shared_params(X_tr, Y_tr, X_te, shared_params, verbose=False):
    L = Y_tr.shape[1]
    P_te = np.zeros((X_te.shape[0], L), dtype=np.float32)
    models = [None] * L

    with tqdm(total=L, desc="[XGB] OVR fit (shared params)", ncols=100) as pbar:
        for j in range(L):
            yj = Y_tr[:, j]
            pos_tr = int(yj.sum()); neg_tr = yj.shape[0] - pos_tr
            spw = float(neg_tr) / float(pos_tr + 1e-6)

            clf = XGBClassifier(**shared_params, n_jobs=1, scale_pos_weight=spw)
            try:
                clf.fit(X_tr, yj, verbose=verbose)
            except TypeError:
                clf.fit(X_tr, yj)

            models[j] = clf
            P_te[:, j] = clf.predict_proba(X_te)[:, 1]
            pbar.update(1)
    return P_te, models

def perform_shap_analysis(models, X_data, feature_names, label_names, sample_size=SHAP_SAMPLE_SIZE,
                          top_features=TOP_FEATURES, target_labels=TARGET_LABELS, tag=""):
    """执行SHAP分析并生成科研级别的可视化图表"""
    if not SHAP_AVAILABLE:
        print("SHAP不可用，跳过可解释性分析")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/shap", exist_ok=True)

    # 采样以减少计算时间（固定随机种子以确保可复现性）
    rng = np.random.RandomState(42)
    sample_size = min(sample_size, X_data.shape[0])
    sample_indices = rng.choice(X_data.shape[0], sample_size, replace=False)
    X_sample = X_data[sample_indices]

    print(f"Performing SHAP analysis on {X_sample.shape[0]} samples...")

    # 找出目标标签的索引
    target_indices = []
    for label in target_labels:
        if label in label_names:
            target_indices.append(label_names.index(label))
        else:
            print(f"Warning: Target label '{label}' not found in label names")

    # 如果没有找到目标标签，使用前5个标签
    if not target_indices:
        target_indices = list(range(min(5, len(label_names))))
        target_labels = [label_names[i] for i in target_indices]

    print(f"SHAP analysis target labels: {target_labels}")

    # 计算所有特征的全局SHAP值
    all_shap_values = []
    for i, model in enumerate(models):
        # 防御：模型可能为 None（极端少样本标签）
        if model is None:
            # 该标签没有成功保留模型（极少见），用零填充以不影响全局平均
            all_shap_values.append(np.zeros((X_sample.shape[0], len(feature_names)), dtype=float))
            continue

        # 使用CPU预测器进行SHAP解释（更稳定）
        try:
            booster = model.get_booster()
            booster.set_param({'predictor': 'cpu_predictor'})  # GPU训练、CPU解释
            explainer = shap.TreeExplainer(booster)
        except Exception as e:
            print(f"Warning: Failed to set CPU predictor for label {i}, using default: {e}")
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)
        all_shap_values.append(shap_values)

    # 转换为numpy数组
    all_shap_values = np.array(all_shap_values)  # shape: (n_labels, n_samples, n_features)

    # 确保top_features不超过特征总数
    top_features = min(top_features, len(feature_names))

    # 图1: 全局特征重要性
    plt.figure(figsize=(12, 8))

    # 计算全局平均SHAP绝对值
    mean_abs_shap = np.mean(np.abs(all_shap_values), axis=(0, 1))

    # 获取最重要的特征索引
    top_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
    top_features_names = [feature_names[i] for i in top_indices]
    top_features_importance = mean_abs_shap[top_indices]

    # 创建条形图
    plt.barh(range(len(top_features_names)), top_features_importance[::-1])
    plt.yticks(range(len(top_features_names)), top_features_names[::-1])
    plt.xlabel('Mean |SHAP value| (feature importance)')
    plt.title('Global feature importance (mean |SHAP| across labels)')
    plt.tight_layout()
    plt.savefig(f"results/shap/{tag}_{timestamp}_global_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"results/shap/{tag}_{timestamp}_global_feature_importance.pdf", bbox_inches='tight')
    plt.close()

    # 导出全局特征重要性CSV（Top K和全量）
    pd.DataFrame({
        "feature": top_features_names,
        "mean_abs_shap": top_features_importance
    }).to_csv(f"results/shap/{tag}_{timestamp}_global_feature_importance_top{top_features}.csv", index=False)

    # 导出全量特征重要性
    order = np.argsort(mean_abs_shap)[::-1]
    pd.DataFrame({
        "feature": np.array(feature_names)[order],
        "mean_abs_shap": mean_abs_shap[order]
    }).to_csv(f"results/shap/{tag}_{timestamp}_global_feature_importance_full.csv", index=False)

    # 图2: 特征贡献分布散点图 (beeswarm plot) - 全局
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        all_shap_values.mean(axis=0),  # 平均所有标签
        X_sample,
        feature_names=feature_names,
        max_display=top_features,
        show=False
    )
    plt.title('Global feature contribution distribution')
    plt.tight_layout()
    plt.savefig(f"results/shap/{tag}_{timestamp}_beeswarm_global.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"results/shap/{tag}_{timestamp}_beeswarm_global.pdf", bbox_inches='tight')
    plt.close()

    # 图3: 标签特异性特征解释
    for i, label_idx in enumerate(target_indices):
        label_name = label_names[label_idx]
        print(f"Generating SHAP plots for label: {label_name}")

        # 获取该标签的SHAP值
        label_shap_values = all_shap_values[label_idx]

        # 创建标签特异性特征重要性图
        plt.figure(figsize=(12, 8))

        # 计算该标签的平均SHAP绝对值
        label_mean_abs_shap = np.mean(np.abs(label_shap_values), axis=0)

        # 获取最重要的特征索引
        label_top_indices = np.argsort(label_mean_abs_shap)[-top_features:][::-1]
        label_top_features_names = [feature_names[i] for i in label_top_indices]
        label_top_features_importance = label_mean_abs_shap[label_top_indices]

        # 创建条形图
        plt.barh(range(len(label_top_features_names)), label_top_features_importance[::-1])
        plt.yticks(range(len(label_top_features_names)), label_top_features_names[::-1])
        plt.xlabel('Mean |SHAP value| (feature importance)')
        plt.title(f'Label-specific feature importance — "{label_name}"')
        plt.tight_layout()
        plt.savefig(f"results/shap/{tag}_{timestamp}_{label_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"results/shap/{tag}_{timestamp}_{label_name}_feature_importance.pdf", bbox_inches='tight')
        plt.close()

        # 导出标签特异性特征重要性CSV
        pd.DataFrame({
            "feature": label_top_features_names,
            "mean_abs_shap": label_top_features_importance
        }).to_csv(f"results/shap/{tag}_{timestamp}_{label_name}_feature_importance_top{top_features}.csv", index=False)

        # 创建标签特异性beeswarm图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            label_shap_values,
            X_sample,
            feature_names=feature_names,
            max_display=min(10, top_features),  # 只显示前10个最重要的特征
            show=False
        )
        plt.title(f'Label-specific beeswarm — "{label_name}"')
        plt.tight_layout()
        plt.savefig(f"results/shap/{tag}_{timestamp}_{label_name}_beeswarm.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"results/shap/{tag}_{timestamp}_{label_name}_beeswarm.pdf", bbox_inches='tight')
        plt.close()

    print(f"SHAP分析完成，结果保存在 results/shap/ 目录下")

# =======================
# Attention-based Fusion
# =======================

class ModProject(nn.Module):
    """单模态投影到同一隐空间 + LayerNorm + Dropout"""
    def __init__(self, in_dim, out_dim, p=0.2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.do = nn.Dropout(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.constant_(self.fc.bias, 0.0)
    def forward(self, x):
        z = self.fc(x)
        z = self.ln(z)
        z = torch.relu(z)
        return self.do(z)

class CrossAttnFusion(nn.Module):
    """
    Cross-attention multimodal fusion:
    - 先各自投影到同维 d：h_c, h_p
    - 双向多头交叉注意力：z_c = Attn(h_c <- h_p), z_p = Attn(h_p <- h_c)
    - 门控融合：beta ∈ [0,1]，z = beta * z_c + (1-beta) * z_p
    - 逐维门控重标定原拼接（x_rw）用于“Concat+Z”对照
    - 输出：(z, x_rw, alpha)；alpha 这里返回 [beta, 1-beta] 以对齐你原先可视化逻辑
    """
    def __init__(self, in_c, in_p, d=256, nheads=4, p=0.2, tau=2.0, dropout_modal=0.1):
        super().__init__()
        self.proj_c = ModProject(in_c, d, p)
        self.proj_p = ModProject(in_p, d, p)
        self.mha_cp = MultiheadAttention(d, nheads, batch_first=True)  # q: c, kv: p
        self.mha_pc = MultiheadAttention(d, nheads, batch_first=True)  # q: p, kv: c
        self.g_beta  = nn.Linear(4*d, 1)    # 决定 z_c vs z_p 的权重
        self.g_gate  = nn.Linear(2*d, in_c + in_p)
        self.tau = float(tau)
        self.dropout_modal = float(dropout_modal)

        nn.init.xavier_uniform_(self.g_beta.weight); nn.init.constant_(self.g_beta.bias, 0.0)
        nn.init.xavier_uniform_(self.g_gate.weight); nn.init.constant_(self.g_gate.bias, 0.0)

    def forward(self, x_c, x_p):
        h_c = self.proj_c(x_c)   # [B, d]
        h_p = self.proj_p(x_p)   # [B, d]

        # 模态 dropout：随机屏蔽单模态表征，抗塌缩
        if self.training and self.dropout_modal > 0:
            if torch.rand(1).item() < self.dropout_modal:
                h_c = torch.zeros_like(h_c)
            if torch.rand(1).item() < self.dropout_modal:
                h_p = torch.zeros_like(h_p)

        # 交叉注意力（把每个模态当作长度1的“token”序列来交互）
        q_c, k_p, v_p = h_c.unsqueeze(1), h_p.unsqueeze(1), h_p.unsqueeze(1)
        q_p, k_c, v_c = h_p.unsqueeze(1), h_c.unsqueeze(1), h_c.unsqueeze(1)
        z_c, _ = self.mha_cp(q_c, k_p, v_p)   # [B, 1, d]
        z_p, _ = self.mha_pc(q_p, k_c, v_c)   # [B, 1, d]
        z_c = z_c.squeeze(1); z_p = z_p.squeeze(1)

        # 门控融合（sigmoid） + 温度
        fuse_feat = torch.cat([h_c, h_p, z_c, z_p], dim=1)  # [B, 4d]
        beta = torch.sigmoid(self.g_beta(fuse_feat) / self.tau)  # [B, 1]
        z = beta * z_c + (1.0 - beta) * z_p                   # [B, d]

        # 逐维门控对原拼接重标定（保留你的论文消融/SHAP使用）
        gate = torch.sigmoid(self.g_gate(torch.cat([h_c, h_p], dim=1)))  # [B, in_c+in_p]
        x_rw = torch.cat([x_c, x_p], dim=1) * gate

        # 建议返回 5 个量，便于正则项或诊断
        alpha = torch.cat([beta, 1.0 - beta], dim=1)
        return z, x_rw, alpha, h_c, h_p


class LabelDecoder(nn.Module):
    """
    Label-Embedding Decoder:
    - 学习一个 [L, d] 的标签嵌入矩阵 E，logits = z @ E^T + b
    - 天然建模标签相关性（E 的行间夹角）
    """
    def __init__(self, d, L, use_bias=True):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(L, d))
        self.bias = nn.Parameter(torch.zeros(L)) if use_bias else None
        nn.init.xavier_uniform_(self.emb)

    def forward(self, z):
        logits = z @ self.emb.t()
        if self.bias is not None:
            logits = logits + self.bias
        return logits

class AsymmetricLossMultiLabel(nn.Module):
    """
    ASL for multi-label:
    - γ+=0, γ-=4（可调）
    - clipping δ=0.05 抑制易负样本主导
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)
        loss = loss_pos * ((1 - pt_pos) ** self.gp) + loss_neg * ((1 - pt_neg) ** self.gn)
        return -loss.mean()

def align_loss(h_c, h_p):
    # 同一样本跨模态对齐：拉近方向（单位化后余弦距离）
    h_c = F.normalize(h_c, dim=1)
    h_p = F.normalize(h_p, dim=1)
    return (1.0 - (h_c * h_p).sum(dim=1)).mean()

def orth_loss(h_c, h_p):
    # 去冗余：让两模态学到的方向尽量正交（越小越好）
    h_c = F.normalize(h_c, dim=1)
    h_p = F.normalize(h_p, dim=1)
    return ((h_c * h_p).sum(dim=1) ** 2).mean()

def supcon_multilabel(z, y, temp=0.07):
    """
    Supervised contrastive for multi-label:
    - 共享至少一个标签视为正对
    - 其它为负对
    """
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / temp              # [B, B]
    with torch.no_grad():
        # 任一标签相同即为正对（对角置0防止自身）
        pos_mask = (y @ y.t()) > 0
        eye = torch.eye(y.size(0), device=y.device).bool()
        pos_mask = pos_mask & (~eye)
    # 对每个 i：L_i = - log( sum_j exp(sim_ij) over pos / sum_k exp(sim_ik) over all_except_i )
    exp_sim = torch.exp(sim) * (~eye)
    pos_sum = (exp_sim * pos_mask).sum(dim=1) + 1e-8
    all_sum = exp_sim.sum(dim=1) + 1e-8
    loss = -torch.log(pos_sum / all_sum)
    return loss.mean()

# ========== train_fusion_and_transform (CrossAttn + LED + ASL + Multi-regularizers) ==========
def train_fusion_and_transform(
        X_tr_chem, X_tr_padel, Y_tr,
        X_va_chem, X_va_padel, Y_va,
        X_te_chem, X_te_padel,
        d=256, epochs=100, batch=256, lr=1e-3, dropout=0.2, patience=10, seed=42,
        # 新增可调超参（与 CrossAttnFusion / 正则项配套）
        tau=2.0, dropout_modal=0.1,
        lam_align=0.05, lam_orth=0.01, lam_supcon=0.05, lam_ent=0.01
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_c = X_tr_chem.shape[1]; in_p = X_tr_padel.shape[1]
    L = Y_tr.shape[1]

    # 交叉注意力融合
    fusion = CrossAttnFusion(in_c, in_p, d=d, nheads=4, p=dropout, tau=tau, dropout_modal=dropout_modal).to(device)
    # 标签嵌入解码头
    clf = LabelDecoder(d, L).to(device)

    params = list(fusion.parameters()) + list(clf.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=max(2, patience//2), factor=0.5)
    crit = AsymmetricLossMultiLabel(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)

    def to_tensor(x): return torch.from_numpy(x).float().to(device)
    tr_ds = TensorDataset(to_tensor(X_tr_chem), to_tensor(X_tr_padel), torch.from_numpy(Y_tr).float().to(device))
    va_ds = TensorDataset(to_tensor(X_va_chem), to_tensor(X_va_padel), torch.from_numpy(Y_va).float().to(device))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, drop_last=False)

    best_state, best_val, bad = None, float('inf'), 0

    for ep in range(1, epochs + 1):
        fusion.train(); clf.train()
        tr_loss = 0.0
        for xb_c, xb_p, yb in tr_ld:
            opt.zero_grad()
            # 注意这里接 5 个返回（z, x_rw, alpha, h_c, h_p）
            z, _, alpha, h_c_mid, h_p_mid = fusion(xb_c, xb_p)
            logits = clf(z)

            # 主任务：ASL
            cls_loss = crit(logits, yb)

            # 三正则（如不想启用，把 lam_* 设为 0）
            loss = cls_loss
            if lam_align > 0:
                loss = loss + lam_align * align_loss(h_c_mid, h_p_mid)
            if lam_orth > 0:
                loss = loss + lam_orth * orth_loss(h_c_mid, h_p_mid)
            if lam_supcon > 0:
                loss = loss + lam_supcon * supcon_multilabel(z, yb)
            if lam_ent > 0:
                eps = 1e-8
                entropy = -(alpha * (alpha + eps).log()).sum(dim=1).mean()
                loss = loss - lam_ent * entropy

            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tr_loss += loss.item() * xb_c.size(0)
        tr_loss /= len(tr_ds)

        # 验证（早停）：只看主任务损失
        fusion.eval(); clf.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb_c, xb_p, yb in va_ld:
                z, _, _alpha_dbg, _hc, _hp = fusion(xb_c, xb_p)
                logits = clf(z)
                va_loss += crit(logits, yb).item() * xb_c.size(0)
        va_loss /= len(va_ds)
        sched.step(va_loss)

        improved = va_loss < best_val - 1e-4
        if improved:
            best_val = va_loss; bad = 0
            best_state = {'fusion': fusion.state_dict(), 'clf': clf.state_dict()}
        else:
            bad += 1

        if bad >= patience:
            print(f"[Fusion] Early stop at epoch {ep} (best val={best_val:.4f})")
            break

        if ep % 10 == 0:
            with torch.no_grad():
                xb_c, xb_p, _ = next(iter(tr_ld))
                _z, _xrw, alpha_dbg, _hc, _hp = fusion(xb_c.to(device), xb_p.to(device))
                a_mean = alpha_dbg.mean(0).detach().cpu().numpy()
            print(f"[Fusion] ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | alpha_mean={a_mean}")

    if best_state is not None:
        fusion.load_state_dict(best_state['fusion'])
        clf.load_state_dict(best_state['clf'])
    fusion.eval(); clf.eval()

    # 抽取三划分的 Z/x_rw/alpha
    with torch.no_grad():
        def get_all(Xc, Xp):
            z_list, xrw_list, alpha_list = [], [], []
            bs = 2048
            for i in range(0, len(Xc), bs):
                xb_c = torch.from_numpy(Xc[i:i+bs]).float().to(device)
                xb_p = torch.from_numpy(Xp[i:i+bs]).float().to(device)
                z, xrw, alpha, _hc, _hp = fusion(xb_c, xb_p)
                z_list.append(z.cpu().numpy())
                xrw_list.append(xrw.cpu().numpy())
                alpha_list.append(alpha.cpu().numpy())
            return (np.vstack(z_list).astype(np.float32),
                    np.vstack(xrw_list).astype(np.float32),
                    np.vstack(alpha_list).astype(np.float32))

        Z_tr, Xrw_tr, A_tr = get_all(X_tr_chem, X_tr_padel)
        Z_va, Xrw_va, A_va = get_all(X_va_chem, X_va_padel)
        Z_te, Xrw_te, A_te = get_all(X_te_chem, X_te_padel)

    return (Z_tr, Z_va, Z_te), (Xrw_tr, Xrw_va, Xrw_te), (A_tr, A_va, A_te)

# ====== Load and CV =======
def main():
    args = parse_args()
    feature_type = args.features
    print(f"Using feature type: {feature_type}")

    # === 读入标签表 ===
    print("Loading Odor-MM.csv...")
    df_full = pd.read_csv(ODOR_MM_PATH)
    smiles_col, label_cols = detect_columns(df_full)
    for c in label_cols:
        df_full[c] = df_full[c].fillna(0).astype(int)
    df_full = df_full.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    Y_all = df_full[label_cols].values.astype(int)
    X_placeholder = df_full[[smiles_col]].values  # 仅占位给KFold

    # === 读入特征（全量），chemBERTa直接索引；PaDEL合并后按折内拟合/变换 ===
    chemberta_features = None
    if feature_type in ["chemberta", "both"]:
        print("Loading chemBERTa features...")
        chemberta_features = np.load(CHEMBERTA_FEATURES_PATH)
        assert len(chemberta_features) == len(df_full), "chemBERTa行数与数据不匹配"
        chemberta_features = chemberta_features.astype(np.float32)

    padel_df = None
    padel_cols = []
    if feature_type in ["padel", "both"]:
        print("Loading PaDEL features...")
        padel_df = pd.read_csv(PADEL_PATH, sep=None, engine="python")

        # ——关键的4行：标准化列名 & 清理 Unnamed 列 & 兼容各种写法——
        padel_df.columns = [str(c).strip() for c in padel_df.columns]
        padel_df = padel_df.loc[:, ~padel_df.columns.str.match(r"^Unnamed")]  # 删掉 Excel 残留
        name_lower = {c.lower(): c for c in padel_df.columns}
        if "smiles" in name_lower:  # 兼容 smiles/Smiles/SMILES/Canonical_SMILES
            padel_df = padel_df.rename(columns={name_lower["smiles"]: "SMILES"})
        elif "canonical_smiles" in name_lower:
            padel_df = padel_df.rename(columns={name_lower["canonical_smiles"]: "SMILES"})

        assert "SMILES" in padel_df.columns, f"PaDEL没有SMILES列，实际列名示例: {padel_df.columns.tolist()[:10]}"
        padel_df = padel_df.drop_duplicates(subset=["SMILES"], keep="first").copy()
        padel_cols = [c for c in padel_df.columns if c != "SMILES"]

    # === 构建5折 ===
    kfold = MultilabelStratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)

    # === 若指定仅首折做SHAP ===
    shap_fold_idx = int(args.shap_fold)

    # === Optuna仅运行一次：在第0折的训练/验证划分上 ===
    shared_params = None

    # === 记录每折总体指标、阈值等 ===
    cv_overall_rows = []
    os.makedirs("results/cv5", exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(X_placeholder, Y_all)):
        print(f"\n========== Fold {fold_idx+1}/{args.cv_folds} ==========")
        Y_tr, Y_va = Y_all[tr_idx], Y_all[va_idx]

        # ---- 组装每折的特征矩阵 ----
        # chem
        if feature_type in ["chemberta", "both"]:
            X_tr_chem = chemberta_features[tr_idx]
            X_va_chem = chemberta_features[va_idx]
        else:
            X_tr_chem = np.empty((len(tr_idx), 0), dtype=np.float32)
            X_va_chem = np.empty((len(va_idx), 0), dtype=np.float32)

        # PaDEL（无泄露：折内训练集拟合，再变换验证集）
        if feature_type in ["padel", "both"]:
            df_tr = df_full.iloc[tr_idx][[smiles_col]].copy()
            df_va = df_full.iloc[va_idx][[smiles_col]].copy()

            # 左连接，保持行数与顺序（one_to_one）
            tr_m = pd.merge(df_tr, padel_df, on="SMILES", how="left", sort=False, validate="one_to_one")
            va_m = pd.merge(df_va, padel_df, on="SMILES", how="left", sort=False, validate="one_to_one")

            for c in padel_cols:
                tr_m[c] = pd.to_numeric(tr_m[c], errors="coerce")
                va_m[c] = pd.to_numeric(va_m[c], errors="coerce")

            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            Xpd_tr = imp.fit_transform(tr_m[padel_cols].to_numpy())
            Xpd_va = imp.transform(va_m[padel_cols].to_numpy())
            Xpd_tr = scaler.fit_transform(Xpd_tr).astype(np.float32)
            Xpd_va = scaler.transform(Xpd_va).astype(np.float32)
        else:
            Xpd_tr = np.empty((len(tr_idx), 0), dtype=np.float32)
            Xpd_va = np.empty((len(va_idx), 0), dtype=np.float32)

        # concat 或者 attn-fusion
        if feature_type == "both" and args.fusion == "attn":
            print("==> Using attention-based multimodal fusion (per-fold)")
            (Z_tr, Z_va, _Zte), (Xrw_tr, Xrw_va, _Xrwte), (A_tr, A_va, _Ate) = train_fusion_and_transform(
                X_tr_chem, Xpd_tr, Y_tr,
                X_va_chem, Xpd_va, Y_va,
                X_va_chem, Xpd_va,  # 这里仅需生成到验证集，无需 test
                d=args.fusion_dim,
                epochs=args.fusion_epochs,
                batch=args.fusion_batch,
                lr=args.fusion_lr,
                dropout=args.fusion_dropout,
                patience=args.fusion_patience
            )

            if args.fusion_useZonly:
                # 只使用融合后的 Z 作为 XGBoost 输入
                X_tr_cat, X_va_cat = Z_tr, Z_va
                feature_names = [f"fusionZ_{i}" for i in range(Z_tr.shape[1])]
            else:
                # 将原始 ChemBERTa + PaDEL + Z 一起拼接给 XGBoost
                X_tr_cat = np.hstack([X_tr_chem, Xpd_tr, Z_tr])
                X_va_cat = np.hstack([X_va_chem, Xpd_va, Z_va])
                feature_names = (
                        [f"chemBERTa_{i}" for i in range(X_tr_chem.shape[1])] +
                        padel_cols +
                        [f"fusionZ_{i}" for i in range(Z_tr.shape[1])]
                )

            # ====== 新增：保存本折验证集的 Z / Y / 标签名，用于 Z–label heatmap 等可解释性分析 ======
            debug_dir = "results/fusion_debug"
            os.makedirs(debug_dir, exist_ok=True)

            # 保存本折验证集的 Z 和 Y
            np.save(os.path.join(debug_dir, f"fold{fold_idx}_Z_va.npy"), Z_va)
            np.save(os.path.join(debug_dir, f"fold{fold_idx}_Y_va.npy"), Y_va)

            # 只在第 0 折保存一次标签名称（138 个气味描述词）
            if fold_idx == 0:
                np.save(
                    os.path.join(debug_dir, "label_names.npy"),
                    np.array(label_cols, dtype=object)
                )

        else:
            # 不使用注意力融合时的拼接方式
            X_tr_cat = np.hstack([X_tr_chem, Xpd_tr])
            X_va_cat = np.hstack([X_va_chem, Xpd_va])
            feature_names = []
            if feature_type in ["chemberta", "both"]:
                feature_names += [f"chemBERTa_{i}" for i in range(X_tr_chem.shape[1])]
            if feature_type in ["padel", "both"]:
                feature_names += padel_cols

        # ---- 在当前折上运行 Optuna，得到该折的最优参数（每折独立）----
        print("[Optuna] searching fold-wise params (objective = macro PR-AUC)...")
        label_order = None
        if args.optuna_sample_labels and args.optuna_sample_labels > 0:
            pos_count = Y_tr.sum(axis=0).astype(int)
            label_order = list(np.argsort(-pos_count)[:int(args.optuna_sample_labels)])

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_tr_cat, Y_tr, X_va_cat, Y_va, label_order),
            n_trials=int(args.optuna_trials),
            timeout=int(args.optuna_timeout) if args.optuna_timeout > 0 else None,
            show_progress_bar=True
        )
        best = study.best_params
        fold_params = build_shared_xgb_params(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            subsample=best["subsample"],
            colsample_bytree=best["colsample_bytree"],
            reg_lambda=best["reg_lambda"],
            reg_alpha=best["reg_alpha"],
            min_child_weight=best["min_child_weight"],
            gamma=best["gamma"],
            seed=SEED
        )
        print(f"[Optuna][fold {fold_idx}] best params:", fold_params)

        # ---- 用共享参数训练 OVR -> 得到验证集概率 ----
        Pva, models = train_ovr_with_shared_params(X_tr_cat, Y_tr, X_va_cat, fold_params, verbose=False)

        # ---- 阈值：在该折验证集上逐标签调优（F1最大化）并评估 ----
        thr = tune_thresholds(Y_va, Pva)
        overall, labelwise = compute_comprehensive_metrics(Y_va, Pva, thr, label_cols)

        # 计算宏平均ROC-AUC（逐标签AUC的nanmean）
        auc_macro = float(np.nanmean([row["auc"] for row in labelwise]))
        overall["auc_macro"] = auc_macro
        overall["fold"] = int(fold_idx)

        # 保存本折结果
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(labelwise).to_csv(f"results/cv5/fold{fold_idx}_labelwise.csv", index=False)
        with open(f"results/cv5/fold{fold_idx}_overall.json", "w") as f:
            json.dump(overall, f, indent=2)
        np.save(f"results/cv5/fold{fold_idx}_probs.npy", Pva)
        np.save(f"results/cv5/fold{fold_idx}_thr.npy", thr)

        cv_overall_rows.append(overall)

        # ---- SHAP：仅在指定折（默认0）执行一次 ----
        if args.shap and (fold_idx == shap_fold_idx):
            perform_shap_analysis(
                models=models,
                X_data=X_va_cat,
                feature_names=feature_names,
                label_names=label_cols,
                sample_size=SHAP_SAMPLE_SIZE,
                top_features=TOP_FEATURES,
                target_labels=TARGET_LABELS,
                tag=f"cv5_fold{fold_idx}"
            )

    # === 汇总5折均值±标准差 ===
    cv_overall_df = pd.DataFrame(cv_overall_rows)
    metrics_for_summary = [
        "micro_f1", "macro_f1", "weighted_f1",
        "micro_precision", "micro_recall", "macro_precision", "macro_recall",
        "subset_accuracy", "balanced_accuracy_macro", "mcc_macro",
        "hamming_loss", "jaccard_samples",
        "lrap", "map_macro", "map_micro",
        "auc_macro", "auc_micro"
    ]

    summary = {}
    for m in metrics_for_summary:
        if m in cv_overall_df.columns:
            vals = cv_overall_df[m].values.astype(float)
            summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    with open("results/cv5/cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[CV] mean±std summary saved to results/cv5/cv_summary.json")

if __name__ == "__main__":
    main()
