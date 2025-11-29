# -*- coding: utf-8 -*-
"""
为每个标签绘制一张SCI风格雷达图（Calibri + 加粗）。
- 读取 folds_summary.csv（按标签的“均值 ± 标准差”或数值列）
- 自动解析六个指标：PR-AUC, AU-ROC, F1, REC, PREC, ACC
- 每个标签生成一张 PNG 和 PDF（600dpi）
- 额外导出 per_label_means.csv（每个标签六指标的均值，用于核对）

使用：python radar_per_label.py
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 组合绘图（把多个标签画到同一张雷达图）======
TARGET_LABELS = [
    "odorless", "sulfurous", "green", "sweet", "aromatic", "hyacinth"
]  # 不区分大小写

# 柔和“马卡龙”配色（可增删；不够会循环使用）
MACARON_COLORS = [
    "#FFB3BA",  # 粉
    "#FFDFBA",  # 橙粉
    "#FFFFBA",  # 淡黄
    "#BAFFC9",  # 薄荷绿
    "#BAE1FF",  # 淡蓝
    "#E7BAFF",  # 淡紫
    "#FFD1DC",  # 樱花粉
    "#C6E2FF",  # 冰蓝
]
ANNOTATE_VALUES_MULTI = False  # 多系列图是否在顶点标数值
LINEWIDTH_MULTI = 2.5
MARKER_MULTI = "o"
FILL_ALPHA_MULTI = 0.10

# ====== 配置：输入/输出路径 ======
CSV_PATH = r"..."
OUT_DIR  = r"..."
MODEL_TITLE_PREFIX = "ChemBERTa+XGBoost"   # 图标题前缀，可按需调整
DPI = 600
ANNOTATE_VALUES = False   # 是否在每个顶点标注数值

# ====== 字体：Calibri + 加粗 ======
plt.rcParams["font.family"] = "Calibri"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# ====== 工具函数 ======
def extract_mean(x):
    """从 '0.812 ± 0.023'、'0.812±0.023' 或者数值里提取均值(float)。"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip().replace("＋", "+").replace("－", "-").replace("，", ",")
    parts = re.split(r"\s*±\s*|\s*\+\/-\s*|\s*\+-\s*|\s*\\\+\-\s*", s)
    # 先试第一个片段
    try:
        return float(parts[0])
    except Exception:
        # 回退到正则抓第一个数字
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(nums[0]) if nums else np.nan

def pick_metric_columns(df: pd.DataFrame):
    """
    根据常见关键词自动匹配六个指标列；返回 {标准名: 实际列名}
    """
    wanted = {
        "PR-AUC": ["pr-auc", "prauc", "ap", "average precision", "pr_auc"],
        "AU-ROC": ["au-roc", "auroc", "roc-auc", "auc", "roc_auc"],
        "F1":     ["f1"],
        "REC":    ["rec", "recall", "tpr", "sensitivity"],
        "PREC":   ["prec", "precision"],
        "ACC":    ["acc", "accuracy"]
    }
    colmap = {}
    cols_lower = [c.lower() for c in df.columns]
    for std, keys in wanted.items():
        hit = None
        for k in keys:
            for c in df.columns:
                if k in c.lower():
                    hit = c
                    break
            if hit: break
        if hit: colmap[std] = hit
    return colmap

def detect_label_column(df: pd.DataFrame):
    """
    识别标签名所在列：优先第1列；若列名中包含 label/tag/name/descriptor 也可。
    """
    first = df.columns[0]
    candidates = [c for c in df.columns
                  if any(k in c.lower() for k in ["label", "tag", "name", "descriptor", "category"])]
    return candidates[0] if candidates else first

def safe_filename(s: str) -> str:
    """
    将标签名转为安全文件名（去除特殊字符；空格->下划线）。
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = re.sub(r"[^\w\-\.\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    if not s:
        s = "label"
    return s[:150]  # 防过长

def draw_radar(values_dict, title, out_png, annotate=True):
    """
    画单张雷达图，保存 PNG 与 PDF。
    values_dict: {'PR-AUC':x, 'AU-ROC':x, 'F1':x, 'REC':x, 'PREC':x, 'ACC':x}
    """
    metrics_order = ["PR-AUC", "AU-ROC", "F1", "REC", "PREC", "ACC"]
    angles = np.linspace(0, 2*np.pi, len(metrics_order), endpoint=False).tolist()
    angles += angles[:1]

    vals = [values_dict.get(m, np.nan) for m in metrics_order]
    vals += vals[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_order, fontweight="bold")
    ax.set_rlabel_position(0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{t:.1f}" for t in [0.2, 0.4, 0.6, 0.8, 1.0]], fontweight="bold")

    ax.plot(angles, vals, linewidth=2.5, marker="o", label="Mean")
    ax.fill(angles, vals, alpha=0.10)

    if annotate:
        # 在每个顶点标数值
        for ang, v in zip(angles[:-1], vals[:-1]):
            if np.isfinite(v):
                ax.text(ang, min(max(v, 0.02), 0.98), f"{v:.2f}",
                        ha="center", va="center", fontsize=9, fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=10)
    ax.set_title(title, pad=20, fontweight="bold", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf

def draw_radar_multi(series_list, title, out_png,
                     annotate=ANNOTATE_VALUES_MULTI,
                     colors=MACARON_COLORS):
    """
    画多系列雷达图并保存 PNG/PDF。
    series_list: List[ (label_name, {'PR-AUC':..., 'AU-ROC':..., 'F1':..., 'REC':..., 'PREC':..., 'ACC':...}) ]
    """
    metrics_order = ["PR-AUC", "AU-ROC", "F1", "REC", "PREC", "ACC"]
    angles = np.linspace(0, 2*np.pi, len(metrics_order), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_order, fontweight="bold")
    ax.set_rlabel_position(0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{t:.1f}" for t in [0.2, 0.4, 0.6, 0.8, 1.0]], fontweight="bold")

    # 逐系列画线
    for idx, (lab, stats) in enumerate(series_list):
        col = colors[idx % len(colors)]
        vals = [stats.get(m, np.nan) for m in metrics_order]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=LINEWIDTH_MULTI, marker=MARKER_MULTI, label=lab, color=col)
        ax.fill(angles, vals, alpha=FILL_ALPHA_MULTI, color=col)

        if annotate:
            for ang, v in zip(angles[:-1], vals[:-1]):
                if np.isfinite(v):
                    ax.text(ang, min(max(v, 0.03), 0.97), f"{v:.2f}",
                            ha="center", va="center", fontsize=8, fontweight="bold", color=col)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.10), frameon=False, fontsize=9)
    ax.set_title(title, pad=20, fontweight="bold", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf

# ====== 主流程 ======
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # 识别列
    label_col = detect_label_column(df)
    metric_cols = pick_metric_columns(df)
    if len(metric_cols) == 0:
        raise ValueError(f"未识别到指标列，请检查 {CSV_PATH} 的列名：{list(df.columns)}")

    # 解析均值为数值
    means_table = pd.DataFrame()
    means_table[label_col] = df[label_col]
    for std_name, real_col in metric_cols.items():
        means_table[std_name] = df[real_col].map(extract_mean).astype(float)

    # 导出一个纯“均值表”方便核查
    means_csv = os.path.join(OUT_DIR, "per_label_means.csv")
    means_table.to_csv(means_csv, index=False, encoding="utf-8-sig")

    # 导出一个纯“均值表”方便核查
    means_csv = os.path.join(OUT_DIR, "per_label_means.csv")
    means_table.to_csv(means_csv, index=False, encoding="utf-8-sig")

    # ====== 新增：多标签同图的雷达图 ======
    if TARGET_LABELS:
        label_col_lower = means_table.columns[0]  # detect_label_column 已经放到第1列
        # 建立不区分大小写的索引
        name_to_row = {
            str(r[label_col]).strip().lower(): r
            for _, r in means_table.iterrows()
        }

        series_list = []
        not_found = []
        for name in TARGET_LABELS:
            key = str(name).strip().lower()
            if key in name_to_row:
                row = name_to_row[key]
                stats = {
                    "PR-AUC": row.get("PR-AUC", np.nan),
                    "AU-ROC": row.get("AU-ROC", np.nan),
                    "F1":     row.get("F1", np.nan),
                    "REC":    row.get("REC", np.nan),
                    "PREC":   row.get("PREC", np.nan),
                    "ACC":    row.get("ACC", np.nan),
                }
                # 显示用原始名字（保留大小写）
                orig_name = means_table.loc[row.name, label_col]
                series_list.append((str(orig_name), stats))
            else:
                not_found.append(name)

        if series_list:
            out_png_multi = os.path.join(OUT_DIR, "radar_selected_labels.png")
            title_multi = f"{MODEL_TITLE_PREFIX} — " + "/".join([s[0] for s in series_list])
            draw_radar_multi(series_list, title_multi, out_png_multi)
            print(f"[OK] 组合雷达图已保存：{out_png_multi}")
        if not_found:
            print("[warn] 未在CSV中找到这些标签：", not_found)


    # 逐标签画图
    total = len(means_table)
    for i, row in means_table.iterrows():
        label_name = str(row[label_col])
        stats = {
            "PR-AUC": row.get("PR-AUC", np.nan),
            "AU-ROC": row.get("AU-ROC", np.nan),
            "F1":     row.get("F1", np.nan),
            "REC":    row.get("REC", np.nan),
            "PREC":   row.get("PREC", np.nan),
            "ACC":    row.get("ACC", np.nan),
        }
        safe_label = safe_filename(label_name)
        out_png = os.path.join(OUT_DIR, f"radar_{safe_label}.png")
        title = f"{MODEL_TITLE_PREFIX} — {label_name}"
        draw_radar(stats, title, out_png, annotate=ANNOTATE_VALUES)
        print(f"[{i+1}/{total}] Saved: {out_png}")

    print(f"\n[OK] 全部完成。均值汇总表：{means_csv}")
    print(f"[OK] 雷达图保存在：{OUT_DIR}")

if __name__ == "__main__":
    main()
