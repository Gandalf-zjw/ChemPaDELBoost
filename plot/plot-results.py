# -*- coding: utf-8 -*-
"""
cv5_overall_sorted.py
从 cv_summary.json 读取 5 折结果，绘制一个整体的有序横向条形图：
- 所有指标按均值降序排列
- 条末标注 mean ± std
- 字体 Calibri + 加粗，SCI风格
- 输出 PNG+PDF（600dpi）
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ★ 新增：用于生成马卡龙渐变
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # ★新增
import matplotlib.colors as mcolors  # ★新增

# 路径（按需修改）
IN_JSON = r"..."
OUT_DIR = r"..."
OUT_FILE = "cv5_overall_metrics_sorted"

MODEL_NAME = "ChemBERTa + XGBoost"

# 样式
plt.rcParams["font.family"] = "Calibri"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# ★ 新增：马卡龙基础色（9色），再插值为任意数量
MACARON_COLORS = [
    "#FFB3BA",  # pink
    "#FFDFBA",  # apricot
    "#FFFFBA",  # lemon
    "#BAFFC9",  # mint
    "#BAE1FF",  # sky
    "#E0BBE4",  # lavender
    "#FFCCE5",  # peach pink
    "#B5EAD7",  # seafoam
    "#C7CEEA",  # periwinkle
]
macaron_cmap = LinearSegmentedColormap.from_list("macaron", MACARON_COLORS, N=512)  # ★新增

def darker(color, ratio=0.85):
    """把颜色稍微压暗，用于误差线 ecolor（让误差线与柱色协调）。"""
    r, g, b, a = to_rgba(color)
    return (r*ratio, g*ratio, b*ratio, a)

# 读取
with open(IN_JSON, "r", encoding="utf-8") as f:
    summary = json.load(f)

# 整理数据
metrics_map = {
    "micro_f1": "Micro-F1",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "micro_precision": "Micro-Precision",
    "micro_recall": "Micro-Recall",
    "macro_precision": "Macro-Precision",
    "macro_recall": "Macro-Recall",
    "balanced_accuracy_macro": "Balanced Acc (Macro)",
    "mcc_macro": "MCC (Macro)",
    "hamming_loss": "Hamming Loss",
    "jaccard_samples": "Jaccard (samples)",
    "lrap": "LRAP",
    "map_macro": "mAP (Macro)",
    "map_micro": "mAP (Micro)",
    "auc_macro": "AUC (Macro)",
    "auc_micro": "AUC (Micro)",
    "subset_accuracy": "Subset Accuracy",
}

rows = []
for k, disp in metrics_map.items():
    mean = float(summary[k]["mean"])
    std = float(summary[k]["std"])
    rows.append([disp, mean, std, f"{mean:.3f} ± {std:.3f}"])

# 额外添加 1 - Hamming Loss
hl_mean, hl_std = summary["hamming_loss"]["mean"], summary["hamming_loss"]["std"]
rows.append(["1 - Hamming Loss", 1 - hl_mean, hl_std, f"{1 - hl_mean:.3f} ± {hl_std:.3f}"])

df = pd.DataFrame(rows, columns=["Metric", "Mean", "Std", "Mean ± Std"])
df_sorted = df.sort_values("Mean", ascending=False).reset_index(drop=True)

# 绘图
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(df_sorted))

# ★ 修改：使用马卡龙渐变为每根柱子着色（按排序后的顺序均匀取色）
n = len(df_sorted)
if n > 1:
    colors = [macaron_cmap(i / (n - 1)) for i in range(n)]  # ★修改
else:
    colors = [macaron_cmap(0.5)]

# ★ 可选：让误差线颜色比柱子略深，保证可见性
ecolors = [darker(c, 0.80) for c in colors]  # ★新增

bars = ax.barh(
    y, df_sorted["Mean"],
    xerr=df_sorted["Std"],
    color=colors,
    edgecolor="white", linewidth=0.6,
    # ↓↓↓ 所有误差线样式统一放在 error_kw 里 ↓↓↓
    error_kw={"ecolor": "#666666", "elinewidth": 1.0, "capsize": 3}
)


ax.set_yticks(y)
ax.set_yticklabels(df_sorted["Metric"], fontweight="bold")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Score", fontweight="bold")
ax.set_title(f"{MODEL_NAME} — Overall metrics (5-fold mean ± std)", fontweight="bold", pad=12)
ax.grid(axis="x", linestyle=(0, (3, 3)), alpha=0.35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# 数值标注
for rect, m in zip(bars, df_sorted["Mean"]):
    ax.text(rect.get_width() + 0.03,  # 略收一点，避免超框
            rect.get_y() + rect.get_height()/2,
            f"{m:.3f}",
            va="center", ha="left", fontsize=9, fontweight="bold")

fig.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, OUT_FILE + ".png")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_png.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[OK] 已保存: {out_png} 和 PDF")
