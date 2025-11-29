# -*- coding: utf-8 -*-
"""
Odor Descriptor Map: Many Semantic Islands (MDS + KMeans + Ellipse)

流程：
1. 用 Z–label 相关矩阵表示每个标签在融合空间中的“指纹”；
2. 计算标签间 correlation 距离，做 classical MDS 得到 2D 坐标；
3. 在 2D 坐标上用 KMeans 聚成 K 个小簇（例如 K=14）；
4. 每个簇拟合一个椭圆（均值+协方差），画成淡色“气味岛”；
5. 每个岛自动挑一个代表标签作为岛名（黑底白字），所有标签点用 Calibri 粗体显示。

你之后可以把自动生成的 Cluster 名改成更语义化的名字。
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from adjustText import adjust_text
from matplotlib.patches import Ellipse

# ========= 路径 =========
BASE_DIR = r"..."
IN_PATH = os.path.join(BASE_DIR, "interpret", "Z_label_correlation_full.csv")
OUT_PNG = os.path.join(BASE_DIR, "interpret", "odor_islands_mds_kmeans.png")
OUT_PDF = os.path.join(BASE_DIR, "interpret", "odor_islands_mds_kmeans.pdf")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ========= 画图风格 =========
mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["font.size"] = 10
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# ========= 1. 读取 Z–label 相关矩阵 & MDS =========
corr_df = pd.read_csv(IN_PATH, index_col=0)  # 行=label，列=Z0..Zd-1
labels = corr_df.index.tolist()
X = corr_df.values
L, D = X.shape
print(f"Loaded corr matrix: labels={L}, dims={D}")

# 去掉方差几乎为 0 的维度
dim_std = X.std(axis=0)
keep = dim_std > 1e-3
X = X[:, keep]
print("After var filter dims:", X.shape[1])

# correlation 距离 + classical MDS
Dmat = pairwise_distances(X, metric="correlation")
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    max_iter=3000,
    n_init=20,
)
coords = mds.fit_transform(Dmat)  # [L, 2]

df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "label": labels,
})

# ========= 2. 在 MDS 平面上做聚类（更多“小类”） =========
K = 14  # 你可以试 12、14、16 等，看哪个视觉上最好
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(df[["x", "y"]].values)

print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())

# 给每个 cluster 起一个占位名字：Cluster 1,2,...（之后你可以手动改成 Harsh Freshness 等）
cluster_names = {c: f"Cluster {c+1}" for c in range(K)}

# ========= 3. 画椭圆“气味岛” =========
fig, ax = plt.subplots(figsize=(12, 12))

# “马卡龙”风格的 K 色调色板
palette = sns.color_palette("Set3", K)  # 也可以换成 "pastel", "husl" 等
cluster_colors = {c: palette[c] for c in range(K)}

ellipse_base_scale = 2.2  # 控制岛的整体大小，可调 1.8 ~ 2.5

for c in range(K):
    sub = df[df["cluster"] == c]
    if len(sub) < 2:
        continue

    pts = sub[["x", "y"]].values
    center = pts.mean(axis=0)

    # 协方差矩阵 -> 椭圆主轴
    cov = np.cov(pts.T)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        continue

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # 根据簇大小稍微缩放一下，点少的簇不至于太大
    scale = ellipse_base_scale * (len(sub) ** 0.15)

    width = 2.0 * scale * np.sqrt(max(vals[0], 1e-6))
    height = 2.0 * scale * np.sqrt(max(vals[1], 1e-6))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    color = cluster_colors[c]
    ell = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        lw=1.5,
        zorder=1,
    )
    ax.add_patch(ell)

    # 自动选一个“中心标签”作为该岛的名字（离中心最近的点）
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    idx_min = np.argmin(dists)
    rep_label = str(sub.iloc[idx_min]["label"]).upper()
    island_name = cluster_names[c]


# ========= 4. 画点 + 小标签 =========
sns.scatterplot(
    data=df,
    x="x", y="y",
    hue="cluster",
    palette=cluster_colors,
    s=35,
    edgecolor="white",
    linewidth=0.7,
    ax=ax,
    zorder=2,
    legend=False,  # 图例我们单独做
)

texts = []
for _, row in df.iterrows():
    texts.append(
        ax.text(
            row["x"], row["y"],
            str(row["label"]).upper(),
            fontfamily="Calibri",
            fontweight="bold",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            zorder=3,
        )
    )

adjust_text(
    texts,
    ax=ax,
    expand_points=(1.05, 1.05),
    expand_text=(1.1, 1.1),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
)

# ========= 5. 轴标题 & 图例 =========
ax.set_xlabel("MDS Dimension 1", fontsize=14, fontweight="bold")
ax.set_ylabel("MDS Dimension 2", fontsize=14, fontweight="bold")
ax.set_title("Odor descriptor map: semantic islands (MDS + KMeans)",
             fontsize=16, fontweight="bold", pad=18)

# 简单图例：每个簇一个小色块 + 名字
handles = []
labels_legend = []
for c in range(K):
    handles.append(
        plt.Line2D([], [], marker="o", linestyle="",
                   markerfacecolor=cluster_colors[c],
                   markeredgecolor="white",
                   markersize=8)
    )
    labels_legend.append(cluster_names[c])

sns.despine(offset=10)
ax.grid(False)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
