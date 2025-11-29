#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-figure SCI plot:
- Read cv5/fold*_labelwise.csv
- Aggregate to label-level means across folds
- Split Head/Middle/Tail by support_pos (2-8-2)
- Plot AP/F1/MCC/Balanced Accuracy with bootstrap 95% CI
- Mann–Whitney U (Head vs Tail) with Holm–Bonferroni across metrics
- Draw significance stars ONLY between Head and Tail
- Calibri (bold), light/pastel colors, 600 dpi
"""

import os, glob, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ====== Paths (edit if needed) ======
INPUT_DIR  = r"...\results\cv5"
OUTPUT_DIR = r"..."
FIG_NAME   = "HMT_grouped_metrics_HEAD-vs-TAIL_only_light.png"
TABLE_NAME = "HMT_groupwise_summary_HEAD-vs-TAIL.csv"

# ====== Config ======
METRICS = ["ap", "f1", "mcc", "balanced_accuracy"]
METRIC_LABELS = {
    "ap": "AP",
    "f1": "F1",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced Accuracy"
}
HEAD_FRAC, TAIL_FRAC = 0.2, 0.2
N_BOOT, CI, SEED = 2000, 95, 42
np.random.seed(SEED)

# Fonts (Calibri + bold; will fallback if missing)
matplotlib.rcParams["font.family"] = "Calibri"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["xtick.labelsize"] = 11
matplotlib.rcParams["ytick.labelsize"] = 11
matplotlib.rcParams["legend.fontsize"] = 10

# Light / pastel colors (SCI-friendly, colorblind-safe inspired)
COLORS = {
    "Head":   "#A6CEE3",  # light blue
    "Middle": "#BDBDBD",  # light gray
    "Tail":   "#B2DF8A",  # light green
}

def significance_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def bootstrap_ci(values, n_boot=N_BOOT, ci=CI):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    n = v.shape[0]
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = v[np.random.randint(0, n, n)].mean()
    boots.sort()
    mean = float(v.mean())
    low  = float(np.percentile(boots, (100-ci)/2))
    high = float(np.percentile(boots, 100 - (100-ci)/2))
    return mean, low, high

# ---------- Read & aggregate ----------
csvs = sorted(glob.glob(os.path.join(INPUT_DIR, "fold*_labelwise.csv")))
if len(csvs) != 5:
    raise FileNotFoundError(f"Expect 5 fold files in {INPUT_DIR}, got {len(csvs)}")

df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

need = {"label","support_pos"} | set(METRICS)
miss = need - set(df.columns)
if miss:
    raise ValueError(f"Missing columns: {miss}")

# label-level means (across folds)
label_means = (df.groupby("label", as_index=False)
                 .agg({**{m:"mean" for m in METRICS}, "support_pos":"mean"}))

# ---------- Head/Middle/Tail split (2-8-2) ----------
label_means = label_means.sort_values("support_pos", ascending=False).reset_index(drop=True)
n = len(label_means)
head_n = max(1, int(round(HEAD_FRAC*n)))
tail_n = max(1, int(round(TAIL_FRAC*n)))
mid_start, mid_end = head_n, n - tail_n

head_set   = set(label_means.iloc[:head_n]["label"])
middle_set = set(label_means.iloc[mid_start:mid_end]["label"])
tail_set   = set(label_means.iloc[mid_end:]["label"])

def assign_group(lbl):
    if lbl in head_set: return "Head"
    if lbl in tail_set:  return "Tail"
    return "Middle"

label_means["group"] = label_means["label"].apply(assign_group)

# ---------- Bootstrap CI per group & metric ----------
summary_rows = []
vals = {"Head":{m:[] for m in METRICS},
        "Middle":{m:[] for m in METRICS},
        "Tail":{m:[] for m in METRICS}}

for g in ["Head","Middle","Tail"]:
    sub = label_means[label_means["group"]==g]
    for m in METRICS:
        arr = sub[m].to_numpy(float)
        vals[g][m] = arr
        mean, lo, hi = bootstrap_ci(arr, N_BOOT, CI)
        summary_rows.append({"group":g,"metric":m,"mean":mean,"ci_low":lo,"ci_high":hi})

summary = pd.DataFrame(summary_rows)

# ---------- Significance: Head vs Tail only (Holm–Bonferroni across metrics) ----------
pvals = {}
for m in METRICS:
    a = vals["Head"][m]; a = a[~np.isnan(a)]
    b = vals["Tail"][m]; b = b[~np.isnan(b)]
    if len(a)==0 or len(b)==0:
        pvals[m] = np.nan
    else:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        pvals[m] = p

# Holm–Bonferroni correction (across these metrics)
valid = [(m,p) for m,p in pvals.items() if not np.isnan(p)]
valid.sort(key=lambda x: x[1])
adj = {m: np.nan for m in METRICS}
m_tests = len(valid)
for i,(m,p) in enumerate(valid, start=1):
    adj[m] = min(p*(m_tests - i + 1), 1.0)

# ---------- Save summary table ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
pivot = summary.pivot(index="metric", columns="group", values="mean")
for b in ["ci_low","ci_high"]:
    pivot = pd.concat([pivot,
                       summary.pivot(index="metric", columns="group", values=b).add_suffix(f"_{b}")],
                      axis=1)
pivot["p_adj_Head_vs_Tail"] = [adj.get(m, np.nan) for m in pivot.index]
pivot.index.name = "metric"
pivot.to_csv(os.path.join(OUTPUT_DIR, TABLE_NAME), encoding="utf-8-sig")

# ---------- Plot (single figure) ----------
x = np.arange(len(METRICS))
bar_w = 0.24
offset = {"Head":-bar_w, "Middle":0.0, "Tail":bar_w}

fig, ax = plt.subplots(figsize=(8.6, 5.6))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.35, linewidth=0.6)

# bars + CI
for g in ["Head","Middle","Tail"]:
    means = [summary[(summary.group==g)&(summary.metric==m)]["mean"].values[0] for m in METRICS]
    los   = [summary[(summary.group==g)&(summary.metric==m)]["ci_low"].values[0] for m in METRICS]
    his   = [summary[(summary.group==g)&(summary.metric==m)]["ci_high"].values[0] for m in METRICS]
    yerr  = np.vstack([np.array(means)-np.array(los),
                       np.array(his)-np.array(means)])
    xpos  = x + offset[g]
    ax.bar(xpos, means, width=bar_w, label=g,
           color=COLORS[g], edgecolor="black", linewidth=0.6)
    ax.errorbar(xpos, means, yerr=yerr, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3, capthick=0.8)

# axes/labels
ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontweight="bold")
ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Group-wise Performance with 95% CI (Bootstrap)\nSignificance: Head vs Tail only", fontweight="bold")

# y-limit with headroom
ymax = float(np.nanmax(summary["ci_high"].values))
ax.set_ylim(0, min(1.0, ymax*1.12 + 0.02))

# legend
leg = ax.legend(title="Group", frameon=False)
if leg.get_title():
    leg.get_title().set_fontweight("bold")

# ---- draw ONLY Head vs Tail brackets per metric ----
def draw_bracket(x1, y1, x2, y2, ypad=0.014, ax=None):
    ax = ax or plt.gca()
    y = max(y1, y2) + ypad
    ax.plot([x1, x1, x2, x2], [y-0.004, y, y, y-0.004], color="black", linewidth=0.9)
    return y

for i, m in enumerate(METRICS):
    p = adj.get(m, np.nan)
    if np.isnan(p):
        continue
    # top of CI for Head and Tail at this metric
    x_head = (x + offset["Head"])[i]
    y_head = summary[(summary.group=="Head")&(summary.metric==m)]["ci_high"].values[0]
    x_tail = (x + offset["Tail"])[i]
    y_tail = summary[(summary.group=="Tail")&(summary.metric==m)]["ci_high"].values[0]
    y_line = draw_bracket(x_head, y_head, x_tail, y_tail, ypad=0.02, ax=ax)
    ax.text((x_head+x_tail)/2, y_line+0.008, significance_stars(p),
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# clean frame
for sp in ["top","right"]:
    ax.spines[sp].set_visible(False)

plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, FIG_NAME)
plt.savefig(out_fig, dpi=600)
plt.close()

print(f"[OK] Figure saved: {out_fig}")
print(f"[OK] Summary table: {os.path.join(OUTPUT_DIR, TABLE_NAME)}")
print("Adjusted p-values (Head vs Tail):")
for m in METRICS:
    print(f" - {METRIC_LABELS[m]:<18s} p_adj = {adj[m] if not np.isnan(adj[m]) else np.nan:.3g} ({significance_stars(adj[m]) if not np.isnan(adj[m]) else 'NA'})")
